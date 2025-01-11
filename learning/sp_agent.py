import copy
import numpy as np
import os
import time
import random
import torch
from rl_games.algos_torch import a2c_discrete
from rl_games.common.a2c_common import swap_and_flatten01, print_statistics
from learning.strategies import rule_based_strategy, random_strategy
from learning.sp_player_pool import SinglePlayer, BasePlayerPool


class PPOAgent(a2c_discrete.DiscreteA2CAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self.base_model_config = {
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'num_seqs': self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        print("num_agents", self.num_agents)
        print("actions_num", self.actions_num)
        print("obs_shape", self.obs_shape)

        self.now = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))

        self.sp_dir = os.path.join(self.experiment_dir, self.__class__.__name__)
        os.makedirs(self.sp_dir, exist_ok=True)
        self.seed_dir = os.path.join(self.sp_dir, str(params['seed']))
        os.makedirs(self.seed_dir, exist_ok=True)
        self.reward_dir = os.path.join(self.seed_dir, 'reward_record')
        os.makedirs(self.reward_dir, exist_ok=True)
        self.nn_dir = os.path.join(self.seed_dir, 'nn')
        os.makedirs(self.nn_dir, exist_ok=True)
        self.summaries_dir = os.path.join(self.seed_dir, 'summaries')
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.now_update_steps = 0
        self._decay = 0.995
        self.epoch_mean_reward = torch.zeros((1,), dtype=torch.float32, device=self.device)
        self.update_op_num = 0
        self.is_deterministic = False
        self.swap_camp = True

        self.reset_metric()


    def play_steps(self):
        update_list = self.update_list
        step_time = 0.0
        self.epoch_mean_reward[:] = 0
        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                masks_ego = torch.where(self.env_camp.repeat(self.num_agents, 1), masks[0, ...], masks[1, ...])
                masks_op = torch.where(self.env_camp.repeat(self.num_agents, 1), masks[1, ...], masks[0, ...])

                res_dict = self.get_masked_action_values(self.obs, masks_ego)
                res_dict_op = self.get_masked_action_values(self.obs, masks_op, is_op=True)
            else:
                res_dict = self.get_action_values(self.obs)
                res_dict_op = self.get_action_values(self.obs, is_op=True)

            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.perf_counter()

            actions_ego = res_dict['actions'].reshape(self.num_actors * self.num_agents, -1)
            actions_op = res_dict_op['actions'].reshape(self.num_actors * self.num_agents, -1)
            actions_red = torch.where(self.env_camp.repeat(self.num_agents, 1), actions_ego, actions_op)
            actions_blue = torch.where(self.env_camp.repeat(self.num_agents, 1), actions_op, actions_ego)
            actions = torch.stack((actions_red, actions_blue), dim=0)
            self.obs, rewards, self.dones, infos = self.env_step(actions)
            self.epoch_mean_reward += rewards.mean()
            step_time_end = time.perf_counter()
            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()
            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            self.all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = self.all_done_indices[::self.num_agents]
     
            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(self.current_shaped_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            self.update_metric_step(infos=infos)

        last_values = self.get_values(self.obs)
        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time
        return batch_dict

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions)
        obs_ego = torch.where(self.env_camp.repeat(self.num_agents, 1), obs['obs'], obs['obs_op'])
        obs_op = torch.where(self.env_camp.repeat(self.num_agents, 1), obs['obs_op'], obs['obs'])
        obs['obs'] = obs_ego
        obs['obs_op'] = obs_op
        rewards = torch.where(self.env_camp.repeat(self.num_agents, 1).squeeze(), rewards, -rewards)
        print("reward: {}".format(rewards.mean()))

        win_train = torch.where(self.env_camp.squeeze(), infos["win"], infos["lose"])
        lose_train = torch.where(self.env_camp.squeeze(), infos["lose"], infos["win"])
        infos["win"] = win_train
        infos["lose"] = lose_train
    
        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(
                dones).to(self.ppo_device), infos

    def env_reset(self):
        obs = self.vec_env.reset()
        obs = self.obs_to_tensors(obs)
        self.redistribute_camp()
        obs_ego = torch.where(self.env_camp.repeat(self.num_agents, 1), obs['obs'], obs['obs_op'])
        obs_op = torch.where(self.env_camp.repeat(self.num_agents, 1), obs['obs_op'], obs['obs'])
        obs['obs'] = obs_ego
        obs['obs_op'] = obs_op
        return obs

    def redistribute_camp(self):
        if self.swap_camp:
            self.env_camp = torch.randint(2, (self.num_actors, 1), device=self.device, dtype=torch.bool)
        else:
            self.env_camp = torch.ones((self.num_actors, 1), device=self.device, dtype=torch.bool)


    def train(self):
        self.init_tensors()
        self.mean_rewards = self.last_mean_rewards = -100500
        start_time = time.perf_counter()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()

        init_checkpoint_name = self.config['name'] + '_ep_' + str(0) + '_rew_' + str(0)
        self.save(os.path.join(self.nn_dir, 'last_' + init_checkpoint_name))

        while True:
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
            self.writer.add_scalar('epoch_mean_reward/epoch_num', self.epoch_mean_reward, epoch_num)
            self.dataset.update_values_dict(None)
            total_time += sum_time
            curr_frames = self.curr_frames
            self.frame += curr_frames
            should_exit = False
            if self.global_rank == 0:
                self.diagnostics.epoch(self, current_epoch = epoch_num)
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                frame = self.frame // self.num_agents
                print_statistics(self.print_stats, curr_frames, step_time, scaled_play_time, scaled_time, 
                                epoch_num, self.max_epochs, frame, self.max_frames)
                self.write_stats(total_time, epoch_num, step_time, play_time, update_time,
                                a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, 
                                scaled_time, scaled_play_time, curr_frames)
                self.algo_observer.after_print_stats(frame, epoch_num, total_time)
                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)
                        self.writer.add_scalar('shaped_' + rewards_name + '/step'.format(i), mean_shaped_rewards[i], frame)
                        self.writer.add_scalar('shaped_' + rewards_name + '/iter'.format(i), mean_shaped_rewards[i], epoch_num)
                        self.writer.add_scalar('shaped_' + rewards_name + '/time'.format(i), mean_shaped_rewards[i], total_time)
                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)
                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])
                    if self.save_freq > 0:
                        if epoch_num % self.save_freq == 0:
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))
                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))
                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Maximum reward achieved. Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True
                if epoch_num >= self.max_epochs:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf
                    self.save(os.path.join(self.nn_dir, self.config['name'] + '_maxep'))
                    print('MAX EPOCHS NUM!')
                    should_exit = True
                self.update_metric()
                update_time = 0
            if should_exit:
                return self.last_mean_rewards, epoch_num
            
    def update_metric(self):
        self.now_update_steps += 1
        win_rate = self._wins / self._games
        print("total_games_num: ", self._total_games.item())
        print("win_rate: ", win_rate.item())
    
    def get_action_values(self, obs, is_op=False):
        processed_obs = self._preproc_obs(obs['obs_op'] if is_op else obs['obs'])
        if not is_op:
            self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': processed_obs,
            'rnn_states': self.rnn_states
        }
        with torch.no_grad():
            if not is_op:
                res_dict = self.model(input_dict)

            else:
                res_dict = {}
                res_dict['actions'] = 2 * torch.rand((self.num_actors, self.actions_num), device=self.device) - 1
                res_dict['mus'] = res_dict['actions']

            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states': states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        return res_dict
    

    def get_masked_action_values(self, obs, action_masks, is_op=False):
        processed_obs = self._preproc_obs(obs['obs_op'] if is_op else obs['obs'])
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'action_masks' : action_masks,
            'rnn_states' : self.rnn_states
        }
        with torch.no_grad():
            if not is_op:
                res_dict = self.model(input_dict)
            else:
                res_dict = {}
                res_dict['actions'] = random_strategy(processed_obs, action_masks)
                # res_dict['actions'] = rule_based_strategy(processed_obs, action_masks)
            if self.has_central_value:
                input_dict = {
                    'is_train': False,
                    'states' : obs['states'],
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value

        res_dict['action_masks'] = action_masks
        return res_dict


    def create_model(self):
        model = self.network.build(self.base_model_config)
        model.to(self.ppo_device)
        return model
    
    def update_metric_step(self, infos):
        win_count = torch.sum(infos['win'])
        lose_count = torch.sum(infos['lose'])
        draw_count = torch.sum(infos['draw'])

        for stats in (self._games, self._wins, self._loses, self._draws):
            stats *= self._decay

        self._games += win_count + lose_count + draw_count
        self._wins += win_count
        self._loses += lose_count
        self._draws += draw_count

        self._total_games += win_count + lose_count + draw_count

    def reset_metric(self):
        self._games = torch.tensor(0, device=self.device, dtype=torch.float)
        self._wins = torch.tensor(0, device=self.device, dtype=torch.float)
        self._loses = torch.tensor(0, device=self.device, dtype=torch.float)
        self._draws = torch.tensor(0, device=self.device, dtype=torch.float)
        self._total_games = torch.tensor(0, device=self.device, dtype=torch.float)

    def random_avail_actions_shoot_first(self, masks):
        return  7 - torch.argmax(masks.reshape(self.num_actors, 2, -1).int().flip(dims=[-1]), dim=-1)


class MAPPOAgent(PPOAgent):
    def __init__(self, base_name, params):
        params['network']['space'] = {'discrete': None} 
        params['model']['name'] = 'discrete_a2c'  

        super().__init__(base_name, params)


class SPBaseAgent(PPOAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self.players_dir = os.path.join(self.seed_dir, 'policy_dir')
        os.makedirs(self.players_dir, exist_ok=True)

        self.update_op_num = 0
        self.update_win_rate = params['update_win_rate']
        self.games_to_check = params['games_to_check']
        self.player_pool_length = params['player_pool_length']
        self.player_winrate = list()

        self.player_pool = BasePlayerPool(max_length=self.player_pool_length, device=self.device, num_envs=self.num_actors)
        player_model = self.copy_training_model()
        self.update_player_pool(player_model, player_idx=self.update_op_num)
        self.resample_op(torch.arange(end=self.num_actors, device=self.device, dtype=torch.long))

    def copy_training_model(self):
        player_model = self.create_model()
        player_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        if hasattr(self.model, 'running_mean_std'):
            player_model.running_mean_std.load_state_dict(copy.deepcopy(self.model.running_mean_std.state_dict()))
        player_model.eval()
        return player_model

    def update_player_pool(self, player_model, player_idx):
        print("update_player_pool!")
        player = SinglePlayer(player_idx, player_model, self.device, self.num_actors, self._decay)
        self.player_pool.add_player(player)
        self.player_winrate = [player.win_rate() for player in self.player_pool.players]

    def resample_op(self, resample_indices):
        for player in self.player_pool.players:
            player.remove_envs(resample_indices)
                
        for env_idx in resample_indices:
            player = self.sample_player()
            player.add_envs(env_idx)

        for player in self.player_pool.players:
            player.reset_envs()


    def update_metric_step(self, infos):
        self.player_pool.update_player_metric(infos=infos)
        self.resample_op(self.all_done_indices.flatten())

    def update_metric(self):
        self.now_update_steps += 1
        sum_win_count = 0
        sum_games_num = 0
        sum_tot_games_num = 0
        for player in self.player_pool.players:
            win_rate = player.win_rate()
            games = player.games_num()
            tot_games = player.total_games_num()
            sum_win_count += win_rate * games
            sum_games_num += games
            sum_tot_games_num += tot_games
        ave_win_rate = sum_win_count / sum_games_num
        print("total_games_num: {0}/{1}".format(sum_tot_games_num.item(), self.games_to_check))
        print("win_num/games_num: {0}/{1}".format(sum_win_count.item(), sum_games_num.item()))
        print("win_rate: {0}/{1}  update_steps: {2}".format(ave_win_rate, self.update_win_rate, self.now_update_steps))
        if sum_tot_games_num > self.games_to_check and ave_win_rate > self.update_win_rate:
            self.update_opponent()
            self.obs = self.env_reset()

    def get_action_values(self, obs, is_op=False):
        processed_obs = self._preproc_obs(obs['obs_op'] if is_op else obs['obs'])
        if not is_op:
            self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': processed_obs,
            'rnn_states': self.rnn_states
        }
        with torch.no_grad():
            if not is_op:
                res_dict = self.model(input_dict)
            else:
                res_dict = {
                    "actions": torch.zeros((self.num_actors, self.actions_num),
                                           device=self.device),
                    "mus": torch.zeros((self.num_actors, self.actions_num),
                                           device=self.device),
                    "values": torch.zeros((self.num_actors, 1), device=self.device)
                }
                self.player_pool.inference(input_dict, res_dict, processed_obs)

            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states': states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        return res_dict
    
    def get_masked_action_values(self, obs, action_masks, is_op=False):
        processed_obs = self._preproc_obs(obs['obs_op'] if is_op else obs['obs'])
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'action_masks' : action_masks,
            'rnn_states' : self.rnn_states
        }
        with torch.no_grad():
            if not is_op:
                res_dict = self.model(input_dict)
            else:
                res_dict = {
                    "neglogpacs": torch.zeros((self.num_actors), dtype=torch.float32, device=self.ppo_device),
                    "values": torch.zeros((self.num_actors, 1), dtype=torch.float32, device=self.ppo_device),
                    "actions": torch.zeros((self.num_actors, len(self.actions_num)), dtype=torch.int64, device=self.ppo_device),
                    "logits": [torch.zeros((self.num_actors, n_action), dtype=torch.float32, device=self.ppo_device) for n_action in self.actions_num],
                }
                self.player_pool.inference(input_dict, res_dict, processed_obs, action_masks)
                res_dict['rnn_states'] = None
            if self.has_central_value:
                input_dict = {
                    'is_train': False,
                    'states' : obs['states'],
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value

        res_dict['action_masks'] = action_masks
        return res_dict

    def update_opponent(self):
        self.update_op_num += 1
        self.save(os.path.join(self.players_dir, f'policy_{self.update_op_num}'))
        print('add opponent to player pool')
        self.now_update_steps = 0
        player_model = self.copy_training_model()
        self.update_player_pool(player_model, player_idx=self.update_op_num)
        self.resample_op(torch.arange(end=self.num_actors, device=self.device, dtype=torch.long))
        for player in self.player_pool.players:
            print("player{} with win-rate: {}/{}, was distributed to {} envs ".format(player.player_idx, player.win_rate(), player.games_num(), player.env_indices[0].shape[0]), end='')
            print(player.env_indices[0].data)
        self.player_pool.clear_player_metric()

    def sample_player(self):
        raise NotImplementedError


class SPAgent(SPBaseAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)

    def sample_player(self):
        player = self.player_pool.players[-1]
        return player


class DUSPAgent_0(SPBaseAgent):
    def __init__(self, base_name, params):
        self.delta = 0.0
        super().__init__(base_name, params)

    def sample_player(self):
        sample_upper = len(self.player_pool.players)
        sample_lower = int(self.delta * sample_upper)
        rand_idx = np.random.randint(sample_lower, sample_upper)
        player = self.player_pool.players[rand_idx]
        return player


class DUSPAgent_5(SPBaseAgent):
    def __init__(self, base_name, params):
        self.delta = 0.5
        super().__init__(base_name, params)
        
    def sample_player(self):
        sample_upper = len(self.player_pool.players)
        sample_lower = int(self.delta * sample_upper)
        rand_idx = np.random.randint(sample_lower, sample_upper)
        player = self.player_pool.players[rand_idx]
        return player


class DUSPAgent_8(SPBaseAgent):
    def __init__(self, base_name, params):
        self.delta = 0.8
        super().__init__(base_name, params)
        
    def sample_player(self):
        sample_upper = len(self.player_pool.players)
        sample_lower = int(self.delta * sample_upper)
        rand_idx = np.random.randint(sample_lower, sample_upper)
        player = self.player_pool.players[rand_idx]
        return player


class PFSPAgent(SPBaseAgent):
    def __init__(self, base_name, params):
        self.weightings = {
            "variance": lambda x: x * (1 - x),
            "linear": lambda x: 1 - x,
            "squared": lambda x: (1 - x) ** 2,
        }
        self.weight_func = self.weightings["linear"]
        super().__init__(base_name, params)

    def sample_player(self):
        self.sample_weights = [self.weight_func(winrate) for winrate in self.player_winrate]
        player = random.choices(self.player_pool.players, weights=self.sample_weights)[0]
        return player





