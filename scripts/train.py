import os 
import torch
from datetime import datetime

from utils.hydra_cfg.hydra_utils import *
from utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from learning.sp_agent import PPOAgent, MAPPOAgent, SPAgent, DUSPAgent_0, DUSPAgent_5, DUSPAgent_8, PFSPAgent

import hydra
from omegaconf import DictConfig

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from rl_games.common.algo_observer import AlgoObserver
from rl_games.algos_torch import torch_ext

from omni.isaac.gym.vec_env import VecEnvBase



class RLGPUAlgoObserver(AlgoObserver):
    """Allows us to log stats from the env along with the algorithm running stats. """

    def __init__(self):
        pass

    def after_init(self, algo):
        self.algo = algo
        self.mean_scores = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.ep_infos = []
        self.direct_info = {}
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        assert isinstance(infos, dict), "RLGPUAlgoObserver expects dict info"
        if isinstance(infos, dict):
            if 'episode' in infos:
                self.ep_infos.append(infos['episode'])

            if len(infos) > 0 and isinstance(infos, dict):  # allow direct logging from env
                self.direct_info = {}
                for k, v in infos.items():
                    # only log scalars
                    if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                        self.direct_info[k] = v

    def after_clear_stats(self):
        self.mean_scores.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.ep_infos:
            for key in self.ep_infos[0]:
                    infotensor = torch.tensor([], device=self.algo.device)
                    for ep_info in self.ep_infos:
                        # handle scalar and zero dimensional tensor infos
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.Tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, ep_info[key].to(self.algo.device)))
                    value = torch.mean(infotensor)
                    self.writer.add_scalar('Episode/' + key, value, epoch_num)
            self.ep_infos.clear()
        
        for k, v in self.direct_info.items():
            self.writer.add_scalar(f'{k}/frame', v, frame)
            self.writer.add_scalar(f'{k}/iter', v, epoch_num)
            self.writer.add_scalar(f'{k}/time', v, total_time)

        if self.mean_scores.current_size > 0:
            mean_scores = self.mean_scores.get_mean()
            self.writer.add_scalar('scores/mean', mean_scores, frame)
            self.writer.add_scalar('scores/iter', mean_scores, epoch_num)
            self.writer.add_scalar('scores/time', mean_scores, total_time)


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)

    def step(self, action):
        return  self.env.step(action)

    def reset(self):
        return self.env.reset()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        info['agents'] = self.get_number_of_agents()

        if self.env.num_states > 0:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info

    def get_action_masks(self):
        return self.env.get_action_masks()



class VecEnvRLGames(VecEnvBase):
    def _process_data(self):
        self._rew = self._rew.to(self._task.rl_device).clone()
        self._resets = self._resets.to(self._task.rl_device).clone()
        self._extras = self._extras.copy()

    def set_task(
        self, task, backend="numpy", sim_params=None, init_sim=True
    ) -> None:
        super().set_task(task, backend, sim_params, init_sim)
        self.num_states = self._task.num_states
        self.state_space = self._task.state_space

    def step(self, actions):
        if self._task.randomize_actions:
            actions = self._task._dr_randomizer.apply_actions_randomization(actions=actions, reset_buf=self._task.reset_buf)
        self._task.pre_physics_step(actions)
        for i in range(self._task.control_frequency_inv):
            if i == self._task.control_frequency_inv - 1:
                self._world.step(render=True)
            else:
                self._world.step(render=self._render)
            self.sim_frame_count += 1
        self._obs, self._rew, self._resets, self._extras = self._task.post_physics_step()
        if self._task.randomize_observations:
            self._obs = self._task._dr_randomizer.apply_observations_randomization(
                observations=self._obs.to(device=self._task.rl_device), reset_buf=self._task.reset_buf)
        self._states = self._task.get_states()
        self._process_data()
        obs_dict = {"obs": self._obs[0, ...], "obs_op": self._obs[1, ...], "states": self._states}
        if self._task._num_agents > 1:
            self._rew = self._rew.repeat(self._task._num_agents)
            self._resets = self._resets.repeat(self._task._num_agents)
        return obs_dict, self._rew, self._resets, self._extras


    def reset(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] Running RL reset")
        self._task.reset()
        actions = torch.ones((2, self._task._num_envs * self._task._num_agents, self._task.num_actions), dtype=torch.int64, device=self._task.rl_device)
        obs_dict, _, _, _ = self.step(actions)
        return obs_dict

    def get_action_masks(self):
        return self._task.get_action_masks()

    def get_number_of_agents(self):
        return self._task.get_number_of_agents()



class RLGTrainer():
    def __init__(self, cfg, cfg_dict):
        self.cfg = cfg
        self.cfg_dict = cfg_dict

    def launch_rlg_hydra(self, env):
        self.cfg_dict["task"]["test"] = self.cfg.test
        vecenv.register('RLGPU',
                        lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
        env_configurations.register('rlgpu', {
            'vecenv_type': 'RLGPU',
            'env_creator': lambda **kwargs: env
        })
        self.rlg_config_dict = omegaconf_to_dict(self.cfg.train)

    def run(self):
        runner = Runner(RLGPUAlgoObserver())
        runner.algo_factory.register_builder('ppo', lambda **kwargs: PPOAgent(**kwargs))
        runner.algo_factory.register_builder('mappo', lambda **kwargs: MAPPOAgent(**kwargs))
        runner.algo_factory.register_builder('sp', lambda **kwargs: SPAgent(**kwargs))
        runner.algo_factory.register_builder('pfsp', lambda **kwargs: PFSPAgent(**kwargs))
        runner.algo_factory.register_builder('dusp', lambda **kwargs: DUSPAgent_0(**kwargs))
        runner.algo_factory.register_builder('dusp_5', lambda **kwargs: DUSPAgent_5(**kwargs))
        runner.algo_factory.register_builder('dusp_8', lambda **kwargs: DUSPAgent_8(**kwargs))

        runner.load(self.rlg_config_dict)
        runner.reset()
        experiment_dir = os.path.join('runs', self.cfg.train.params.config.name)
        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        runner.run({
            'train': not self.cfg.test,
            'play': self.cfg.test,
            'sigma': None
        })


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    headless = cfg.headless
    env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id)
    from omni.isaac.core.utils.extensions import enable_extension
    if not headless:
        enable_extension("omni.kit.viewport.bundle")
    cfg_dict = omegaconf_to_dict(cfg)
    print("cfg_dict:")
    print_dict(cfg_dict)

    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed

    from learning.engine import RoboGame
    from utils.config_utils.sim_config import SimConfig
    sim_config = SimConfig(cfg_dict)
    task = RoboGame(name=cfg.task_name, sim_config=sim_config, env=env)
    env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch")

    if cfg.wandb_activate:
        import wandb
        run_name = f"{cfg.wandb_name}_{time_str}"
        wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            config=cfg_dict,
            sync_tensorboard=True,
            id=run_name,
            resume="allow",
            monitor_gym=True,
        )

    rlg_trainer = RLGTrainer(cfg, cfg_dict)
    rlg_trainer.launch_rlg_hydra(env)
    rlg_trainer.run()
    env.close()

    if cfg.wandb_activate:
        wandb.finish()

if __name__ == '__main__':
    parse_hydra_configs()
