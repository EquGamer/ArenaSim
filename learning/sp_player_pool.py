import torch


class SinglePlayer:
    def __init__(self, player_idx, model, device, obs_batch_len=0, decay=0.995):
        self.model = model
        self.player_idx = player_idx
        self._games = torch.tensor(0, device=device, dtype=torch.float)
        self._wins = torch.tensor(0, device=device, dtype=torch.float)
        self._loses = torch.tensor(0, device=device, dtype=torch.float)
        self._draws = torch.tensor(0, device=device, dtype=torch.float)
        self._total_games = torch.tensor(0, device=device, dtype=torch.float)
        self._decay = decay
        self._has_env = torch.zeros((obs_batch_len,), device=device, dtype=torch.bool)
        self.device = device
        self.env_indices = torch.tensor([], device=device, dtype=torch.long, requires_grad=False)

    def __call__(self, input_dict):
        return self.model(input_dict)

    def reset_envs(self):
        self.env_indices = self._has_env.nonzero(as_tuple=True)

    def remove_envs(self, env_indices):
        self._has_env[env_indices] = False

    def add_envs(self, env_indices):
        self._has_env[env_indices] = True

    def clear_envs(self):
        self.env_indices = torch.tensor([], device=self.device, dtype=torch.long, requires_grad=False)

    def remove_all_envs(self):
        self._has_env.fill_(0)

    def update_metric(self, wins, loses, draws):
        win_count = torch.sum(wins[self.env_indices])
        lose_count = torch.sum(loses[self.env_indices])
        draw_count = torch.sum(draws[self.env_indices])
        for stats in (self._games, self._wins, self._loses, self._draws):
            stats *= self._decay
        self._games += win_count + lose_count + draw_count
        self._wins += win_count
        self._loses += lose_count
        self._draws += draw_count
        self._total_games += win_count + lose_count + draw_count

    def clear_metric(self):
        self._games = torch.tensor(0, device=self.device, dtype=torch.float)
        self._wins = torch.tensor(0, device=self.device, dtype=torch.float)
        self._loses = torch.tensor(0, device=self.device, dtype=torch.float)
        self._draws = torch.tensor(0, device=self.device, dtype=torch.float)
        self._total_games = torch.tensor(0, device=self.device, dtype=torch.float)

    def win_rate(self):
        if self.model is None:
            raise RuntimeError
        elif self._games == 0:
            return torch.tensor(0.5, device=self.device, dtype=torch.float)
        return (self._wins + 0.5 * self._draws) / self._games

    def games_num(self):
        return self._games
    
    def total_games_num(self):
        return self._total_games


class BasePlayerPool:
    def __init__(self, max_length, device, num_envs):
        assert max_length > 0
        self.players = []
        self.max_length = max_length
        self.device = device
        self.num_envs = num_envs

    def add_player(self, player):
        if len(self.players) < self.max_length:
            self.players.append(player)
        else:
            print("too many envs")
            raise RuntimeError

    def update_player_metric(self, infos):
        for player in self.players:
            player.update_metric(infos['win'], infos['lose'], infos['draw'])

    def clear_player_metric(self):
        for player in self.players:
            player.clear_metric()

    def inference(self, input_dict, res_dict, processed_obs, action_masks):
        for i, player in enumerate(self.players):
            if len(player.env_indices[0]) == 0:
                continue
            input_dict['obs'] = processed_obs[player.env_indices]
            input_dict['action_masks'] = action_masks[player.env_indices]
            out_dict = player(input_dict)
            for key in res_dict:
                if key == "logits":
                    for i in range(len(res_dict[key])):
                        res_dict[key][i][player.env_indices] = out_dict[key][i]
                else: 
                    res_dict[key][player.env_indices] = out_dict[key]

    def sample_player(self):
        raise NotImplementedError


