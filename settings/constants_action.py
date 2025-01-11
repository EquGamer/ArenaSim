import torch
from settings.constants_common import torch_device
from settings.constants_team import NUM_ROBOTS_PER_TEAM



## action
# dim_discrete_actions = 8

ACTION_NO_OP = 0
ACTION_STOP = 1
ACTION_MOVE_EAST = 2
ACTION_MOVE_SOUTH = 3
ACTION_MOVE_WEST = 4
ACTION_MOVE_NORTH = 5
ACTION_SHOOT_ENEMY_ZERO = 6
ACTION_SHOOT_ENEMY_ONE = 7
ACTION_SHOOT_ENEMY_TWO = 8
ACTION_SHOOT_ENEMY_THREE = 9
ACTION_SHOOT_ENEMY_FOUR = 10


n_actions_no_move = 2
n_actions_move = 4
n_actions_no_attack = n_actions_no_move + n_actions_move
n_actions_attack = NUM_ROBOTS_PER_TEAM
n_actions_total = n_actions_no_attack + n_actions_attack

## move
move_directions = torch.tensor([
    [0, 1], [1, 0], [0, -1], [-1, 0], 
], dtype=torch.int32, device=torch_device)


shoot_range = 2 



