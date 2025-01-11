import torch
import pandas as pd

from settings.constants_common import torch_device

## map
map_rows = 7
map_cols = 8
map_area = map_rows * map_cols

map_grid = torch.tensor(
    [
        [0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0]
    ],
    device=torch_device
)

map_resolution = 0.6
map_block_center = torch.tensor([-map_resolution / 2, map_resolution / 2], device=torch_device)

map_origin = torch.tensor([2.4, 2.1], device=torch_device)

df=pd.read_csv("settings/csv/dist.csv")
dist_map_p2p = torch.from_numpy(df.iloc[:,1:].to_numpy()).to(torch.float32).to(torch_device) 
obstruct_shoot_pairs_coord = torch.tensor([
    [[0, 5], [0, 7]],
    [[0, 0], [2, 0]],
    [[0, 3], [2, 3]],
    [[0, 3], [2, 4]],
    [[0, 4], [2, 3]],
    [[0, 4], [2, 4]],
    [[2, 1], [4, 1]],
    [[2, 1], [4, 2]],
    [[2, 2], [4, 1]],
    [[2, 2], [4, 2]],
    [[2, 5], [4, 5]],
    [[2, 5], [4, 6]],
    [[2, 6], [4, 5]],
    [[2, 6], [4, 6]],
    [[4, 3], [6, 3]],
    [[4, 3], [6, 4]],
    [[4, 4], [6, 3]],
    [[4, 4], [6, 4]],
    [[4, 7], [6, 7]],
    [[6, 0], [6, 2]],
], 
dtype=torch.int32,
device=torch_device)


