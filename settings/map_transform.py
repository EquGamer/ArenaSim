import torch

from settings.constants_map import map_rows, map_cols, map_origin, map_resolution, map_block_center

def translation_to_coordinate(translation):
    trans_origin = torch.cat((map_origin[1] + translation[..., 1:], map_origin[0] - translation[..., 0:1]), dim=-1)
    coord = (trans_origin // map_resolution).int()
    coord[..., 0] = coord[..., 0].clamp(0, map_rows-1)
    coord[..., 1] = coord[..., 1].clamp(0, map_cols-1)
    return coord

def coordinate_to_translation(coords):
    trans_origin = coords * map_resolution
    return torch.cat((map_origin[0] - trans_origin[..., 1:], trans_origin[..., 0:1] - map_origin[1]), dim=-1) + map_block_center

def coordinate_to_index(coords):
    return coords[..., 0] * map_cols + coords[..., 1]

def index_to_coordinate(index):
    return torch.cat((index // map_cols, index % map_cols), dim=-1)
