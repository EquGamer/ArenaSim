import torch
from settings.constants_team import NUM_ROBOTS_PER_TEAM
from settings.constants_action import move_directions, n_actions_total
from settings.constants_map import map_rows, map_cols, dist_map_p2p
from settings.constants_health import full_health

from settings.map_transform import coordinate_to_index

# only for 2v2
def rule_based_strategy(origin_obs, origin_masks, siege_health=False):
    MAX_HEALTH = 100
    MAX_DIS = 100
    num_actors = (origin_masks.shape[0] * origin_masks.shape[1]) // (2 * n_actions_total)
    num_agents = origin_masks.shape[0] // num_actors
    device = origin_masks.device
    obs = origin_obs[:num_actors, 0:12] 
    coordinate_x = (obs[:, 0:4] * map_rows).int()
    coordinate_y = (obs[:, 4:8] * map_cols).int()
    enemy_health = (obs[:, 10:12] * full_health).int()

    coordinate_ego0 = torch.stack((coordinate_x[:, 0], coordinate_y[:, 0]), dim=-1)
    coordinate_ego1 = torch.stack((coordinate_x[:, 1], coordinate_y[:, 1]), dim=-1)
    coordinate_enemy0 = torch.stack((coordinate_x[:, 2], coordinate_y[:, 2]), dim=-1)
    coordinate_enemy1 = torch.stack((coordinate_x[:, 3], coordinate_y[:, 3]), dim=-1)
    coord_ego = torch.stack((coordinate_ego0, coordinate_ego1), dim=0)
    
    relative_coordinate0 = torch.stack([coordinate_enemy0 - coordinate_ego0, coordinate_enemy1 - coordinate_ego0])
    relative_coordinate1 = torch.stack([coordinate_enemy0 - coordinate_ego1, coordinate_enemy1 - coordinate_ego1])
    relative_coordinate  = torch.stack([relative_coordinate0, relative_coordinate1])
    relative_manhattan_coord_distance = torch.abs(relative_coordinate[..., 0]) + torch.abs(relative_coordinate[..., 1]) 

    if siege_health:
        enemy_health = torch.where(enemy_health == 0, MAX_HEALTH, enemy_health)
        least_enemy_health = torch.min(enemy_health, dim=1, keepdim=True).values
        target_enemy_mask = (least_enemy_health == enemy_health) 
        target_enemy_idx0 = torch.multinomial(target_enemy_mask.float(), num_samples=1) 
        target_enemy_idx1 = torch.multinomial(target_enemy_mask.float(), num_samples=1) 
        target_enemy_idx0 = torch.where(((enemy_health[:, 0] == 10) & (enemy_health[:, 1] == 10)).unsqueeze(-1), 1, target_enemy_idx0)
        target_enemy_idx1 = torch.where(((enemy_health[:, 0] == 10) & (enemy_health[:, 1] == 10)).unsqueeze(-1), 1, target_enemy_idx1)
    else:
        relative_manhattan_coord_distance[:, 0, :] = torch.where(enemy_health[:, 0].unsqueeze(0) == 0, MAX_DIS, relative_manhattan_coord_distance[:, 0, :])
        relative_manhattan_coord_distance[:, 1, :] = torch.where(enemy_health[:, 1].unsqueeze(0) == 0, MAX_DIS, relative_manhattan_coord_distance[:, 1, :])
        nearest_enemy_dis = torch.min(relative_manhattan_coord_distance, dim=1, keepdim=True).values
        target_enemy_mask = (nearest_enemy_dis == relative_manhattan_coord_distance)
        target_enemy_idx0 = torch.multinomial(target_enemy_mask[0].permute(1, 0).float(), num_samples=1)
        target_enemy_idx1 = torch.multinomial(target_enemy_mask[1].permute(1, 0).float(), num_samples=1)

    if num_agents == 1:
        mask_ego0 = origin_masks[:, 0:8]
        mask_ego1 = origin_masks[:, 8:16]
    else:
        mask_ego0 = origin_masks[:num_actors, :]
        mask_ego1 = origin_masks[num_actors:, :]

    actions = torch.ones((num_actors, 2), dtype=torch.int64, device=device)
    actions[:, 0] = torch.where(mask_ego0[:, 6], 6, actions[:, 0])
    actions[:, 0] = torch.where(mask_ego0[:, 7], 7, actions[:, 0])
    actions[:, 1] = torch.where(mask_ego1[:, 6], 6, actions[:, 1])
    actions[:, 1] = torch.where(mask_ego1[:, 7], 7, actions[:, 1])
    coordinate_target_enemy0 = torch.where(target_enemy_idx0 == 0, coordinate_enemy0, coordinate_enemy1)
    coordinate_target_enemy1 = torch.where(target_enemy_idx1 == 0, coordinate_enemy0, coordinate_enemy1)
    mask_move_ego = torch.stack((mask_ego0[:, 2:6], mask_ego1[:, 2:6]), dim=0) 
    coord_target_enemy = torch.stack((coordinate_target_enemy0, coordinate_target_enemy1), dim=0) 
    index_target_enemy = coordinate_to_index(coord_target_enemy).unsqueeze(-1) 
    index_ego = coordinate_to_index(coord_ego).unsqueeze(-1) 
    dist_target_enemy = dist_map_p2p[index_target_enemy, index_ego].squeeze(-1)
    coord_neighbors = coord_ego.unsqueeze(-2) + move_directions 
    coord_neighbors[..., 0] = coord_neighbors[..., 0].clamp(0, map_rows-1)
    coord_neighbors[..., 1] = coord_neighbors[..., 1].clamp(0, map_cols-1)
    index_neighbors = coordinate_to_index(coord_neighbors) 
    dist_target_enemy_neighbors = dist_map_p2p[index_target_enemy, index_neighbors]
    dist_target_enemy_neighbors = torch.where(mask_move_ego, dist_target_enemy_neighbors, MAX_DIS)
    min_dist, min_indices = torch.min(dist_target_enemy_neighbors, dim=-1, keepdim=False) 
    actions[:, 0:2] = torch.where((min_dist < dist_target_enemy).permute(1, 0), 2 + min_indices.permute(1, 0), actions[:, 0:2])

    target_enemy_flag_ego0 = mask_ego0[:, 6:8].gather(1, target_enemy_idx0).squeeze(1)
    target_enemy_flag_ego1 = mask_ego1[:, 6:8].gather(1, target_enemy_idx1).squeeze(1)
    actions[:, 0] = torch.where(target_enemy_flag_ego0, 6+target_enemy_idx0.squeeze(1), actions[:, 0])
    actions[:, 1] = torch.where(target_enemy_flag_ego1, 6+target_enemy_idx1.squeeze(1), actions[:, 1])
    actions[:, 0] = torch.where(mask_ego0[:, 0], 0, actions[:, 0])
    actions[:, 1] = torch.where(mask_ego1[:, 0], 0, actions[:, 1])

    if num_agents == 1:
        return actions
    else:
        actions = torch.cat((actions[:, 0], actions[:, 1]), dim=0)
        return actions


def random_strategy(origin_obs, origin_masks):
    num_actors = (origin_masks.shape[0] * origin_masks.shape[1]) // (NUM_ROBOTS_PER_TEAM * n_actions_total)
    num_agents = origin_masks.shape[0] // num_actors
    device = origin_masks.device

    masks = torch.zeros(num_actors, NUM_ROBOTS_PER_TEAM, n_actions_total)
    if num_agents == 1:
        for r in range(NUM_ROBOTS_PER_TEAM):
            masks[:, r, :] = origin_masks[:, n_actions_total * r:n_actions_total * (r+1)]
    else:
        for r in range(NUM_ROBOTS_PER_TEAM):
            masks[:, r, :] = origin_masks[num_actors * r:num_actors * (r+1), :]


    random_actions = torch.ones((num_actors, NUM_ROBOTS_PER_TEAM), dtype=torch.int64, device=device)
    for e in range(num_actors):
        for r in range(NUM_ROBOTS_PER_TEAM):
            avail_actions = torch.nonzero(masks[e, r, :], as_tuple=False).squeeze(-1)
            chosen_action = avail_actions[torch.randint(0, avail_actions.shape[0], (1,))]
            random_actions[e, r] = chosen_action

    if num_agents == 1:
        return random_actions
    else:
        return random_actions.t().reshape(-1)

