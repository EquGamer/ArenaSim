from tasks.rl_task import RLTask
from assets.robot import GameBot, Field

from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.torch.rotations import get_euler_xyz, quat_from_euler_xyz
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.objects import DynamicCuboid, DynamicSphere, VisualSphere
from omni.isaac.core.prims import RigidPrimView, GeometryPrimView
from omni.physx import get_physx_scene_query_interface
from omni.debugdraw import get_debug_draw_interface
import omni.ui as ui
from pxr import UsdGeom, Gf, Vt, Sdf


from settings.constants_team import NUM_ROBOTS_PER_TEAM
from settings.constants_action import (
    ACTION_NO_OP,
    ACTION_STOP,
    ACTION_MOVE_EAST,
    ACTION_MOVE_SOUTH,
    ACTION_MOVE_WEST,
    ACTION_MOVE_NORTH,
)

from settings.constants_action import move_directions, shoot_range, n_actions_no_move ,n_actions_no_attack, n_actions_total
from settings.constants_health import full_health
from settings.constants_map import map_rows, map_cols, map_grid, obstruct_shoot_pairs_coord
from settings.map_transform import translation_to_coordinate, coordinate_to_translation, coordinate_to_index, index_to_coordinate
import numpy as np
import torch
import math
import time
import os
from datetime import datetime
from gym import spaces

COLOR_RED = 0xffff0000
COLOR_GREEN = 0xff00ff00
COLOR_BLUE = 0xff0000ff
COLOR_YELLOW = 0xffffff00
COLOR_CYAN = 0xff00ffff


class RoboGame(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        
        print("RoboGame Init")

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]
        self._clipObservations = self._task_cfg["env"]["clipObservations"]
        self._battle_vis = self._task_cfg["env"].get("battle_vis", False)
        self._dt = self._task_cfg["sim"]["dt"]
        self.randomize_actions = self._task_cfg["env"].get("randomize_actions", False)
        self.randomize_observations = self._task_cfg["env"].get("randomize_observations", False)
        self.hit_reward_factor = self._task_cfg["env"].get("hitRewardFactor")
        self.kill_reward_factor = self._task_cfg["env"].get("killRewardFactor")
        self.game_reward_factor = self._task_cfg["env"].get("gameRewardFactor")
        self.enable_multi_agent = self._cfg["train"]["params"].get("enableMultiAgent", False)

        self.team_names = ['red', 'blue']
        self.num_teams = len(self.team_names)
        self.rpt = NUM_ROBOTS_PER_TEAM
        self.robot_name = [f"{self.team_names[team_index]}{robot_index}" for team_index in range(self.num_teams) for robot_index in range(self.rpt)]
        print("*" * 50, "\n", "robot_name", self.robot_name,  "\n", "*" * 50)
        
        if not self.enable_multi_agent:
            self._num_agents = 1
            self._num_observations = self.num_teams * self.rpt * 3
            self._num_actions = self.rpt
            self.action_space = spaces.Tuple([spaces.Discrete(n_actions_total)] * self._num_actions)
        else:
            self._num_agents = self.rpt
            self._num_observations = self.rpt * (self.num_teams * 3 + 1)
            self._num_actions = 1
            self.action_space = spaces.Discrete(n_actions_total)

        if self._battle_vis:
            self.laser_render = True
        else:
            self.laser_render = False
        if self.laser_render:
            self.init_debugdraw()

        RLTask.__init__(self, name, env)

        self._wheel_radius = 0.07
        self._wheel_base = 0.4
        self._axle_base = 0.414
        self._base = (self._wheel_base + self._axle_base) / 2
        self.maxRobotSpeed = 1.5
        self.maxWheelSpeed = self.maxRobotSpeed / self._wheel_radius
        self.gimbal_pos_vel_scale = 2 / math.pi
        self.alpha_vel = 0.8
        self.smooth_vel = True
        self.resolving_matrix = torch.tensor([[1, -1, 1, -1],
                                              [1, 1, -1, -1],
                                              [1, 1, 1, 1]], device=self._device)
        self.vel_factor = torch.tensor([[self.maxRobotSpeed], [self.maxRobotSpeed], [self.maxRobotSpeed]], device=self._device)
        self.vel_matrix = self.resolving_matrix * self.vel_factor / self._wheel_radius
        self.joint_indices = None

        self.aim_decimation = 2
        self.move_decimation = 8
        self.control_decimation = 6
        self.coord_reach_dis = 0.1
        self.max_coord = 100
        self.max_robot_distance = 100.0
        self.field_half_length = 2.6
        self.field_half_width = 2.3

        self.obstruct_shoot_pairs_index = coordinate_to_index(obstruct_shoot_pairs_coord).sort(dim=-1).values
        self.id_obs = torch.eye(self.rpt, dtype=torch.float32, device=self._device).unsqueeze(1).repeat(1, self._num_envs, 1)

        self.robot_poses = {
            'red0': [[-2.1,  1.8, 0], [1, 0, 0, 0]],
            'red1': [[-2.1, -1.8, 0], [1, 0, 0, 0]],
            'red2': [[-2.1,  0.0, 0], [1, 0, 0, 0]],
            'red3': [[-1.5,  0.9, 0], [1, 0, 0, 0]],
            'red4': [[-1.5, -0.9, 0], [1, 0, 0, 0]],

            'blue0':[[ 2.1, -1.8, 0], [0, 0, 0, 1]],
            'blue1':[[ 2.1,  1.8, 0], [0, 0, 0, 1]],
            'blue2':[[ 2.1,  0.0, 0], [0, 0, 0, 1]],
            'blue3':[[ 1.5, -0.9, 0], [0, 0, 0, 1]],
            'blue4':[[ 1.5,  0.9, 0], [0, 0, 0, 1]],
        }

        self.goalpoint_color = {
            'red0': np.array([255/255, 0/255,    0/255]),
            'red1': np.array([255/255, 64/255,   0/255]),
            'red2': np.array([139/255, 0/255,  139/255]),
            'red3': np.array([255/255, 0/255,   64/255]),
            'red4': np.array([255/255, 68/255,  79/255]),

            'blue0':np.array([  0/255, 0/255,   255/255]),
            'blue1':np.array([  0/255, 192/255, 255/255]),
            'blue2':np.array([ 72/255, 209/255, 204/255]),
            'blue3':np.array([ 87/255, 0/255,   255/255]),
            'blue4':np.array([ 109/255, 52/255, 204/255]),
        }

        self.pre_time = time.time()
        print("RoboGame Init Done")
        return
    

    def set_up_scene(self, scene) -> None:
        print("set_up_scene")
        self._stage = get_current_stage()
        for name in self.robot_name:
            self.get_gamebot(name, self.robot_poses[name][0], self.robot_poses[name][1])
            self.get_goalpoint(f"goalpoint_{name}", self.goalpoint_color[name])
        self.get_field()
        super().set_up_scene(scene)

        self._robot_views = [[ArticulationView(prim_paths_expr=f"/World/envs/.*/{team}{idx}", name=f"{team}{idx}_view", reset_xform_properties=False) for idx in range(self.rpt)] for team in self.team_names]
        self._goal_views = [[GeometryPrimView(prim_paths_expr=f"/World/envs/.*/goalpoint_{team}{idx}", name=f"goalpoint_{team}{idx}_view", reset_xform_properties=False) for idx in range(self.rpt)] for team in self.team_names]
        for t in range(self.num_teams):
            for r in range(self.rpt):
                scene.add(self._robot_views[t][r])
        for t in range(self.num_teams):
            for r in range(self.rpt):
                scene.add(self._goal_views[t][r])
        print("set_up_scene done")

        return

    def get_gamebot(self, name, translation, orientation):
        gamebot = GameBot(
            prim_path=f"{self.default_zero_env_path}/{name}", 
            name=name, 
            translation=translation, 
            orientation=orientation
        )
        self._sim_config.apply_articulation_settings(
            name, 
            get_prim_at_path(gamebot.prim_path), 
            self._sim_config.parse_actor_config('robot')
        )
        if self.joint_indices is None:
            self.joint_indices = gamebot._wheel_joint_idx.to(self._device)

    def get_goalpoint(self, name, color):
        radius = 0.05
        goalpoint = VisualSphere(
            prim_path=f"{self.default_zero_env_path}/{name}", 
            translation=np.array([0.0, 0.0, 0.0]),
            name=name,
            radius=radius,
            color=color,
            visible=False,
        )

    def get_field(self):
        self.field = Field(
            prim_path=f"{self.default_zero_env_path}/field", 
            name="field", 
        )
        self.barriers_pos = [block[0:4] for block in self.field.blocks[4:]]
  
    def get_observations(self) -> dict:
        for t in range(self.num_teams):
            for r in range(self.rpt):
                self.robot_pos[t, r, ...], self.robot_rot[t, r, ...] = self._robot_views[t][r].get_world_poses(clone=False)
                self.robot_roll[t, r, :], _, self.robot_yaw[t, r, :] = get_euler_xyz(self.robot_rot[t, r, ...])

        self.robot_translation = (self.robot_pos - self._env_pos)[..., 0:2]
        self.robot_coordinate = translation_to_coordinate(self.robot_translation)

        common_obs_red = torch.hstack((
            self.robot_coordinate[0, ..., 0].permute(1,0) / map_rows,
            self.robot_coordinate[1, ..., 0].permute(1,0) / map_rows,
            self.robot_coordinate[0, ..., 1].permute(1,0) / map_cols,
            self.robot_coordinate[1, ..., 1].permute(1,0) / map_cols,
            self.robot_health[0, ...].permute(1,0) / full_health,
            self.robot_health[1, ...].permute(1,0) / full_health
        ))

        common_obs_blue = torch.hstack((
            self.robot_coordinate[1, ..., 0].permute(1,0) / map_rows,
            self.robot_coordinate[0, ..., 0].permute(1,0) / map_rows,
            self.robot_coordinate[1, ..., 1].permute(1,0) / map_cols,
            self.robot_coordinate[0, ..., 1].permute(1,0) / map_cols,
            self.robot_health[1, ...].permute(1,0) / full_health,
            self.robot_health[0, ...].permute(1,0) / full_health
        ))

        if not self.enable_multi_agent:
            self.obs_buf[0, :, :] = common_obs_red
            self.obs_buf[1, :, :] = common_obs_blue
        else: 
            for r in range(self.rpt):
                self.obs_buf[0, self._num_envs * r:self._num_envs * (r+1), :] = torch.hstack((common_obs_red, self.id_obs[r]))
                self.obs_buf[1, self._num_envs * r:self._num_envs * (r+1), :] = torch.hstack((common_obs_blue, self.id_obs[r]))

        observations = {
            "robogame": {
                "obs_buf": self.obs_buf
            }
        }
        
        return observations

    def can_move(self, delta):
        new_coord = self.robot_coordinate + delta
        in_bounds = self.check_bounds(new_coord)
        valid_coord = new_coord.clone()
        valid_coord[~in_bounds] = 0
        return in_bounds & self.check_blocks(valid_coord) & self.check_robots(new_coord)

    def check_bounds(self, coord):
        coord_x = coord[..., 0]
        coord_y = coord[..., 1]
        return (coord_x >= 0) & (coord_x < map_rows) & (coord_y >= 0) & (coord_y < map_cols)

    def check_blocks(self, coord):
        coord_x = coord[..., 0]
        coord_y = coord[..., 1]
        return map_grid[coord_x, coord_y] == 0
    
    def check_robots(self, coord):
        equal = (coord.unsqueeze(2).unsqueeze(3) == self.robot_coordinate.unsqueeze(0).unsqueeze(1))
        return ~equal.all(-1).any(2).any(2)

    def can_shoot(self):
        return self.check_range() & self.check_obstruct() & self.check_alive()

    def check_range(self):
        for i in range(self.rpt):
            for j in range(self.rpt):
                self.mutual_coord[i, j, ...] = self.robot_coordinate[1, j, ...] - self.robot_coordinate[0, i, ...]
        self.mutual_manhattan_coord_distance[...] = torch.abs(self.mutual_coord[..., 0]) + torch.abs(self.mutual_coord[..., 1])
        within_range = torch.zeros((self.num_teams, self.rpt, self._num_envs, self.rpt), dtype=torch.bool, device=self._device)

        for i in range(self.rpt):
            for j in range(self.rpt):
                within_range[1, j, :, i] = within_range[0, i, :, j] = (self.mutual_manhattan_coord_distance[i, j, :] <= shoot_range)

        return within_range
    
    def check_obstruct(self):
        return self.check_obstruct_blocks() & self.check_obstruct_robots()
    
    def check_obstruct_blocks(self):
        robot_index = coordinate_to_index(self.robot_coordinate) 
        robot_index_pairs = torch.zeros((self.rpt, self.rpt, self._num_envs, 2), dtype=torch.int32, device=self._device)
        for i in range(self.rpt):
            for j in range(self.rpt):
                robot_index_pairs[i, j, ...] = torch.stack([robot_index[0, i, :], robot_index[1, j, :]], dim=-1).sort(dim=-1).values

        matches = (robot_index_pairs.unsqueeze(3) == self.obstruct_shoot_pairs_index).all(dim=-1)
        result = ~(matches.any(dim=-1))
        not_obstructed = torch.zeros((self.num_teams, self.rpt, self._num_envs, self.rpt), dtype=torch.bool, device=self._device)
        for i in range(self.rpt):
            for j in range(self.rpt):
                not_obstructed[1, j, :, i] = not_obstructed[0, i, :, j] = result[i, j, :]
        return not_obstructed
    

    def check_obstruct_robots(self):
        x_coord = self.robot_coordinate[..., 0]
        y_coord = self.robot_coordinate[..., 1]

        x_equal = (x_coord.unsqueeze(2).unsqueeze(3) == x_coord.unsqueeze(0).unsqueeze(1))
        y_equal = (y_coord.unsqueeze(2).unsqueeze(3) == y_coord.unsqueeze(0).unsqueeze(1))
        x_greater = (x_coord.unsqueeze(2).unsqueeze(3) > x_coord.unsqueeze(0).unsqueeze(1))
        y_greater = (y_coord.unsqueeze(2).unsqueeze(3) > y_coord.unsqueeze(0).unsqueeze(1))
        x_less = (x_coord.unsqueeze(2).unsqueeze(3) < x_coord.unsqueeze(0).unsqueeze(1)) 
        y_less = (y_coord.unsqueeze(2).unsqueeze(3) < y_coord.unsqueeze(0).unsqueeze(1)) 

        mutual_obstruct_x = torch.zeros((self.rpt, self.rpt, self._num_envs), dtype=torch.bool, device=self._device)
        mutual_obstruct_y = torch.zeros((self.rpt, self.rpt, self._num_envs), dtype=torch.bool, device=self._device)
        not_obstructed = torch.zeros((self.num_teams, self.rpt, self._num_envs, self.rpt), dtype=torch.bool, device=self._device)
        for i in range(self.rpt):
            for j in range(self.rpt):
                mutual_obstruct_x[i, j, :] = (x_equal[0, i, ...] & x_equal[1, j, ...] & ((y_greater[0, i, ...] & y_less[1, j, ...]) | (y_greater[1, j, ...] & y_less[0, i, ...]))).any(0).any(0) 
                mutual_obstruct_y[i, j, :] = (y_equal[0, i, ...] & y_equal[1, j, ...] & ((x_greater[0, i, ...] & x_less[1, j, ...]) | (x_greater[1, j, ...] & x_less[0, i, ...]))).any(0).any(0) 
                not_obstructed[1, j, :, i] = not_obstructed[0, i, :, j] =  ~(mutual_obstruct_x[i, j, :] | mutual_obstruct_y[i, j, :])
        return not_obstructed


    def check_alive(self):
        enemy_alive = torch.zeros((self.num_teams, self.rpt, self._num_envs, self.rpt), dtype=torch.bool, device=self._device)
        for i in range(self.rpt):
            for j in range(self.rpt):
                enemy_alive[0, i, :, j] = self.robot_alive[1, j, :]
                enemy_alive[1, i, :, j] = self.robot_alive[0, j, :]
        return enemy_alive

    def get_action_masks(self):
        """Update the available actions."""
        self.avail_actions[..., ACTION_NO_OP] = ~self.robot_alive
        self.avail_actions[..., ACTION_STOP] = True
        self.avail_actions[..., ACTION_MOVE_EAST] = self.can_move(move_directions[0])
        self.avail_actions[..., ACTION_MOVE_SOUTH] = self.can_move(move_directions[1])
        self.avail_actions[..., ACTION_MOVE_WEST] = self.can_move(move_directions[2])
        self.avail_actions[..., ACTION_MOVE_NORTH] = self.can_move(move_directions[3])
        self.avail_actions[..., n_actions_no_attack:n_actions_total] = self.can_shoot()
        self.avail_actions[..., 1:] = torch.where(self.avail_actions[..., 0:1], False, self.avail_actions[..., 1:])

        if not self.enable_multi_agent:
            return self.avail_actions.permute(0,2,1,3).flatten(2,3)
        else:
            return self.avail_actions.flatten(1,2) 

    def robot_priority(self, team_id, robot_id):
        return robot_id * self.num_teams + team_id

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            print("not playing")
            raise RuntimeError
        
        cur_time = time.time()
        print("fps sim step: {:.0f}".format((self.aim_decimation + self.move_decimation) * self.control_decimation * self._num_envs / (cur_time - self.pre_time)))
        self.pre_time = cur_time

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.clone().to(self._device)
        if not self.enable_multi_agent:
            actions_red = actions[0, :, :]
            actions_blue = actions[1, :, :]
        else:
            actions_red =  torch.hstack([actions[0, r * (self._num_envs):(r+1) * self._num_envs] for r in range(self.rpt)])
            actions_blue = torch.hstack([actions[1, r * (self._num_envs):(r+1) * self._num_envs] for r in range(self.rpt)])

        robot_actions = torch.stack([actions_red.T, actions_blue.T], dim=0)
        self.avail_actions[..., reset_env_ids, 0] = False
        self.avail_actions[..., reset_env_ids, 1:] = True
        robot_actions[..., reset_env_ids] = ACTION_STOP
        robot_actions_unsqueezed = robot_actions.unsqueeze(-1)

        for r in range(self.rpt):
            self.aim_enemy = torch.where(robot_actions == n_actions_no_attack + r, r, self.aim_enemy)
        for r in range(self.rpt):
            self.aim_enemy[0, :, :] = torch.where(~self.robot_alive[1, r, :].unsqueeze(0), (r + 1) % self.rpt, self.aim_enemy[0, :, :])
            self.aim_enemy[1, :, :] = torch.where(~self.robot_alive[0, r, :].unsqueeze(0), (r + 1) % self.rpt, self.aim_enemy[1, :, :])

        delta_coord = torch.zeros((self.num_teams, self.rpt, self._num_envs, 2), dtype=torch.int32, device=self._device)
        for i in range(move_directions.shape[0]):
            delta_coord = torch.where((robot_actions_unsqueezed == i + n_actions_no_move), move_directions[i], delta_coord)

        self.goal_coord = self.robot_coordinate + delta_coord

        same_goal_mask = (self.goal_coord.unsqueeze(2).unsqueeze(3) == self.goal_coord.unsqueeze(0).unsqueeze(1)).all(dim=-1) 
        conflict_indices = same_goal_mask.nonzero()
        for t0, r0, t1, r1, e in conflict_indices:
            if self.robot_priority(t0, r0) < self.robot_priority(t1, r1):
                self.goal_coord[t1, r1, e, :] = self.robot_coordinate[t1, r1, e, :]
            elif self.robot_priority(t0, r0) > self.robot_priority(t1, r1):
                self.goal_coord[t0, r0, e, :] = self.robot_coordinate[t0, r0, e, :]

        self.goal_translation[..., 0:2] = coordinate_to_translation(self.goal_coord)
        self.goal_translation[self.killed_ids] = 0
        self.execute_goal_set()

        target_translation = coordinate_to_translation(self.robot_coordinate)
        for _ in range(self.aim_decimation):
            self.update_gimbal_angle()
            self.update_chassis_vel(target_translation)

            self.execute_chassis_vel()
            self.execute_gimbal_position()

            for j in range(self.control_decimation):
                SimulationContext.step(self._env._world, render=self._env._render)
            self.refresh_robot_state()

        self.update_gimbal_angle()
        self.shoot(robot_actions)

        target_translation = self.goal_translation[..., 0:2].clone()
        for _ in range(self.move_decimation):
            self.update_gimbal_angle()
            self.update_chassis_vel(target_translation)

            self.execute_chassis_vel()
            self.execute_gimbal_position()

            for j in range(self.control_decimation):
                SimulationContext.step(self._env._world, render=self._env._render)
            self.refresh_robot_state()


    def refresh_robot_state(self):
        for t in range(self.num_teams):
            for r in range(self.rpt):
                self.robot_pos[t, r, ...], self.robot_rot[t, r, ...] = self._robot_views[t][r].get_world_poses(clone=False)
                self.robot_roll[t, r, :], _, self.robot_yaw[t, r, :] = get_euler_xyz(self.robot_rot[t, r, ...])
        self.robot_translation = (self.robot_pos - self._env_pos)[..., 0:2]

    def execute_goal_set(self):
        for t in range(self.num_teams):
            for r in range(self.rpt):
                self._goal_views[t][r].set_world_poses(self.goal_translation[t, r, ...] + self._env_pos)


    def execute_chassis_vel(self):
        self.wheel_dof_vel = torch.matmul(self.chassis_vel, self.vel_matrix)
        max_dof_vel, _ = torch.max(torch.abs(self.wheel_dof_vel), dim=3)
        scale_dof_vel = torch.where(max_dof_vel > self.maxWheelSpeed, self.maxWheelSpeed / max_dof_vel, 1.0).unsqueeze(3)
        self.wheel_dof_vel = self.wheel_dof_vel * scale_dof_vel

        for t in range(self.num_teams):
            for r in range(self.rpt):
                self._robot_views[t][r].set_joint_velocities(self.wheel_dof_vel[t, r, ...], joint_indices=self.joint_indices)

    def execute_gimbal_position(self):
        for t in range(self.num_teams):
            for r in range(self.rpt):
                self._robot_views[t][r].set_joint_position_targets(self.gimbal_position_targets[t, r, ...], joint_indices=self.gimbal_dof_idx)

    def update_chassis_vel(self, target_translation):
        rel_translation = target_translation - self.robot_translation 
        rel_distance = torch.norm(rel_translation, dim=-1, keepdim=True)
        self.global_vel_dir = torch.atan2(rel_translation[..., 1], rel_translation[..., 0])
        self.chassis_vel[..., 0] = torch.cos(self.global_vel_dir - self.robot_yaw)
        self.chassis_vel[..., 1] = torch.sin(self.global_vel_dir - self.robot_yaw)
        self.chassis_vel[..., 0:2] = torch.where(rel_distance < self.coord_reach_dis, 0, self.chassis_vel[..., 0:2])
        if self.smooth_vel:
            self.chassis_vel[..., 0:2] = self.alpha_vel * self.chassis_vel[..., 0:2] + (1 - self.alpha_vel) * self.last_chassis_vel[..., 0:2]

        self.chassis_vel[..., 2] = (self.gimbal_position_targets[..., 0] * self.gimbal_pos_vel_scale)
        self.chassis_vel[self.killed_ids] = 0
        self.last_chassis_vel = self.chassis_vel.clone()

    def update_gimbal_angle(self):
        for i in range(self.rpt):
            for j in range(self.rpt):
                self.mutual_position[i, j, ...] = self.robot_translation[1, j, ...] - self.robot_translation[0, i, ...]
        self.mutual_orientation = torch.atan2(self.mutual_position[..., 1], self.mutual_position[..., 0])
        self.mutual_distance = torch.norm(self.mutual_position, dim=-1)

        for r in range(self.rpt):
            self.shoot_distance[0, r, :] = (self.mutual_distance[r, :, :].gather(0, self.aim_enemy[0, r, :].unsqueeze(0)))[0, :]
            self.shoot_distance[1, r, :] = (self.mutual_distance[:, r, :].gather(0, self.aim_enemy[1, r, :].unsqueeze(0)))[0, :]
            self.shoot_orientation[0, r, :] = (self.mutual_orientation[r, :, :].gather(0, self.aim_enemy[0, r, :].unsqueeze(0)))[0, :]
            self.shoot_orientation[1, r, :] = (self.mutual_orientation[:, r, :].gather(0, self.aim_enemy[1, r, :].unsqueeze(0)))[0, :]

        self.shoot_orientation[1, :, :] -= math.pi
        self.shoot_orientation = self.shoot_orientation - 2 * math.pi * torch.floor((self.shoot_orientation + math.pi) / (2 * math.pi))
        self.gimbal_position_targets[..., 0] = self.shoot_orientation - self.robot_yaw
        self.gimbal_position_targets = self.gimbal_position_targets - 2 * math.pi * torch.floor((self.gimbal_position_targets + math.pi) / (2 * math.pi))
        self.gimbal_position_targets[:] = tensor_clamp(self.gimbal_position_targets, self.gimbal_dof_lower_limits, self.gimbal_dof_upper_limits)
        self.gimbal_position_targets[self.killed_ids] = 0


    def shoot(self, robot_actions):
        health_delta = torch.zeros((self.num_teams, self.rpt, self._num_envs), dtype=torch.int32, device=self._device)
        for r in range(self.rpt):
            health_delta[0, r, :] = torch.sum((robot_actions[1, :, :] == n_actions_no_attack + r).int(), dim=0)
            health_delta[1, r, :] = torch.sum((robot_actions[0, :, :] == n_actions_no_attack + r).int(), dim=0)
        self.robot_health = self.robot_health - health_delta
        self.robot_health = torch.where(self.robot_health < 0, 0, self.robot_health)
        self.robot_alive = (self.robot_health > 0)
        self.killed_ids = torch.nonzero(~self.robot_alive, as_tuple=True)

        damage_delta = torch.zeros((self.num_teams, self.rpt, self._num_envs), dtype=torch.int32, device=self._device)
        for t in range(self.num_teams):
            for r in range(self.rpt):
                damage_delta[t, r, :] = (robot_actions[t, r, :] >= n_actions_no_attack).int()
        self.damage_dealt = self.damage_dealt + damage_delta

        if self.laser_render:
            self.color_recovery()
            shoot_ready = robot_actions >= n_actions_no_attack
            self.render_shoot(shoot_ready)

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)  
        self.dof_pos[env_ids, :] = 0.0
        self.dof_vel[env_ids, :] = 0.0

        for t in range(self.num_teams):
            for r in range(self.rpt):
                self._robot_views[t][r].set_joint_positions(self.dof_pos[env_ids], indices=indices)
                self._robot_views[t][r].set_joint_velocities(self.dof_vel[env_ids], indices=indices)
                self._robot_views[t][r].set_world_poses(self.initial_pos[t, r, env_ids].clone(), self.initial_rot[t, r, env_ids].clone(), indices=indices)
                self._robot_views[t][r].set_velocities(self.default_vel[env_ids], indices=indices)
                self._goal_views[t][r].set_world_poses(self.initial_goal_pos[t, r, env_ids].clone(), indices=indices)

        self.robot_coordinate[..., env_ids, :] = self.initial_robot_coordinate[..., env_ids, :].clone()
        self.robot_health[..., env_ids] = full_health
        self.robot_alive[..., env_ids] = True
        self.last_robot_health[..., env_ids] = full_health
        self.last_robot_alive[..., env_ids] = True
        self.last_chassis_vel[..., env_ids, :] = 0.0
        self.damage_dealt[..., env_ids] = 0
        self.killed_ids = torch.nonzero(~self.robot_alive, as_tuple=True)

        if self.laser_render:
            self.reset_color(env_ids)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0


    def post_reset(self): 
        self.dof_pos = self._robot_views[0][0].get_joint_positions() 
        self.dof_vel = self._robot_views[0][0].get_joint_velocities()
        self.robot_num_dof = self._robot_views[0][0].num_dof  
        self.gimbal_dof_idx = self._robot_views[0][0].get_dof_index("gimbal_joint") 
        self.dof_limits = self._robot_views[0][0].get_dof_limits()  
        self.gimbal_dof_lower_limits = self.dof_limits[0, self.gimbal_dof_idx, 0].to(device=self._device)
        self.gimbal_dof_upper_limits = self.dof_limits[0, self.gimbal_dof_idx, 1].to(device=self._device)
        self.gimbal_dof_idx = torch.tensor(self.gimbal_dof_idx, device=self._device)

        # pose
        self.initial_pos = torch.zeros((self.num_teams, self.rpt, self._num_envs, 3), dtype=torch.float32, device=self._device)
        self.initial_rot = torch.zeros((self.num_teams, self.rpt, self._num_envs, 4), dtype=torch.float32, device=self._device)
        for t in range(self.num_teams):
            for r in range(self.rpt):
                self.initial_pos[t, r, ...], self.initial_rot[t, r, ...] = self._robot_views[t][r].get_world_poses()
        self.robot_pos = self.initial_pos.clone()
        self.robot_rot = self.initial_rot.clone()
        self.robot_roll = torch.zeros((self.num_teams, self.rpt, self._num_envs), dtype=torch.float32, device=self._device)
        self.robot_yaw = torch.zeros((self.num_teams, self.rpt, self._num_envs), dtype=torch.float32, device=self._device)
        self.robot_translation = torch.zeros((self.num_teams, self.rpt, self._num_envs, 2), dtype=torch.float32, device=self._device)
        self.robot_translation = (self.robot_pos - self._env_pos)[..., 0:2]
        self.initial_robot_coordinate = torch.zeros((self.num_teams, self.rpt, self._num_envs, 2), dtype=torch.int32, device=self._device)
        self.initial_robot_coordinate[...] = translation_to_coordinate(self.robot_translation)
        self.robot_coordinate = self.initial_robot_coordinate.clone()
        
        # goal
        self.initial_goal_pos = torch.zeros((self.num_teams, self.rpt, self._num_envs, 3), dtype=torch.float32, device=self._device)
        for t in range(self.num_teams):
            for r in range(self.rpt):
                self.initial_goal_pos[t, r, ...], _ = self._goal_views[t][r].get_world_poses()
        self.goal_translation = torch.ones((self.num_teams, self.rpt, self._num_envs, 3), dtype=torch.float32, device=self._device) * 0.06
        self.goal_coord = torch.zeros((self.num_teams, self.rpt, self._num_envs, 2), dtype=torch.int32, device=self._device)

        # vel
        self.default_vel = torch.zeros((self._num_envs, 6), device=self._device)
        self.velocities_idx = torch.tensor([0, 1, 5], dtype=torch.int64)
        self.chassis_vel = torch.zeros((self.num_teams, self.rpt, self._num_envs, 3), dtype=torch.float32, device=self._device)
        self.last_chassis_vel = torch.zeros((self.num_teams, self.rpt, self._num_envs, 3), dtype=torch.float32, device=self._device)
        self.wheel_dof_vel = torch.zeros((self.num_teams, self.rpt, self._num_envs, 4), dtype=torch.float32, device=self._device)
        self.global_vel_dir = torch.zeros((self.num_teams, self.rpt, self._num_envs), dtype=torch.float32, device=self._device)

       # gimbal
        self.aim_enemy = torch.zeros((self.num_teams, self.rpt, self._num_envs), dtype=torch.int64, device=self._device)
        self.shoot_orientation = torch.zeros((self.num_teams, self.rpt, self._num_envs), dtype=torch.float32, device=self._device)
        self.shoot_distance = torch.ones((self.num_teams, self.rpt, self._num_envs), dtype=torch.float32, device=self._device) * self.max_robot_distance
        self.gimbal_position_targets = torch.zeros((self.num_teams, self.rpt, self._num_envs, 1), dtype=torch.float32, device=self._device)
        self.gimbal_position = torch.zeros((self.num_teams, self.rpt, self._num_envs, 1), dtype=torch.float32, device=self._device)

        # action
        self.avail_actions = torch.ones((self.num_teams, self.rpt, self._num_envs, n_actions_total), dtype=torch.bool, device=self._device)
        self.avail_actions[..., 0] = False

        # health
        self.robot_health = torch.ones((self.num_teams, self.rpt, self._num_envs), dtype=torch.int32, device=self._device) * full_health
        self.robot_alive = torch.ones((self.num_teams, self.rpt, self._num_envs), dtype=torch.bool, device=self._device) 
        self.last_robot_health = torch.ones((self.num_teams, self.rpt, self._num_envs), dtype=torch.int32, device=self._device) * full_health
        self.last_robot_alive = torch.ones((self.num_teams, self.rpt, self._num_envs), dtype=torch.bool, device=self._device) 
        self.damage_dealt = torch.zeros((self.num_teams, self.rpt, self._num_envs), dtype=torch.int32, device=self._device)
        self.killed_ids = (torch.tensor([]), torch.tensor([]), torch.tensor([]))

        # game
        self.win = torch.zeros((self._num_envs,), device=self._device, dtype=torch.bool)
        self.lose = torch.zeros((self._num_envs,), device=self._device, dtype=torch.bool)
        self.draw = torch.zeros((self._num_envs,), device=self._device, dtype=torch.bool)
        self.extras = {
            'win': torch.zeros((self._num_envs,), device=self.rl_device, dtype=torch.bool),
            'lose': torch.zeros((self._num_envs,), device=self.rl_device, dtype=torch.bool),
            'draw': torch.zeros((self._num_envs,), device=self.rl_device, dtype=torch.bool),}
        
        # mutual
        self.mutual_position = torch.ones((self.rpt, self.rpt, self._num_envs, 2), dtype=torch.float32, device=self._device)
        self.mutual_orientation = torch.zeros((self.rpt, self.rpt, self._num_envs), dtype=torch.float32, device=self._device)
        self.mutual_distance = torch.ones((self.rpt, self.rpt, self._num_envs), dtype=torch.float32, device=self._device) * self.max_robot_distance
        self.mutual_coord = torch.ones((self.rpt, self.rpt, self._num_envs, 2), dtype=torch.int32, device=self._device) * self.max_coord
        self.mutual_manhattan_coord_distance = torch.ones((self.rpt, self.rpt, self._num_envs), dtype=torch.int32, device=self._device) * self.max_coord
        print("post reset done.")


    def calculate_metrics(self) -> None:
        last_health_adv = torch.sum(self.last_robot_health[0, ...], dim=0) - torch.sum(self.last_robot_health[1, ...], dim=0)
        last_alive_adv = torch.sum(self.last_robot_alive[0, ...].int(), dim=0) - torch.sum(self.last_robot_alive[1, ...].int(), dim=0)
        self.last_robot_health = self.robot_health.clone()
        self.last_robot_alive = self.robot_alive.clone()
        current_health_adv = torch.sum(self.robot_health[0, ...], dim=0) - torch.sum(self.robot_health[1, ...], dim=0)
        current_alive_adv = torch.sum(self.robot_alive[0, ...].int(), dim=0) - torch.sum(self.robot_alive[1, ...].int(), dim=0)
        hit_reward = (current_health_adv - last_health_adv) * self.hit_reward_factor
        kill_reward = (current_alive_adv - last_alive_adv) * self.kill_reward_factor

        win_battle = torch.sum(self.robot_alive[1, ...].int(), dim=0) == 0
        lose_battle = torch.sum(self.robot_alive[0, ...].int(), dim=0) == 0
        draw_battle = win_battle & lose_battle
        win_battle = win_battle ^ draw_battle
        lose_battle = lose_battle ^ draw_battle
        win_timeover = torch.sum(self.robot_health[0, ...], dim=0) > torch.sum(self.robot_health[1, ...], dim=0)
        lose_timeover = torch.sum(self.robot_health[0, ...], dim=0) < torch.sum(self.robot_health[1, ...], dim=0)
        draw_timeover = torch.sum(self.robot_health[0, ...], dim=0) == torch.sum(self.robot_health[1, ...], dim=0)
        timeover = self.progress_buf >= self._max_episode_length
        self.win[:] = win_battle | (win_timeover & timeover)
        self.lose[:] = lose_battle | (lose_timeover & timeover)
        self.draw[:] = draw_battle | (draw_timeover & timeover)

        game_reward = (self.win.float() - self.lose.float()) * self.game_reward_factor
        self.extras['win'][:] = self.win
        self.extras['lose'][:] = self.lose
        self.extras['draw'][:] = self.draw      
        reward = hit_reward + kill_reward + game_reward
        self.rew_buf[:] = reward
        

    def is_done(self) -> None:
        outside_x = (self.robot_translation[..., 0] > self.field_half_length).any(0).any(0)
        outside_y = (self.robot_translation[..., 1] > self.field_half_width).any(0).any(0)
        outside = outside_x | outside_y

        min_roll, _ = torch.min(torch.abs(self.robot_roll.flatten(0, 1) - 3.14), dim=0)
        overturn = torch.where(min_roll < 0.14, True, False)
        resets = self.win | self.lose | self.draw | outside | overturn

        resets = torch.where(self.progress_buf >= self._max_episode_length, torch.ones_like(self.reset_buf), resets)
        self.reset_buf[:] = resets


    def render_shoot(self, shoot_ready):
        self.rayOrigin[..., 0:2] = self.robot_pos[..., 0:2].cpu()
        self.rayDir[..., 0] = (torch.cos(self.shoot_orientation) * self.shoot_distance).cpu()
        self.rayDir[..., 1] = (torch.sin(self.shoot_orientation) * self.shoot_distance).cpu()
        for e in range(self._num_envs):
            for t in range(self.num_teams):
                for r in range(self.rpt):
                    if shoot_ready[t, r, e]:
                        hitInfo = get_physx_scene_query_interface().raycast_closest(self.rayOrigin[t,r,e], self.rayDir[t,r,e], self.rayMaxDist)
                        if hitInfo["hit"]:
                            rigidBody = hitInfo["rigidBody"]
                            rigidBodyList = rigidBody.split('/')
                            hited_ground = False
                            if rigidBodyList[2] != 'envs':
                                hited_ground = True
                            else:
                                hit_obj = rigidBodyList[4] 
                                hit_entity = rigidBodyList[5] 
                            with Sdf.ChangeBlock():
                                hitPos = hitInfo["position"]
                                if t == 0:
                                    self._debugDraw.draw_line(self.rayOrigin[t,r,e], COLOR_RED, hitPos, COLOR_RED)
                                else:
                                    self._debugDraw.draw_line(self.rayOrigin[t,r,e], COLOR_BLUE, hitPos, COLOR_BLUE)
                                self._debugDraw.draw_sphere(hitPos, 0.03, COLOR_CYAN)
                                if not hited_ground:
                                    if hit_entity[0] != 'o':
                                        if hit_entity == "chassis":
                                            usdGeom = UsdGeom.Mesh.Get(self._stage, rigidBody + "/visuals")
                                            usdGeom.GetDisplayColorAttr().Set(self.hitRobotColor)
                                            self.hit_box.append(usdGeom)
    

    def color_recovery(self):
        with Sdf.ChangeBlock():
            for usdGeom in self.hit_box:
                usdGeom.GetDisplayColorAttr().Set(self.origColor)
            for killed_unit_idx in range(self.killed_ids[0].shape[0]):
                team_idx = self.killed_ids[0][killed_unit_idx].item()
                robot_side = self.team_names[team_idx]
                robot_idx = self.killed_ids[1][killed_unit_idx].item()
                env_idx = self.killed_ids[2][killed_unit_idx].item()
                prim_path = f"/World/envs/env_{env_idx}/{robot_side}{robot_idx}/chassis/visuals"
                usdGeom = UsdGeom.Mesh.Get(self._stage, prim_path)
                usdGeom.GetDisplayColorAttr().Set(self.killedColor)
        self.hit_box.clear()


    def reset_color(self, env_ids):
        with Sdf.ChangeBlock():
            for env_idx in env_ids:
                for t in range(self.num_teams):
                    for r in range(self.rpt):
                        prim_path = f"/World/envs/env_{env_idx.item()}/{self.team_names[t]}{r}/chassis/visuals"
                        usdGeom = UsdGeom.Mesh.Get(self._stage, prim_path)
                        usdGeom.GetDisplayColorAttr().Set(self.origColor)




    def init_debugdraw(self):
        self._debugDraw = get_debug_draw_interface()
        self.hitRobotColor = Vt.Vec3fArray([Gf.Vec3f(255.0 / 255.0, 69.0 / 255.0, 0.0 / 255.0)])
        self.killedColor = Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 0.0)])
        self.origColor = Vt.Vec3fArray([Gf.Vec3f(128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0)])
        self.hit_box = []

        self.rayMaxDist = 10
        self.rayOrigin = np.ones((self.num_teams, self.rpt, self._num_envs, 3)) * 0.36
        self.rayDir = np.ones((self.num_teams, self.rpt, self._num_envs, 3)) * -0.25









