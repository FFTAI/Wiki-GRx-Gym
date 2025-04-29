import torch

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from legged_gym.envs.base.legged_robot_code import LeggedRobot
from legged_gym.envs.fftai.legged_robot_fftai_config import LeggedRobotFFTAICfg


class LeggedRobotFFTAI(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg: LeggedRobotFFTAICfg = cfg

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

    def _init_cfg(self, cfg: LeggedRobotFFTAICfg):
        super()._init_cfg(cfg)

    # ----------------------------------------------

    def _init_buffers_others(self):
        super()._init_buffers_others()

        # env_ids for different commands
        self.env_ids_of_off_command = torch.arange(self.num_envs, dtype=torch.int, device=self.device,
                                                   requires_grad=False)
        self.env_ids_of_stand_command = torch.arange(self.num_envs, dtype=torch.int, device=self.device,
                                                     requires_grad=False)

        # robot info
        self.default_dof_pos_tenors = torch.ones(self.num_envs,
                                                 self.num_dofs,
                                                 dtype=torch.float, device=self.device, requires_grad=False) \
                                      * self.default_dof_pos

        # average values
        self.avg_base_lin_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                            requires_grad=False)
        self.avg_base_ang_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                            requires_grad=False)

        self.avg_feet_contact_force = torch.zeros(self.num_envs,
                                                  len(self.feet_indices),
                                                  dtype=torch.float, device=self.device, requires_grad=False)
        self.avg_feet_speed_xyz = torch.zeros(self.num_envs,
                                              len(self.feet_indices), 3,
                                              device=self.device, requires_grad=False)
        self.avg_feet_speed_rpy = torch.zeros(self.num_envs,
                                              len(self.feet_indices), 3,
                                              device=self.device, requires_grad=False)

        # contact
        self.feet_contact = torch.zeros(self.num_envs,
                                        len(self.feet_indices),
                                        dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_contact_last = torch.zeros(self.num_envs,
                                             len(self.feet_indices),
                                             dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_contact_trig = torch.zeros(self.num_envs,
                                             len(self.feet_indices),
                                             dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_contact_time = torch.zeros(self.num_envs,
                                             len(self.feet_indices),
                                             dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_contact_time_last = torch.zeros(self.num_envs,
                                                  len(self.feet_indices),
                                                  dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs,
                                         len(self.feet_indices),
                                         dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time_last = torch.zeros(self.num_envs,
                                              len(self.feet_indices),
                                              dtype=torch.float, device=self.device, requires_grad=False)

        # feet pos, height
        self.feet_pos = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device, requires_grad=False)
        self.feet_quat = torch.zeros(self.num_envs, len(self.feet_indices), 4, device=self.device, requires_grad=False)

        self.feet_height = torch.zeros(self.num_envs,
                                       len(self.feet_indices),
                                       dtype=torch.float, device=self.device, requires_grad=False)

    # ----------------------------------------------

    def before_physics_step(self):
        self.avg_feet_contact_force = torch.zeros(self.num_envs,
                                                  len(self.feet_indices),
                                                  dtype=torch.float, device=self.device, requires_grad=False)
        self.avg_feet_speed_xyz = torch.zeros(self.num_envs,
                                              len(self.feet_indices), 3,
                                              device=self.device, requires_grad=False)
        self.avg_feet_speed_rpy = torch.zeros(self.num_envs,
                                              len(self.feet_indices), 3,
                                              device=self.device, requires_grad=False)

    def during_physics_step(self):
        super().during_physics_step()

    def _during_physics_step_in_sim(self):
        super()._during_physics_step_in_sim()

        # compute some quantities
        self.avg_feet_contact_force += \
            torch.norm(self.contact_forces[:, self.feet_indices, 0:3], dim=-1)
        self.avg_feet_speed_xyz += \
            torch.abs(self.rigid_body_states[:, self.feet_indices][:, 0:len(self.feet_indices), 7:10])
        self.avg_feet_speed_rpy += \
            torch.abs(self.rigid_body_states[:, self.feet_indices][:, 0:len(self.feet_indices), 10:13])

    def _during_physics_step_after_sim(self):
        super()._during_physics_step_after_sim()

        self.avg_feet_contact_force /= self.cfg.control.decimation
        self.avg_feet_speed_xyz /= self.cfg.control.decimation
        self.avg_feet_speed_rpy /= self.cfg.control.decimation

    def post_physics_step_update_state(self):
        super().post_physics_step_update_state()

        self._calculate_feet_contact()
        self._calculate_feet_height()

    def _calculate_feet_contact(self):
        self.feet_contact_last = self.feet_contact.clone()
        self.feet_contact = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 0.1

        # feet_contact_trig get the state when feet_contact_last is 0, while feet_contact is 1
        self.feet_contact_trig = self.feet_contact & ~self.feet_contact_last

        # record feet_contact_time_last
        self.feet_contact_time_last = self.feet_contact_time.clone()

        # feet_contact_time get the time when feet_contact is 1
        # feet_contact_time = 0 when feet_contact is 0
        self.feet_contact_time += self.feet_contact * self.dt
        self.feet_contact_time *= self.feet_contact

        # record feet_air_time_last
        self.feet_air_time_last = self.feet_air_time.clone()

        # feet_air_time get the time when feet_contact is 0
        # feet_air_time = 0 when feet_contact is 1
        self.feet_air_time += ~self.feet_contact * self.dt
        self.feet_air_time *= ~self.feet_contact

    def _calculate_feet_height(self):
        self.feet_pos = self.rigid_body_states[:, self.feet_indices][:, 0:len(self.feet_indices), 0:3]  # in world frame
        self.feet_quat = self.rigid_body_states[:, self.feet_indices][:, 0:len(self.feet_indices),
                         3:7]  # in world frame

        self.feet_elevation = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device,
                                          requires_grad=False)

        for i in range(len(self.feet_indices)):
            feet_forward = quat_apply(self.feet_quat[:, i], self.forward_vec)
            feet_elevation = torch.asin(feet_forward[:, 2:3])  # in world frame
            feet_elevation *= ~(torch.abs(feet_elevation) < 0.001)  # too small set to 0

            self.feet_elevation[:, i] = feet_elevation.squeeze()

        self.feet_height = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device, requires_grad=False)

        for i in range(len(self.feet_indices)):
            foot_height = torch.mean(
                self.rigid_body_states[:, self.feet_indices][:, i, 2:3]
                - self.cfg.asset.foot_thickness
                - self.measured_heights,
                dim=1)

            self.feet_height[:, i] = foot_height

    # ----------------------------------------------

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

        self.avg_feet_contact_force[env_ids] = 0.0
        self.avg_feet_speed_xyz[env_ids] = 0.0
        self.avg_feet_speed_rpy[env_ids] = 0.0

        self.feet_contact[env_ids] = 0.0
        self.feet_contact_last[env_ids] = 0.0
        self.feet_contact_trig[env_ids] = 0.0
        self.feet_contact_time[env_ids] = 0.0
        self.feet_contact_time_last[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.feet_air_time_last[env_ids] = 0.0

    def reset_last_values(self, env_ids):
        self.last_last_actions[env_ids] = 0.0

        super().reset_last_values(env_ids)

    def record_last_values(self):
        self.last_last_actions[:] = self.last_actions[:]

        super().record_last_values()

    # ----------------------------------------------

    def clip_actions(self, actions):
        actions_clipped = torch.clip(actions, self.clip_actions_min, self.clip_actions_max).to(self.device)
        return actions_clipped

    # ----------------------------------------------

    def get_mirror_observations(self, observations):
        """
        Returns the mirror observations

        Jason 2024-11-02:
        需要注意的是，mirror 的设计针对的是节律性对称性的动作进行设计的，
        如果某项动作不具备节律性和对称性的特点，则不能加入这部分的计算 loss 中。
        """
        if self.cfg.mirror.enable_mirror:
            mirror_observations = torch.zeros_like(observations)

            # 如果使用 stack，则需要对每个 stack 进行 mirror
            if self.cfg.env.use_stack:

                # 1. change direction
                for i in range(self.cfg.env.num_stack):
                    mirror_observations[:,
                    self.cfg.env.num_obs * (i + 0):
                    self.cfg.env.num_obs * (i + 1)] = \
                        observations[:,
                        self.cfg.env.num_obs * (i + 0):
                        self.cfg.env.num_obs * (i + 1)] \
                        * torch.from_numpy(self.cfg.mirror.observations_coefficient).float().to(self.device)

                mirror_observations_change_direction = mirror_observations.clone()  # backup

                # 2. exchange sequence
                for i in range(self.cfg.env.num_stack):
                    for idx1, idx2 in self.cfg.mirror.observations_exchange:
                        mirror_observations[:, idx1 + self.cfg.env.num_obs * (i + 0)] = \
                            mirror_observations_change_direction[:, idx2 + self.cfg.env.num_obs * (i + 0)]
                        mirror_observations[:, idx2 + self.cfg.env.num_obs * (i + 0)] = \
                            mirror_observations_change_direction[:, idx1 + self.cfg.env.num_obs * (i + 0)]

            else:
                # 1. change direction
                mirror_observations = \
                    observations \
                    * torch.from_numpy(self.cfg.mirror.observations_coefficient).float().to(self.device)

                mirror_observations_change_direction = mirror_observations.clone()  # backup

                # 2. exchange sequence
                for idx1, idx2 in self.cfg.mirror.observations_exchange:
                    mirror_observations[:, idx1] = mirror_observations_change_direction[:, idx2]
                    mirror_observations[:, idx2] = mirror_observations_change_direction[:, idx1]

        else:
            mirror_observations = observations

        return mirror_observations

    def get_mirror_actions(self, actions):
        """
        Returns the mirror actions

        Jason 2024-11-02:
        需要注意的是，mirror 的设计针对的是节律性对称性的动作进行设计的，
        如果某项动作不具备节律性和对称性的特点，则不能加入这部分的计算 loss 中。
        """
        if self.cfg.mirror.enable_mirror:
            mirror_actions = torch.zeros_like(actions)

            # 1. change direction
            mirror_actions = \
                actions \
                * torch.from_numpy(self.cfg.mirror.actions_coefficient).float().to(self.device)

            mirror_actions_change_direction = mirror_actions.clone()  # backup

            # 2. exchange sequence
            for idx1, idx2 in self.cfg.mirror.actions_exchange:
                mirror_actions[:, idx1] = mirror_actions_change_direction[:, idx2]
                mirror_actions[:, idx2] = mirror_actions_change_direction[:, idx1]
        else:
            mirror_actions = actions

        return mirror_actions

    # ==========================================================================================================================
    # Reward functions

    def _reward_termination(self):
        # Terminal reward / penalty
        reward_termination = self.reset_buf * ~self.time_out_buf
        return reward_termination

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        collision_error = 1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1)
        collision_error = torch.sum(collision_error, dim=1)  # dims 2->1

        # use exponential to make the reward more sparse
        reward_collision = 1 - torch.exp(self.cfg.rewards.sigma_collision
                                         * collision_error)
        return reward_collision

    # ----------------------------------------------

    def _reward_stand_still(self):
        # Penalize not standing still
        error_stand_still = torch.abs(self.dof_pos - self.default_dof_pos_tenors) \
                            * self.dof_pos_offset_scales
        error_stand_still = torch.sum(error_stand_still, dim=1)  # dims 2->1

        reward_stand_still = torch.exp(self.cfg.rewards.sigma_stand_still
                                       * error_stand_still)

        """
        Jason 2024-03-23:
        Only apply the reward to the environment that is in the stand state
        """
        selector_stand_still = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        selector_stand_still[self.env_ids_of_stand_command] = 1

        reward_stand_still *= selector_stand_still

        return reward_stand_still

    def _reward_stand_still_dof_pos(self):
        """
        Penalize not standing still
        """
        error_stand_still_pos = torch.abs(self.dof_pos - self.default_dof_pos_tenors) \
                                * self.dof_pos_offset_scales
        error_stand_still_pos = torch.sum(error_stand_still_pos, dim=1)  # dims 2->1

        reward_stand_still_pos = torch.exp(self.cfg.rewards.sigma_stand_still_dof_pos
                                           * error_stand_still_pos)

        """
        Jason 2024-03-23:
        Only apply the reward to the environment that is in the stand state
        """
        selector_stand = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)  # dims 1
        selector_stand[self.env_ids_of_stand_command] = 1

        reward_stand_still_pos *= selector_stand

        return reward_stand_still_pos

    def _reward_stand_still_dof_vel(self):
        """
        Penalize not standing still
        """
        error_stand_still_vel = torch.abs(self.dof_vel) \
                                * self.dof_vel_scales
        error_stand_still_vel = torch.sum(error_stand_still_vel, dim=1)  # dims 2->1

        reward_stand_still_vel = torch.exp(self.cfg.rewards.sigma_stand_still_dof_vel
                                           * error_stand_still_vel)

        """
        Jason 2024-03-23:
        Only apply the reward to the environment that is in the stand state
        """
        selector_stand = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)  # dims 1
        selector_stand[self.env_ids_of_stand_command] = 1

        reward_stand_still_vel *= selector_stand

        return reward_stand_still_vel

    # ----------------------------------------------

    def _reward_cmd_diff_base_lin_vel_x(self):
        error_x_vel = torch.abs(self.commands_base_lin_vel_x - self.base_lin_vel[:, 0:1])
        error_x_vel = torch.sum(error_x_vel, dim=1)  # dims 2->1

        reward_x_vel = torch.exp(self.cfg.rewards.sigma_cmd_diff_base_lin_vel_x
                                 * error_x_vel)
        return reward_x_vel

    def _reward_cmd_diff_base_lin_vel_y(self):
        error_y_vel = torch.abs(self.commands_base_lin_vel_y - self.base_lin_vel[:, 1:2])
        error_y_vel = torch.sum(error_y_vel, dim=1)  # dims 2->1

        reward_y_vel = torch.exp(self.cfg.rewards.sigma_cmd_diff_base_lin_vel_y
                                 * error_y_vel)
        return reward_y_vel

    def _reward_cmd_diff_base_ang_vel_yaw(self):
        error_yaw_vel = torch.abs(self.commands_base_ang_vel_yaw - self.base_ang_vel[:, 2:3])
        error_yaw_vel = torch.sum(error_yaw_vel, dim=1)  # dims 2->1

        reward_yaw_vel = torch.exp(self.cfg.rewards.sigma_cmd_diff_base_ang_vel_yaw
                                   * error_yaw_vel)
        return reward_yaw_vel

    # ----------------------------------------------

    def _reward_base_avg_lin_vel_z(self):
        error_z_vel = torch.abs(0 - self.avg_base_lin_vel[:, 2:3])
        error_z_vel = torch.sum(error_z_vel, dim=1)

        reward_z_vel = torch.exp(self.cfg.rewards.sigma_base_avg_lin_vel_z
                                 * error_z_vel)
        return reward_z_vel

    def _reward_base_ang_vel_roll(self):
        error_roll_vel = torch.abs(0 - self.base_ang_vel[:, 0:1])
        error_roll_vel = torch.sum(error_roll_vel, dim=1)  # dims 2->1

        reward_roll_vel = torch.exp(self.cfg.rewards.sigma_base_ang_vel_roll
                                    * error_roll_vel)
        return reward_roll_vel

    def _reward_base_ang_vel_pitch(self):
        error_pitch_vel = torch.abs(0 - self.base_ang_vel[:, 1:2])
        error_pitch_vel = torch.sum(error_pitch_vel, dim=1)  # dims 2->1

        reward_pitch_vel = torch.exp(self.cfg.rewards.sigma_base_ang_vel_pitch
                                     * error_pitch_vel)
        return reward_pitch_vel

    def _reward_base_lin_vel_z(self):
        error_z_vel = torch.abs(0 - self.base_lin_vel[:, 2:3])
        error_z_vel = torch.sum(error_z_vel, dim=1)  # dims 2->1

        reward_z_vel = torch.exp(self.cfg.rewards.sigma_base_lin_vel_z
                                 * error_z_vel)
        return reward_z_vel

    def _reward_base_lin_vel_xy(self):
        error_xy_vel = torch.norm(0 - self.base_lin_vel[:, 0:2], dim=1)
        reward_xy_vel = torch.exp(self.cfg.rewards.sigma_base_lin_vel_xy
                                  * error_xy_vel)
        return reward_xy_vel

    # ----------------------------------------------

    def _reward_base_height_offset(self):
        """
        Reward for base height offset
        """
        error_base_height = torch.abs(self.base_heights_offset)
        error_base_height = torch.sum(error_base_height, dim=1)  # dims 2->1

        reward_base_height = torch.exp(self.cfg.rewards.sigma_base_height_offset
                                       * error_base_height)
        return reward_base_height

    def _reward_base_height_offset_range(self):
        """
        Reward for base height offset range
        """
        error_base_height = torch.abs(self.base_heights_offset)

        # 只计算高度超出范围的情况
        error_base_height = \
            (error_base_height - self.cfg.rewards.base_height_offset_range_limit) \
            * (error_base_height > self.cfg.rewards.base_height_offset_range_limit)
        error_base_height = torch.sum(error_base_height, dim=1)  # dims 2->1

        reward_base_height = torch.exp(self.cfg.rewards.sigma_base_height_offset
                                       * error_base_height)
        return reward_base_height

    # ----------------------------------------------

    def _reward_base_flat_orient(self):
        base_projected_gravity = self.base_projected_gravity

        error_base_flat_orient = torch.abs(base_projected_gravity[:, 0:2])
        error_base_flat_orient = torch.sum(error_base_flat_orient, dim=1)  # dims 2->1

        reward_base_flat_orient = torch.exp(self.cfg.rewards.sigma_base_flat_orient
                                            * error_base_flat_orient)
        return reward_base_flat_orient

    def _reward_torso_flat_orient(self):
        if len(self.torso_indices) > 0:
            torso_projected_gravity = quat_rotate_inverse(
                self.rigid_body_states[:, self.torso_indices][:, 0, 3:7],
                self.gravity_vec)

            error_torso_flat_orient = torch.abs(torso_projected_gravity[:, 0:2])
            error_torso_flat_orient = torch.sum(error_torso_flat_orient, dim=1)  # dims 2->1

            reward_torso_flat_orient = torch.exp(self.cfg.rewards.sigma_torso_flat_orient
                                                 * error_torso_flat_orient)

        else:
            reward_torso_flat_orient = torch.zeros(self.num_envs, device=self.device)  # dims 1

        return reward_torso_flat_orient

    # ----------------------------------------------

    def _reward_action_diff(self):
        """
        Reward for action difference
        Returns:
            reward_action_diff: reward for action difference
        """
        error_action_diff = (self.actions - self.last_actions) \
                            * self.action_scales

        error_action_diff = torch.abs(error_action_diff)
        error_action_diff = torch.sum(error_action_diff, dim=1)  # dims 2->1

        reward_action_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff
                                           * error_action_diff)
        return reward_action_diff

    def _reward_action_diff_diff(self):
        """
        Reward for action difference difference
        Returns:
            reward_action_diff_diff: reward for action difference difference
        """
        error_action_diff = (self.actions - self.last_actions) \
                            * self.action_scales
        error_action_diff_last = (self.last_actions - self.last_last_actions) \
                                 * self.action_scales

        error_action_diff_diff = torch.abs(error_action_diff - error_action_diff_last)
        error_action_diff_diff = torch.sum(error_action_diff_diff, dim=1)  # dims 2->1

        reward_action_diff_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff_diff
                                                * error_action_diff_diff)
        return reward_action_diff_diff

    # ----------------------------------------------

    def _reward_dof_pos_offset(self):
        """
        Reward for dof position offset
        Returns:
            reward_dof_pos_offset: reward for dof position offset
        """
        error_dof_pos_offset = torch.abs(self.dof_pos - self.default_dof_pos) \
                               * self.dof_pos_offset_scales
        error_dof_pos_offset = torch.sum(error_dof_pos_offset, dim=1)  # dims 2->1

        reward_dof_pos_offset = torch.exp(self.cfg.rewards.sigma_dof_pos_offset
                                          * error_dof_pos_offset)
        return reward_dof_pos_offset

    # ----------------------------------------------

    def _reward_dof_vel(self):
        """
        Reward for dof velocity
        Returns:
            reward_dof_vel: reward for dof velocity
        """
        error_dof_vel = torch.abs(self.dof_vel) \
                        * self.dof_vel_scales
        error_dof_vel = torch.sum(error_dof_vel, dim=1)  # dims 2->1

        reward_dof_vel = 1 - torch.exp(self.cfg.rewards.sigma_dof_vel
                                       * error_dof_vel)
        return reward_dof_vel

    # ----------------------------------------------

    def _reward_dof_acc(self):
        """
        Reward for dof acceleration
        Returns:
            reward_dof_acc: reward for dof acceleration
        """
        error_dof_acc = torch.abs((self.dof_vel - self.last_dof_vel) / self.dt) \
                        * self.dof_vel_scales
        error_dof_acc = torch.sum(error_dof_acc, dim=1)  # dims 2->1

        reward_dof_acc = 1 - torch.exp(self.cfg.rewards.sigma_dof_acc
                                       * error_dof_acc)
        return reward_dof_acc

    # ----------------------------------------------

    def _reward_dof_tor(self):
        """
        Reward for dof torque
        Returns:
            reward_dof_tor: reward for dof torque
        """
        error_dof_tor = torch.abs(self.dof_tor) \
                        * self.dof_tor_scales
        error_dof_tor = torch.sum(error_dof_tor, dim=1)  # dims 2->1

        reward_dof_tor = 1 - torch.exp(self.cfg.rewards.sigma_dof_tor
                                       * error_dof_tor)
        return reward_dof_tor

    # ----------------------------------------------

    def _reward_limits_action(self):
        """
        Reward for pass the action limits
        """
        out_of_limits = -(self.actions_untreated - self.clip_actions_min).clip(max=0.)  # lower limit
        out_of_limits += (self.actions_untreated - self.clip_actions_max).clip(min=0.)  # upper limit

        error_limits_action = torch.abs(out_of_limits)
        error_limits_action = torch.sum(error_limits_action, dim=1)  # dims 2->1

        reward_limits_action = 1 \
                               - torch.exp(self.cfg.rewards.sigma_limits_action
                                           * error_limits_action)

        return reward_limits_action

    def _reward_limits_dof_pos(self):
        """
        Reward for pass the dof position limits
        """

        # ----------------------------------------------
        # get all controllable joints
        related_indexes = list(range(self.num_actions))
        # ----------------------------------------------

        out_of_limits = -(self.dof_pos[:, related_indexes] -
                          self.soft_dof_pos_limits[:, 0][related_indexes]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos[:, related_indexes]
                          - self.soft_dof_pos_limits[:, 1][related_indexes]).clip(min=0.)  # upper limit

        error_limits_dof_pos = torch.abs(out_of_limits)
        error_limits_dof_pos = torch.sum(error_limits_dof_pos, dim=1)  # dims 2->1

        reward_limits_dof_pos = 1 - torch.exp(self.cfg.rewards.sigma_limits_dof_pos
                                              * error_limits_dof_pos)

        return reward_limits_dof_pos

    def _reward_limits_dof_vel(self):
        """
        Reward for pass the dof velocity limits
        """

        # ----------------------------------------------
        # get all controllable joints
        related_indexes = list(range(self.num_actions))
        # ----------------------------------------------

        error_limits_dof_vel = (torch.abs(self.dof_vel[:, related_indexes])
                                - self.soft_dof_vel_limits[related_indexes]).clip(min=0.)
        error_limits_dof_vel = torch.sum(error_limits_dof_vel, dim=1)  # dims 2->1

        reward_limits_dof_vel = 1 - torch.exp(self.cfg.rewards.sigma_limits_dof_vel
                                              * error_limits_dof_vel)

        return reward_limits_dof_vel

    def _reward_limits_dof_tor(self):
        """
        Reward for pass the dof torque limits
        """

        # ----------------------------------------------
        # get all controllable joints
        related_indexes = list(range(self.num_actions))
        # ----------------------------------------------

        error_limits_dof_tor = (torch.abs(self.dof_tor[:, related_indexes])
                                - self.soft_dof_tor_limits[related_indexes]).clip(min=0.)
        error_limits_dof_tor = torch.sum(error_limits_dof_tor, dim=1)  # dims 2->1

        reward_limits_dof_tor = 1 - \
                                torch.exp(self.cfg.rewards.sigma_limits_dof_tor
                                          * error_limits_dof_tor)

        return reward_limits_dof_tor

    # ----------------------------------------------

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        left_foot_fxy = torch.norm(self.contact_forces[:, self.feet_indices][:, 0, 0:2], dim=1).unsqueeze(1)
        right_foot_fxy = torch.norm(self.contact_forces[:, self.feet_indices][:, 1, 0:2], dim=1).unsqueeze(1)

        left_foot_fz = self.contact_forces[:, self.feet_indices][:, 0, 2:3]
        right_foot_fz = self.contact_forces[:, self.feet_indices][:, 1, 2:3]

        error_left_foot_f = left_foot_fxy - self.cfg.rewards.feet_stumble_ratio * torch.abs(left_foot_fz)
        error_right_foot_f = right_foot_fxy - self.cfg.rewards.feet_stumble_ratio * torch.abs(right_foot_fz)

        error_left_foot_f = error_left_foot_f * (error_left_foot_f > 0)
        error_right_foot_f = error_right_foot_f * (error_right_foot_f > 0)

        error_left_foot_f = torch.sum(error_left_foot_f, dim=1)  # dims 2->1
        error_right_foot_f = torch.sum(error_right_foot_f, dim=1)  # dims 2->1

        reward_left_foot_f = 1 - torch.exp(self.cfg.rewards.sigma_feet_stumble
                                           * error_left_foot_f)
        reward_right_foot_f = 1 - torch.exp(self.cfg.rewards.sigma_feet_stumble
                                            * error_right_foot_f)

        reward_feet_stumble = reward_left_foot_f + reward_right_foot_f

        return reward_feet_stumble

    # ----------------------------------------------
