import numpy
import torch

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from legged_gym.envs.base.legged_robot import LeggedRobot

from .legged_robot_fftai_config import LeggedRobotFFTAICfg


class LeggedRobotFFTAI(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_cfg(self, cfg: LeggedRobotFFTAICfg):
        super()._init_cfg(cfg)

    # ----------------------------------------------

    def _init_buffers(self):
        super()._init_buffers()

        # robot info
        self.default_dof_pos_tenors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos_tenors = self.default_dof_pos_tenors * self.default_dof_pos

        # actions
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        # average values
        self.avg_feet_contact_force = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)
        self.avg_feet_speed = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)

        # contact
        self.feet_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_contact_last = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)

        # feet height
        self.feet_height = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)

    # ----------------------------------------------

    def before_physics_step(self):
        self.avg_feet_contact_force = 0.0

        self.avg_feet_speed = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device, requires_grad=False)
        self.avg_feet_speed_xyz = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device, requires_grad=False)

    def during_physics_step(self):

        for i in range(self.cfg.control.decimation):

            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.torques *= self.motor_strength
            self.torques = torch.clip(self.torques, -self.torque_limits, self.torque_limits)

            # simulate
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)

            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)

            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # compute some quantities
            self.dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt

            self.avg_feet_contact_force += torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=-1)
            self.avg_feet_speed += torch.norm(self.rigid_body_states[:, self.feet_indices][:, 0:len(self.feet_indices), 7:10], dim=-1)
            self.avg_feet_speed_xyz += torch.abs(self.rigid_body_states[:, self.feet_indices][:, 0:len(self.feet_indices), 7:10])

        self.avg_feet_contact_force /= self.cfg.control.decimation
        self.avg_feet_speed /= self.cfg.control.decimation
        self.avg_feet_speed_xyz /= self.cfg.control.decimation

    def post_physics_step(self):
        reset_env_ids = super().post_physics_step()

        # record last values
        self.last_last_actions[:] = self.last_actions[:]

        # update feet contact air time state
        self.feet_air_time = self.feet_air_time * (~self.feet_contact_filt)

        return reset_env_ids

    def post_physics_step_update_state(self):
        super().post_physics_step_update_state()

        self._calculate_air_time()
        self._calculate_feet_height()
        self._calculate_land_time()

    def _calculate_air_time(self):
        # detect feet contact
        self.feet_contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.feet_contact_filt = torch.logical_or(self.feet_contact, self.feet_contact_last)
        self.feet_contact_last = self.feet_contact
        self.feet_first_contact = (self.feet_air_time > 0) * self.feet_contact_filt

        # 只要有脚接触到地面，就将其对应的 feet_air_time 置为 0
        self.feet_air_time += self.dt

    def _calculate_feet_height(self):
        self.feet_height = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device, requires_grad=False)
        for i in range(len(self.feet_indices)):
            foot_height = torch.mean(
                self.rigid_body_states[:, self.feet_indices][:, i, 2].unsqueeze(1) - self.measured_heights, dim=1)

            self.feet_height[:, i] = foot_height

    def _calculate_land_time(self):
        # detect feet contact
        self.feet_contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.feet_contact_last = self.feet_contact

        # 只要有脚接触到地面，就将其对应的 feet_land_time 置为 0
        self.feet_land_time += self.dt
        self.feet_land_time = self.feet_land_time * self.feet_contact

    # ----------------------------------------------

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

        self.avg_feet_contact_force[env_ids] = 0.0
        self.avg_feet_speed[env_ids] = 0.0

        self.last_last_actions[env_ids] = 0.0

    def compute_observation_variables(self):
        self.base_heights_offset = \
            torch.mean(
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1)
                    - self.cfg.rewards.base_height_target
                    - self.measured_heights,
                    min=-1.0,
                    max=1.0)
                * self.obs_scales.height_measurements, dim=1)

        self.surround_heights_offset = \
            torch.clip(self.root_states[:, 2].unsqueeze(1)
                       - self.cfg.rewards.base_height_target
                       - self.measured_heights,
                       min=-1.0,
                       max=1.0) \
            * self.obs_scales.height_measurements

        self.dof_pos_offset = self.dof_pos - self.default_dof_pos

    # ----------------------------------------------

    def clip_actions(self, actions):
        clip_actions_max = torch.tensor(self.cfg.normalization.clip_actions_max).to(torch.float32).to(self.device)
        clip_actions_min = torch.tensor(self.cfg.normalization.clip_actions_min).to(torch.float32).to(self.device)

        actions_clipped = torch.clip(actions, clip_actions_min, clip_actions_max).to(self.device)

        return actions_clipped

    # ==============================================

    def _reward_termination(self):
        # Terminal reward / penalty
        reward_termination = self.reset_buf * ~self.time_out_buf
        return reward_termination

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        collision_error = torch.sum(
            1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

        # use exponential to make the reward more sparse
        reward_collision = 1 - torch.exp(self.cfg.rewards.sigma_collision
                                         * collision_error)
        return reward_collision

    def _reward_stand_still(self):
        # Penalize not standing still
        env_ids_for_stand_command = torch.where(torch.norm(self.commands[:, :2], dim=1) < 0.1)

        selector_stand_still = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        selector_stand_still[env_ids_for_stand_command] = 1

        error_stand_still = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos_tenors), dim=1)
        reward_stand_still = 1 - torch.exp(self.cfg.rewards.sigma_stand_still
                                           * error_stand_still)
        reward_stand_still *= selector_stand_still
        return reward_stand_still

    # ----------------------------------------------

    # 惩罚 x 速度差异
    def _reward_cmd_diff_lin_vel_x(self):
        error_x_vel = torch.abs(self.commands[:, 0] - self.base_lin_vel[:, 0])
        reward_x_vel = torch.exp(self.cfg.rewards.sigma_cmd_diff_lin_vel_x * error_x_vel)
        return reward_x_vel

    # 惩罚 y 速度差异
    def _reward_cmd_diff_lin_vel_y(self):
        error_y_vel = torch.abs(self.commands[:, 1] - self.base_lin_vel[:, 1])
        reward_y_vel = torch.exp(self.cfg.rewards.sigma_cmd_diff_lin_vel_y * error_y_vel)
        return reward_y_vel

    # 惩罚 z 速度差异
    def _reward_cmd_diff_lin_vel_z(self):
        error_z_vel = torch.abs(0 - self.base_lin_vel[:, 2])
        reward_z_vel = torch.exp(self.cfg.rewards.sigma_cmd_diff_lin_vel_z * error_z_vel)
        return reward_z_vel

    # 惩罚 roll 速度
    def _reward_cmd_diff_ang_vel_roll(self):
        error_roll_vel = torch.abs(0 - self.base_ang_vel[:, 0])
        reward_roll_vel = torch.exp(self.cfg.rewards.sigma_cmd_diff_ang_vel_roll * error_roll_vel)
        return reward_roll_vel

    # 惩罚 pitch 速度
    def _reward_cmd_diff_ang_vel_pitch(self):
        error_pitch_vel = torch.abs(0 - self.base_ang_vel[:, 1])
        reward_pitch_vel = torch.exp(self.cfg.rewards.sigma_cmd_diff_ang_vel_pitch * error_pitch_vel)
        return reward_pitch_vel

    # 惩罚 yaw 速度差异
    def _reward_cmd_diff_ang_vel_yaw(self):
        error_yaw_vel = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        reward_yaw_vel = torch.exp(self.cfg.rewards.sigma_cmd_diff_ang_vel_yaw * error_yaw_vel)
        return reward_yaw_vel

    # 惩罚 base height 差异
    def _reward_cmd_diff_base_height(self):
        error_base_height = torch.abs(self.base_heights_offset) \
                            * (self.base_heights_offset < 0)  # 只计算高度不足的情况
        reward_base_height = torch.exp(self.cfg.rewards.sigma_cmd_diff_base_height * error_base_height)
        return reward_base_height

    # ----------------------------------------------

    # 奖励 base orientation 无差异
    def _reward_cmd_diff_base_orient(self):
        error_base_orient = torch.sum(torch.abs(self.base_projected_gravity[:, :2]), dim=1)
        reward_base_orient = torch.exp(self.cfg.rewards.sigma_cmd_diff_base_orient
                                       * error_base_orient)
        return reward_base_orient

    # ----------------------------------------------

    # 惩罚 action 差异
    def _reward_action_diff(self):
        error_action_diff = (self.last_actions - self.actions) \
                            * self.cfg.control.action_scale
        error_action_diff = torch.sum(torch.abs(error_action_diff), dim=1)
        reward_action_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff
                                           * error_action_diff)
        return reward_action_diff

    def _reward_action_diff_diff(self):
        error_action_diff = (self.last_actions - self.actions) \
                            * self.cfg.control.action_scale
        error_action_diff_last = (self.last_last_actions - self.last_actions) * self.cfg.control.action_scale
        error_action_diff_diff = torch.sum(torch.abs(error_action_diff - error_action_diff_last), dim=1)
        reward_action_diff_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff_diff
                                                * error_action_diff_diff)
        return reward_action_diff_diff

    # ----------------------------------------------

    # 惩罚速度
    def _reward_dof_vel_new(self):
        error_new_dof_vel = torch.sum(torch.abs(self.dof_vel), dim=1)
        reward_new_dof_vel = 1 - torch.exp(self.cfg.rewards.sigma_dof_vel_new
                                           * error_new_dof_vel)
        return reward_new_dof_vel

    # ----------------------------------------------

    # 惩罚加速度
    def _reward_dof_acc_new(self):
        error_new_dof_acc = torch.sum(torch.abs((self.last_dof_vel
                                                 - self.dof_vel) / self.dt), dim=1)
        reward_new_dof_acc = 1 - torch.exp(self.cfg.rewards.sigma_dof_acc_new
                                           * error_new_dof_acc)
        return reward_new_dof_acc

    # ----------------------------------------------

    # 惩罚关节力矩
    def _reward_dof_tor_new(self):
        error_dof_tor_new = torch.sum(torch.abs(self.torques), dim=1)
        reward_dof_tor_new = 1 - torch.exp(self.cfg.rewards.sigma_dof_tor_new
                                           * error_dof_tor_new)
        return reward_dof_tor_new

    # ----------------------------------------------

    def _reward_pose_offset(self):
        error_pose_offset = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        reward_pose_offset = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset
                                           * error_pose_offset)
        return reward_pose_offset

    # ----------------------------------------------

    # 惩罚 指令接近位置上限
    def _reward_limits_actions(self):
        related_indexes = list(range(self.num_actions))

        out_of_limits = -(self.actions[:, related_indexes] * self.cfg.control.action_scale
                          - self.dof_pos_limits[:, 0][related_indexes]).clip(max=0.)  # lower limit
        out_of_limits += (self.actions[:, related_indexes] * self.cfg.control.action_scale
                          - self.dof_pos_limits[:, 1][related_indexes]).clip(min=0.)  # upper limit

        error_limits_actions = torch.sum(torch.square(out_of_limits), dim=1)
        reward_limits_actions = 1 - torch.exp(self.cfg.rewards.sigma_limits_actions
                                              * error_limits_actions)

        return reward_limits_actions

    # 惩罚 接近位置上限
    def _reward_limits_dof_pos(self):
        related_indexes = list(range(self.num_actions))

        out_of_limits = -(self.dof_pos[:, related_indexes] -
                          self.dof_pos_limits[:, 0][related_indexes]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos[:, related_indexes]
                          - self.dof_pos_limits[:, 1][related_indexes]).clip(min=0.)  # upper limit

        error_limits_dof_pos = torch.sum(torch.abs(out_of_limits), dim=1)
        reward_limits_dof_pos = 1 \
                                - torch.exp(self.cfg.rewards.sigma_limits_dof_pos
                                            * error_limits_dof_pos)
        return reward_limits_dof_pos

    # 惩罚 超过速度上限
    def _reward_limits_dof_vel(self):
        error_limits_dof_vel = \
            torch.sum((torch.abs(self.dof_vel)
                       - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
        reward_limits_dof_vel = 1 \
                                - torch.exp(self.cfg.rewards.sigma_limits_dof_vel
                                            * error_limits_dof_vel)
        return reward_limits_dof_vel

        # 惩罚 超过力矩上限

    def _reward_limits_dof_tor(self):
        error_limits_dof_tor = \
            torch.sum((torch.abs(self.torques)
                       - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
        reward_limits_dof_tor = 1 - \
                                torch.exp(self.cfg.rewards.sigma_limits_dof_tor
                                          * error_limits_dof_tor)
        return reward_limits_dof_tor

    # ----------------------------------------------
