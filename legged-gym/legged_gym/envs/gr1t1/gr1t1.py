import numpy
import json

import torch
from isaacgym.torch_utils import *
from isaacgym import gymapi, gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import LeggedRobotFFTAI
from legged_gym.utils.math import quat_apply_yaw
from legged_gym.utils.helpers import class_to_dict

from .gr1t1_config import GR1T1Cfg


class GR1T1(LeggedRobotFFTAI):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.swing_feet_height_target = torch.ones(self.num_envs, 1,
                                                   dtype=torch.float, device=self.device, requires_grad=False) \
                                        * self.cfg.rewards.swing_feet_height_target

        # additionally initialize actuator network hidden state tensors
        self.sea_input = torch.zeros(self.num_envs * self.actor_num_output, 1, 2, device=self.device, requires_grad=False)
        self.sea_hidden_state = torch.zeros(2, self.num_envs * self.actor_num_output, 3, device=self.device, requires_grad=False)
        self.sea_cell_state = torch.zeros(2, self.num_envs * self.actor_num_output, 3, device=self.device, requires_grad=False)
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(2, self.num_envs, self.actor_num_output, 3)
        self.sea_cell_state_per_env = self.sea_cell_state.view(2, self.num_envs, self.actor_num_output, 3)

    def _create_envs_get_indices(self, body_names, env_handle, actor_handle):
        """ Creates a list of indices for different bodies of the robot.
        """
        torso_name = [s for s in body_names if self.cfg.asset.torso_name in s]
        chest_name = [s for s in body_names if self.cfg.asset.chest_name in s]
        forehead_indices = [s for s in body_names if self.cfg.asset.forehead_name in s]

        imu_name = [s for s in body_names if self.cfg.asset.imu_name in s]

        waist_names = [s for s in body_names if self.cfg.asset.waist_name in s]
        head_names = [s for s in body_names if self.cfg.asset.head_name in s]
        thigh_names = [s for s in body_names if self.cfg.asset.thigh_name in s]
        shank_names = [s for s in body_names if self.cfg.asset.shank_name in s]
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        sole_names = [s for s in body_names if self.cfg.asset.sole_name in s]
        upper_arm_names = [s for s in body_names if self.cfg.asset.upper_arm_name in s]
        lower_arm_names = [s for s in body_names if self.cfg.asset.lower_arm_name in s]
        hand_names = [s for s in body_names if self.cfg.asset.hand_name in s]

        arm_base_names = [s for s in body_names if self.cfg.asset.arm_base_name in s]
        arm_end_names = [s for s in body_names if self.cfg.asset.arm_end_name in s]

        self.torso_indices = torch.zeros(len(torso_name), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(torso_name)):
            self.torso_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, torso_name[j])

        self.chest_indices = torch.zeros(len(chest_name), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(chest_name)):
            self.chest_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, chest_name[j])

        self.forehead_indices = torch.zeros(len(forehead_indices), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(forehead_indices)):
            self.forehead_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, forehead_indices[j])

        self.imu_indices = torch.zeros(len(imu_name), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(imu_name)):
            self.imu_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, imu_name[j])

        self.waist_indices = torch.zeros(len(waist_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(waist_names)):
            self.waist_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, waist_names[j])

        self.head_indices = torch.zeros(len(head_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(head_names)):
            self.head_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, head_names[j])

        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(thigh_names)):
            self.thigh_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, thigh_names[j])

        self.shank_indices = torch.zeros(len(shank_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(shank_names)):
            self.shank_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, shank_names[j])

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(feet_names)):
            self.feet_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, feet_names[j])

        self.sole_indices = torch.zeros(len(sole_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(sole_names)):
            self.sole_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, sole_names[j])

        self.upper_arm_indices = torch.zeros(len(upper_arm_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(upper_arm_names)):
            self.upper_arm_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, upper_arm_names[j])

        self.lower_arm_indices = torch.zeros(len(lower_arm_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(lower_arm_names)):
            self.lower_arm_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, lower_arm_names[j])

        self.hand_indices = torch.zeros(len(hand_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(hand_names)):
            self.hand_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, hand_names[j])

        self.arm_base_indices = torch.zeros(len(arm_base_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(arm_base_names)):
            self.arm_base_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, arm_base_names[j])

        self.arm_end_indices = torch.zeros(len(arm_end_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(arm_end_names)):
            self.arm_end_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, arm_end_names[j])

        print("self.torso_indices: " + str(self.torso_indices))
        print("self.chest_indices: " + str(self.chest_indices))
        print("self.forehead_indices: " + str(self.forehead_indices))

        print("self.imu_indices: " + str(self.imu_indices))

        print("self.waist_indices: " + str(self.waist_indices))
        print("self.head_indices: " + str(self.head_indices))

        print("self.thigh_indices: " + str(self.thigh_indices))
        print("self.shank_indices: " + str(self.shank_indices))
        print("self.feet_indices: " + str(self.feet_indices))
        print("self.sole_indices: " + str(self.sole_indices))

        print("self.upper_arm_indices: " + str(self.upper_arm_indices))
        print("self.lower_arm_indices: " + str(self.lower_arm_indices))
        print("self.hand_indices: " + str(self.hand_indices))

        print("self.arm_base_indices: " + str(self.arm_base_indices))
        print("self.arm_end_indices: " + str(self.arm_end_indices))

    def _init_buffers(self):
        super()._init_buffers()

        # Jason 2023-09-19:
        # change from actor_num_output to num_dof
        self.actions = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        self.dof_pos_leg = torch.zeros(self.num_envs, len(self.leg_indices), dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_vel_leg = torch.zeros(self.num_envs, len(self.leg_indices), dtype=torch.float, device=self.device, requires_grad=False)

        # commands
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_heading = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.ones_like(self.commands, dtype=torch.float, device=self.device, requires_grad=False)

        # resample stand command env_ids
        self.env_ids_for_stand_command = list(range(self.num_envs))
        self.env_ids_for_walk_command = list(range(self.num_envs))

        self._init_buffer_orient()

    def _init_buffers_joint_indices(self):

        # get joint indices
        waist_names = self.cfg.asset.waist_name
        waist_yaw_names = self.cfg.asset.waist_yaw_name
        waist_roll_names = self.cfg.asset.waist_roll_name
        waist_pitch_names = self.cfg.asset.waist_pitch_name
        head_names = self.cfg.asset.head_name
        head_roll_names = self.cfg.asset.head_roll_name
        head_pitch_names = self.cfg.asset.head_pitch_name
        hip_names = self.cfg.asset.hip_name
        hip_roll_names = self.cfg.asset.hip_roll_name
        hip_pitch_names = self.cfg.asset.hip_pitch_name
        hip_yaw_names = self.cfg.asset.hip_yaw_name
        knee_names = self.cfg.asset.knee_name
        ankle_names = self.cfg.asset.ankle_name
        ankle_pitch_names = self.cfg.asset.ankle_pitch_name
        ankle_roll_names = self.cfg.asset.ankle_roll_name
        shoulder_names = self.cfg.asset.shoulder_name
        shoulder_pitch_names = self.cfg.asset.shoulder_pitch_name
        shoulder_roll_names = self.cfg.asset.shoulder_roll_name
        shoulder_yaw_names = self.cfg.asset.shoulder_yaw_name
        elbow_names = self.cfg.asset.elbow_name
        wrist_names = self.cfg.asset.wrist_name
        wrist_yaw_names = self.cfg.asset.wrist_yaw_name
        wrist_roll_names = self.cfg.asset.wrist_roll_name
        wrist_pitch_names = self.cfg.asset.wrist_pitch_name

        self.waist_indices = []
        self.waist_yaw_indices = []
        self.waist_roll_indices = []
        self.waist_pitch_indices = []
        self.head_indices = []
        self.head_roll_indices = []
        self.head_pitch_indices = []
        self.hip_indices = []
        self.hip_roll_indices = []
        self.hip_pitch_indices = []
        self.hip_yaw_indices = []
        self.knee_indices = []
        self.ankle_indices = []
        self.ankle_pitch_indices = []
        self.ankle_roll_indices = []
        self.shoulder_indices = []
        self.shoulder_pitch_indices = []
        self.shoulder_roll_indices = []
        self.shoulder_yaw_indices = []
        self.elbow_indices = []
        self.wrist_indices = []
        self.wrist_yaw_indices = []
        self.wrist_roll_indices = []
        self.wrist_pitch_indices = []

        self.leg_indices = []
        self.arm_indices = []

        self.left_leg_indices = []
        self.right_leg_indices = []
        self.left_arm_indices = []
        self.right_arm_indices = []

        for i in range(self.num_dof):
            name = self.dof_names[i]

            if waist_names in name:
                self.waist_indices.append(i)

            if waist_yaw_names in name:
                self.waist_yaw_indices.append(i)

            if waist_roll_names in name:
                self.waist_roll_indices.append(i)

            if waist_pitch_names in name:
                self.waist_pitch_indices.append(i)

            if head_names in name:
                self.head_indices.append(i)

            if head_roll_names in name:
                self.head_roll_indices.append(i)

            if head_pitch_names in name:
                self.head_pitch_indices.append(i)

            if hip_names in name:
                self.hip_indices.append(i)
                self.leg_indices.append(i)

            if hip_roll_names in name:
                self.hip_roll_indices.append(i)

            if hip_pitch_names in name:
                self.hip_pitch_indices.append(i)

            if hip_yaw_names in name:
                self.hip_yaw_indices.append(i)

            if knee_names in name:
                self.knee_indices.append(i)
                self.leg_indices.append(i)

            if ankle_names in name:
                self.ankle_indices.append(i)
                self.leg_indices.append(i)

            if ankle_pitch_names in name:
                self.ankle_pitch_indices.append(i)

            if ankle_roll_names in name:
                self.ankle_roll_indices.append(i)

            if shoulder_names in name:
                self.shoulder_indices.append(i)
                self.arm_indices.append(i)

            if shoulder_pitch_names in name:
                self.shoulder_pitch_indices.append(i)

            if shoulder_roll_names in name:
                self.shoulder_roll_indices.append(i)

            if shoulder_yaw_names in name:
                self.shoulder_yaw_indices.append(i)

            if elbow_names in name:
                self.elbow_indices.append(i)
                self.arm_indices.append(i)

            if wrist_names in name:
                self.wrist_indices.append(i)
                self.arm_indices.append(i)

            if wrist_yaw_names in name:
                self.wrist_yaw_indices.append(i)

            if wrist_roll_names in name:
                self.wrist_roll_indices.append(i)

            if wrist_pitch_names in name:
                self.wrist_pitch_indices.append(i)

        print("self.waist_indices: " + str(self.waist_indices))
        print("self.waist_yaw_indices: " + str(self.waist_yaw_indices))
        print("self.waist_roll_indices: " + str(self.waist_roll_indices))
        print("self.waist_pitch_indices: " + str(self.waist_pitch_indices))
        print("self.head_indices: " + str(self.head_indices))
        print("self.head_roll_indices: " + str(self.head_roll_indices))
        print("self.head_pitch_indices: " + str(self.head_pitch_indices))

        print("self.hip_indices: " + str(self.hip_indices))
        print("self.hip_roll_indices: " + str(self.hip_roll_indices))
        print("self.hip_pitch_indices: " + str(self.hip_pitch_indices))
        print("self.hip_yaw_indices: " + str(self.hip_yaw_indices))
        print("self.knee_indices: " + str(self.knee_indices))
        print("self.ankle_indices: " + str(self.ankle_indices))
        print("self.ankle_pitch_indices: " + str(self.ankle_pitch_indices))
        print("self.ankle_roll_indices: " + str(self.ankle_roll_indices))
        print("self.shoulder_indices: " + str(self.shoulder_indices))
        print("self.shoulder_pitch_indices: " + str(self.shoulder_pitch_indices))
        print("self.shoulder_roll_indices: " + str(self.shoulder_roll_indices))
        print("self.shoulder_yaw_indices: " + str(self.shoulder_yaw_indices))
        print("self.elbow_indices: " + str(self.elbow_indices))
        print("self.wrist_indices: " + str(self.wrist_indices))
        print("self.wrist_yaw_indices: " + str(self.wrist_yaw_indices))
        print("self.wrist_roll_indices: " + str(self.wrist_roll_indices))
        print("self.wrist_pitch_indices: " + str(self.wrist_pitch_indices))

        print("self.leg_indices: " + str(self.leg_indices))
        print("self.arm_indices: " + str(self.arm_indices))

        self.left_leg_indices = self.leg_indices[:len(self.leg_indices) // 2]
        self.right_leg_indices = self.leg_indices[len(self.leg_indices) // 2:]
        self.left_arm_indices = self.arm_indices[:len(self.arm_indices) // 2]
        self.right_arm_indices = self.arm_indices[len(self.arm_indices) // 2:]

        print("self.left_leg_indices: " + str(self.left_leg_indices))
        print("self.right_leg_indices: " + str(self.right_leg_indices))
        print("self.left_arm_indices: " + str(self.left_arm_indices))
        print("self.right_arm_indices: " + str(self.right_arm_indices))

    def _init_buffers_measure_heights(self):
        super()._init_buffers_measure_heights()

        # measured height supervisor
        if self.cfg.terrain.measure_heights_supervisor:
            self.height_points_supervisor = self._init_height_points_supervisor()
        self.measured_heights_supervisor = None

    def _init_buffer_orient(self):
        self._calculate_feet_orient()
        self._calculate_imu_orient()

    def _parse_cfg(self):

        print("----------------------------------------")

        super()._parse_cfg()

        self.ranges_swing_feet_height = class_to_dict(self.cfg.commands.ranges_swing_feet_height)

    def post_physics_step_update_state(self):
        super().post_physics_step_update_state()

        if self.cfg.terrain.measure_heights_supervisor:
            self.measured_heights_supervisor = self._get_heights_supervisor()

        self._calculate_feet_orient()

    def _calculate_feet_orient(self):
        # feet
        self.left_feet_orient_projected = \
            quat_rotate_inverse(self.rigid_body_states[:, self.feet_indices][:, 0, 3:7], self.gravity_vec)
        self.right_feet_orient_projected = \
            quat_rotate_inverse(self.rigid_body_states[:, self.feet_indices][:, 1, 3:7], self.gravity_vec)
        self.feet_orient_projected = torch.cat((
            self.left_feet_orient_projected.unsqueeze(1),
            self.right_feet_orient_projected.unsqueeze(1)
        ), dim=1)

    def check_termination(self):
        super().check_termination()

        # detect chest tilt too much (roll and pitch)
        if len(self.chest_indices) > 0:
            chest_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.chest_indices][:, 0, 3:7],
                                                          self.gravity_vec)
            self.reset_buf = self.reset_buf | (torch.norm(chest_projected_gravity[:, :2], dim=-1)
                                               > self.cfg.asset.terminate_after_base_projected_gravity_greater_than)

        # detect forehead tilt too much (roll and pitch)
        if len(self.forehead_indices) > 0:
            forehead_projected_gravity = quat_rotate_inverse(
                self.rigid_body_states[:, self.forehead_indices][:, 0, 3:7],
                self.gravity_vec)
            self.reset_buf = self.reset_buf | (torch.norm(forehead_projected_gravity[:, :2], dim=-1)
                                               > self.cfg.asset.terminate_after_base_projected_gravity_greater_than)

    def compute_observation_profile(self):
        self.obs_buf = torch.cat(
            (
                # unobservable proprioception
                # self.base_lin_vel * self.obs_scales.lin_vel * self.lin_vel_scales,
                # self.base_heights_offset.unsqueeze(1) * self.obs_scales.height_measurements,

                # imu related
                # self.imu_ang_vel,
                # self.imu_projected_gravity,

                # base related
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.base_projected_gravity,
                self.commands[:, :3] * self.commands_scale,

                # dof related
                self.dof_pos_offset * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ), dim=-1)

        self.pri_obs_buf = torch.cat(
            (
                # unobservable proprioception
                self.base_lin_vel * self.obs_scales.lin_vel * self.lin_vel_scales,
                self.base_heights_offset.unsqueeze(1) * self.obs_scales.height_measurements,

                # imu related
                # self.imu_ang_vel,
                # self.imu_projected_gravity,

                # base related
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.base_projected_gravity,
                self.commands[:, :3] * self.commands_scale,

                # dof related
                self.dof_pos_offset * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,

                # height related
                self.surround_heights_offset_supervisor,

                # contact
                self.feet_contact,

                # foot height
                self.feet_height,
            ), dim=-1)

    # Jason 2023-11-17
    # 创建 noise vector，此程序只在初始化时调用一次，因此不需要考虑运行效率
    def compute_noise_scale_vec_profile(self):
        noise_vec = torch.zeros_like(self.obs_buf[0])

        # base related
        noise_vec[0 + 0: 0 + 3] = self.noise_scales.ang_vel * self.noise_level * self.obs_scales.ang_vel
        noise_vec[0 + 3: 3 + 3] = self.noise_scales.gravity * self.noise_level
        noise_vec[3 + 3: 6 + 3] = 0.  # commands (3)

        # dof related
        noise_vec[9 + 0 * self.num_dof: 9 + 1 * self.num_dof] = \
            self.noise_scales.dof_pos * self.noise_level * self.obs_scales.dof_pos
        noise_vec[9 + 1 * self.num_dof: 9 + 2 * self.num_dof] = \
            self.noise_scales.dof_vel * self.noise_level * self.obs_scales.dof_vel
        noise_vec[9 + 2 * self.num_dof: 9 + 3 * self.num_dof] = \
            self.noise_scales.action * self.noise_level * self.obs_scales.action

        # print("noise_vec: ", noise_vec)
        return noise_vec

    # ----------------------------------------------

    # 惩罚 上半身 Orientation 不水平
    def _reward_cmd_diff_torso_orient(self):
        if len(self.torso_indices) > 0:
            torso_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.torso_indices][:, 0, 3:7],
                                                          self.gravity_vec)
            error_torso_orient = torch.sum(torch.abs(torso_projected_gravity[:, :2]), dim=1)
            reward_torso_orient = torch.exp(self.cfg.rewards.sigma_cmd_diff_torso_orient
                                            * error_torso_orient)
        else:
            reward_torso_orient = 0
        return reward_torso_orient

    def _reward_cmd_diff_chest_orient(self):
        if len(self.chest_indices) > 0:
            chest_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.chest_indices][:, 0, 3:7],
                                                          self.gravity_vec)
            error_chest_orient = torch.sum(torch.abs(chest_projected_gravity[:, :2]), dim=1)
            reward_chest_orient = torch.exp(self.cfg.rewards.sigma_cmd_diff_chest_orient
                                            * error_chest_orient)
        else:
            reward_chest_orient = 0
        return reward_chest_orient

    def _reward_cmd_diff_forehead_orient(self):
        if len(self.forehead_indices) > 0:
            forehead_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.forehead_indices][:, 0, 3:7],
                                                             self.gravity_vec)
            error_forehead_orient = torch.sum(torch.abs(forehead_projected_gravity[:, :2]), dim=1)
            reward_forehead_orient = torch.exp(self.cfg.rewards.sigma_cmd_diff_forehead_orient
                                               * error_forehead_orient)
        else:
            reward_forehead_orient = 0
        return reward_forehead_orient

    # ----------------------------------------------

    def _reward_action_diff_diff_hip_roll(self):
        error_action_diff = (self.last_actions[:, self.hip_indices] - self.actions[:, self.hip_indices]) \
                            * self.cfg.control.action_scale
        error_action_diff_last = (self.last_last_actions[:, self.hip_indices] - self.last_actions[:, self.hip_indices]) \
                                 * self.cfg.control.action_scale
        error_action_diff_diff = torch.sum(torch.abs(error_action_diff - error_action_diff_last), dim=1)
        reward_action_diff_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff_diff_hip_roll
                                                * error_action_diff_diff)
        return reward_action_diff_diff

    # ----------------------------------------------

    def _reward_dof_tor_new_hip_roll(self):
        error_dof_tor_new = torch.sum(torch.abs(self.torques[:, self.hip_roll_indices]), dim=1)
        reward_dof_tor_new = 1 - torch.exp(self.cfg.rewards.sigma_dof_tor_new_hip_roll
                                           * error_dof_tor_new)
        return reward_dof_tor_new

    # ----------------------------------------------
    def _reward_dof_tor_ankle_feet_lift_up(self):
        left_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 0, 2].unsqueeze(1) - self.measured_heights_supervisor, dim=1)
        right_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 1, 2].unsqueeze(1) - self.measured_heights_supervisor, dim=1)

        error_torques_ankle_left_foot_lift_up = torch.sum(
            torch.abs(self.torques[:, self.ankle_indices[:len(self.ankle_indices) // 2]]), dim=1) \
                                                * torch.abs(left_foot_height) \
                                                * (left_foot_height > (self.swing_feet_height_target.squeeze() / 2))
        error_torques_ankle_right_foot_lift_up = torch.sum(
            torch.abs(self.torques[:, self.ankle_indices[len(self.ankle_indices) // 2:]]), dim=1) \
                                                 * torch.abs(right_foot_height) \
                                                 * (right_foot_height > (self.swing_feet_height_target.squeeze() / 2))

        error_dof_tor_ankle_feet_lift_up = error_torques_ankle_left_foot_lift_up + \
                                           error_torques_ankle_right_foot_lift_up

        reward_dof_tor_ankle_feet_lift_up = 1 - torch.exp(self.cfg.rewards.sigma_dof_tor_ankle_feet_lift_up
                                                          * error_dof_tor_ankle_feet_lift_up)

        return reward_dof_tor_ankle_feet_lift_up

    # ----------------------------------------------

    # 惩罚 脚部 接近地面时不水平
    def _reward_orient_diff_feet_put_down(self):
        left_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 0, 2].unsqueeze(1) - self.measured_heights_supervisor, dim=1)
        right_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 1, 2].unsqueeze(1) - self.measured_heights_supervisor, dim=1)

        # Jason 2023-12-27:
        # normalize the error by the target height
        error_distance_to_ground_left_foot = torch.abs(left_foot_height - self.swing_feet_height_target) \
                                             / self.swing_feet_height_target.squeeze()
        error_distance_to_ground_right_foot = torch.abs(right_foot_height - self.swing_feet_height_target) \
                                              / self.swing_feet_height_target.squeeze()

        error_orient_diff_left_foot = torch.sum(torch.abs(self.left_feet_orient_projected[:, :2]), dim=1) \
                                      * (error_distance_to_ground_left_foot ** 2)
        error_orient_diff_right_foot = torch.sum(torch.abs(self.right_feet_orient_projected[:, :2]), dim=1) \
                                       * (error_distance_to_ground_right_foot ** 2)

        error_orient_diff_feet_put_down = error_orient_diff_left_foot + error_orient_diff_right_foot
        reward_orient_diff_feet_put_down = 1 - torch.exp(self.cfg.rewards.sigma_orient_diff_feet_put_down
                                                         * error_orient_diff_feet_put_down)
        return reward_orient_diff_feet_put_down

    def _reward_orient_diff_feet_lift_up(self):
        left_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 0, 2].unsqueeze(1) - self.measured_heights_supervisor, dim=1)
        right_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 1, 2].unsqueeze(1) - self.measured_heights_supervisor, dim=1)

        # Jason 2023-12-27:
        # normalize the error by the target height
        error_distance_to_height_target_left_foot = torch.abs(left_foot_height * (left_foot_height > 0)) \
                                                    / self.swing_feet_height_target.squeeze()
        error_distance_to_height_target_right_foot = torch.abs(right_foot_height * (right_foot_height > 0)) \
                                                     / self.swing_feet_height_target.squeeze()

        error_orient_diff_left_foot = torch.sum(torch.abs(self.left_feet_orient_projected[:, :2]), dim=1) \
                                      * (error_distance_to_height_target_left_foot ** 2)
        error_orient_diff_right_foot = torch.sum(torch.abs(self.right_feet_orient_projected[:, :2]), dim=1) \
                                       * (error_distance_to_height_target_right_foot ** 2)

        error_orient_diff_feet_lift_up = error_orient_diff_left_foot + error_orient_diff_right_foot
        reward_orient_diff_feet_lift_up = 1 - torch.exp(self.cfg.rewards.sigma_orient_diff_feet_lift_up
                                                        * error_orient_diff_feet_lift_up)
        return reward_orient_diff_feet_lift_up

    # ----------------------------------------------

    def _reward_feet_speed_xy_close_to_ground(self):
        left_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 0, 2].unsqueeze(1) - self.measured_heights_supervisor, dim=1)
        right_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 1, 2].unsqueeze(1) - self.measured_heights_supervisor, dim=1)

        error_left_foot_speed_xy_close_to_ground = \
            torch.norm(self.avg_feet_speed_xyz[:, 0, :2], dim=1) \
            * torch.abs(left_foot_height - self.swing_feet_height_target.squeeze() / 2) \
            * (left_foot_height < self.swing_feet_height_target.squeeze() / 2)
        error_right_foot_speed_xy_close_to_ground = \
            torch.norm(self.avg_feet_speed_xyz[:, 1, :2], dim=1) \
            * torch.abs(right_foot_height - self.swing_feet_height_target.squeeze() / 2) \
            * (right_foot_height < self.swing_feet_height_target.squeeze() / 2)

        error_feet_speed_xy_close_to_ground = error_left_foot_speed_xy_close_to_ground + \
                                              error_right_foot_speed_xy_close_to_ground

        reward_feet_speed_xy_close_to_ground = 1 - torch.exp(self.cfg.rewards.sigma_feet_speed_xy_close_to_ground
                                                             * error_feet_speed_xy_close_to_ground)
        return reward_feet_speed_xy_close_to_ground

    def _reward_feet_speed_z_close_to_height_target(self):
        left_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 0, 2].unsqueeze(1)
            - self.measured_heights_supervisor, dim=1)
        right_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 1, 2].unsqueeze(1)
            - self.measured_heights_supervisor, dim=1)

        error_left_foot_speed_z_close_to_height_target = \
            torch.abs(self.avg_feet_speed_xyz[:, 0, 2]) \
            * torch.abs(left_foot_height - self.swing_feet_height_target.squeeze() / 2) \
            * (left_foot_height > self.swing_feet_height_target.squeeze() / 2)
        error_right_foot_speed_z_close_to_height_target = \
            torch.abs(self.avg_feet_speed_xyz[:, 1, 2]) \
            * torch.abs(right_foot_height - self.swing_feet_height_target.squeeze() / 2) \
            * (right_foot_height > self.swing_feet_height_target.squeeze() / 2)

        error_feet_speed_z_close_to_height_target = error_left_foot_speed_z_close_to_height_target + \
                                                    error_right_foot_speed_z_close_to_height_target

        reward_feet_speed_z_close_to_height_target = 1 - torch.exp(
            self.cfg.rewards.sigma_feet_speed_z_close_to_height_target
            * error_feet_speed_z_close_to_height_target)

        return reward_feet_speed_z_close_to_height_target

    # ----------------------------------------------

    def _reward_feet_air_time(self):
        # 计算 first_contact 的个数，如果有接触到地面，则将 feet_air_time 置为 0
        feet_air_time_error = self.feet_air_time - self.cfg.rewards.feet_air_time_target
        feet_air_time_error = torch.abs(feet_air_time_error)

        reward_feet_air_time = torch.exp(self.cfg.rewards.sigma_feet_air_time
                                         * feet_air_time_error)
        reward_feet_air_time *= self.feet_first_contact
        reward_feet_air_time = torch.sum(reward_feet_air_time, dim=1)
        reward_feet_air_time *= torch.norm(self.commands[:, :2], dim=1) > 0.05  # no reward for zero command

        return reward_feet_air_time

    def _reward_feet_air_height(self):
        left_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 0, 2].unsqueeze(1)
            - self.measured_heights_supervisor,
            dim=1)
        right_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 1, 2].unsqueeze(1)
            - self.measured_heights_supervisor,
            dim=1)

        stack_feet_height = torch.stack((left_foot_height,
                                         right_foot_height))
        min_feet_height, min_feet_height_index = torch.min(stack_feet_height, dim=0)

        error_feet_air_height_left_foot = torch.abs(left_foot_height
                                                    - min_feet_height
                                                    - self.swing_feet_height_target.squeeze())
        error_feet_air_height_right_foot = torch.abs(right_foot_height
                                                     - min_feet_height
                                                     - self.swing_feet_height_target.squeeze())

        # Jason 2023-12-25:
        # 用二次项来描述，更加关注于指定时间段内的高度差
        reward_feet_air_height_left_foot = torch.exp(self.cfg.rewards.sigma_feet_air_height
                                                     * error_feet_air_height_left_foot)
        reward_feet_air_height_right_foot = torch.exp(self.cfg.rewards.sigma_feet_air_height
                                                      * error_feet_air_height_right_foot)

        reward_feet_air_height = torch.stack((reward_feet_air_height_left_foot,
                                              reward_feet_air_height_right_foot), dim=1)

        # Jason 2024-03-31:
        # use air time to catch period at height target
        feet_air_time_mid_error = self.feet_air_time - self.cfg.rewards.feet_air_time_target / 2
        feet_air_time_mid_error = torch.abs(feet_air_time_mid_error)
        feet_air_time_mid_error = torch.exp(self.cfg.rewards.sigma_feet_air_time_mid
                                            * feet_air_time_mid_error)

        reward_feet_air_height = feet_air_time_mid_error * reward_feet_air_height
        reward_feet_air_height = torch.sum(reward_feet_air_height, dim=1)
        reward_feet_air_height *= torch.norm(self.commands[:, :2], dim=1) > 0.05  # no reward for zero command

        return reward_feet_air_height

    def _reward_feet_air_force(self):
        reward_feet_air_force_left_foot = torch.exp(self.cfg.rewards.sigma_feet_force
                                                    * torch.abs(self.avg_feet_contact_force[:, 0]))
        reward_feet_air_force_right_foot = torch.exp(self.cfg.rewards.sigma_feet_force
                                                     * torch.abs(self.avg_feet_contact_force[:, 1]))

        reward_feet_air_force = torch.stack((reward_feet_air_force_left_foot,
                                             reward_feet_air_force_right_foot), dim=1)

        # Jason 2024-03-31:
        # use air time to catch period at height target
        feet_air_time_mid_error = self.feet_air_time - self.cfg.rewards.feet_air_time_target / 2
        feet_air_time_mid_error = torch.abs(feet_air_time_mid_error)
        feet_air_time_mid_error = torch.exp(self.cfg.rewards.sigma_feet_air_time_mid
                                            * feet_air_time_mid_error)

        reward_feet_air_force = feet_air_time_mid_error * reward_feet_air_force
        reward_feet_air_force = torch.sum(reward_feet_air_force, dim=1)
        reward_feet_air_force *= torch.norm(self.commands[:, :2], dim=1) > 0.05  # no reward for zero command

        return reward_feet_air_force

    def _reward_feet_land_time(self):
        # 计算 first_contact 的个数，如果有接触到地面，则将 feet_land_time 置为 0
        feet_land_time_error = (self.feet_land_time - self.cfg.rewards.feet_land_time_max) \
                               * (self.feet_land_time > self.cfg.rewards.feet_land_time_max)

        reward_feet_land_time = 1 - torch.exp(self.cfg.rewards.sigma_feet_land_time
                                              * feet_land_time_error)
        reward_feet_land_time = torch.sum(reward_feet_land_time, dim=1)
        reward_feet_land_time *= torch.norm(self.commands[:, :2], dim=1) > 0.05  # no reward for zero command

        return reward_feet_land_time

    def _reward_on_the_air(self):
        # 惩罚两条腿都没有接触到地面的情况
        jumping_error = torch.sum(self.feet_contact, dim=1) == 0

        # use exponential to make the reward more sparse
        reward_jumping = jumping_error
        return reward_jumping

    # ----------------------------------------------

    def _reward_hip_yaw(self):
        # print(torch.sum(torch.abs(self.dof_pos[:,[1,7]]- self.default_dof_pos[:,[1,7]]),dim=1))
        # print(torch.abs(self.commands[:, 2]))
        return torch.sum(torch.abs(self.dof_pos[:, [1, 7]]), dim=1)

    def _reward_hip_roll(self):
        # print(torch.sum(torch.abs(self.dof_pos[:,[1,7]]- self.default_dof_pos[:,[1,7]]),dim=1))
        # print(torch.abs(self.commands[:, 2]))
        return torch.sum(torch.abs(self.dof_vel[:, [0, 6]]), dim=1)

    # ----------------------------------------------

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        left_foot_fxy = torch.norm(self.contact_forces[:, self.feet_indices][:, 0, :2], dim=1)
        right_foot_fxy = torch.norm(self.contact_forces[:, self.feet_indices][:, 1, :2], dim=1)

        left_foot_fz = self.contact_forces[:, self.feet_indices][:, 0, 2]
        right_foot_fz = self.contact_forces[:, self.feet_indices][:, 1, 2]

        error_left_foot_f = left_foot_fxy - self.cfg.rewards.feet_stumble_ratio * torch.abs(left_foot_fz)
        error_right_foot_f = right_foot_fxy - self.cfg.rewards.feet_stumble_ratio * torch.abs(right_foot_fz)

        error_left_foot_f = error_left_foot_f * (error_left_foot_f > 0)
        error_right_foot_f = error_right_foot_f * (error_right_foot_f > 0)

        # print("error_left_foot_f = \n", error_left_foot_f)

        reward_left_foot_f = 1 - torch.exp(self.cfg.rewards.sigma_feet_stumble * error_left_foot_f)
        reward_right_foot_f = 1 - torch.exp(self.cfg.rewards.sigma_feet_stumble * error_right_foot_f)

        reward_feet_stumble = reward_left_foot_f + reward_right_foot_f
        return reward_feet_stumble
