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

    def _create_envs_get_indices(self, body_names, env_handle, actor_handle):
        """ Creates a list of indices for different bodies of the robot.
        """
        torso_name = [s for s in body_names if self.cfg.asset.torso_name in s]
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

        # change from num_actions to num_dof
        self.actions = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        # commands
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_heading = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.ones_like(self.commands, dtype=torch.float, device=self.device, requires_grad=False)

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

            if hip_roll_names in name:
                self.hip_roll_indices.append(i)

            if hip_pitch_names in name:
                self.hip_pitch_indices.append(i)

            if hip_yaw_names in name:
                self.hip_yaw_indices.append(i)

            if knee_names in name:
                self.knee_indices.append(i)

            if ankle_names in name:
                self.ankle_indices.append(i)

            if ankle_pitch_names in name:
                self.ankle_pitch_indices.append(i)

            if ankle_roll_names in name:
                self.ankle_roll_indices.append(i)

            if shoulder_names in name:
                self.shoulder_indices.append(i)

            if shoulder_pitch_names in name:
                self.shoulder_pitch_indices.append(i)

            if shoulder_roll_names in name:
                self.shoulder_roll_indices.append(i)

            if shoulder_yaw_names in name:
                self.shoulder_yaw_indices.append(i)

            if elbow_names in name:
                self.elbow_indices.append(i)

            if wrist_names in name:
                self.wrist_indices.append(i)

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

    def compute_observation_profile(self):
        self.obs_buf = torch.cat(
            (
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
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_heights_offset.unsqueeze(1) * self.obs_scales.height_measurements,

                # base related
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.base_projected_gravity,
                self.commands[:, :3] * self.commands_scale,

                # dof related
                self.dof_pos_offset * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,

                # height related
                self.surround_heights_offset,

                # contact
                self.feet_contact,

                # foot height
                self.feet_height,
            ), dim=-1)

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

    def _reward_action_diff_knee(self):
        error_action_diff = (self.actions[:, self.knee_indices] - self.last_actions[:, self.knee_indices]) \
                            * self.cfg.control.action_scale
        error_action_diff = torch.sum(torch.abs(error_action_diff), dim=1)
        reward_knee_action_diff = 1 - torch.exp(self.cfg.rewards.sigma_action_diff_knee
                                                * error_action_diff)
        return reward_knee_action_diff

    # ----------------------------------------------

    def _reward_dof_vel_new_knee(self):
        error_new_dof_vel = torch.sum(torch.abs(self.dof_vel[:, self.knee_indices]), dim=1)
        reward_new_dof_vel = 1 - torch.exp(self.cfg.rewards.sigma_dof_vel_new_knee
                                           * error_new_dof_vel)
        return reward_new_dof_vel

    # ----------------------------------------------

    def _reward_dof_tor_new_hip_roll(self):
        error_dof_tor_new = torch.sum(torch.abs(self.torques[:, self.hip_roll_indices]), dim=1)
        reward_dof_tor_new = 1 - torch.exp(self.cfg.rewards.sigma_dof_tor_new_hip_roll
                                           * error_dof_tor_new)
        return reward_dof_tor_new

    # ----------------------------------------------

    def _reward_pose_offset_hip_yaw(self):
        error_pose_offset = torch.sum(torch.abs(self.dof_pos[:, self.hip_yaw_indices] - self.default_dof_pos[:, self.hip_yaw_indices]), dim=1)
        reward_pose_offset = 1 - torch.exp(self.cfg.rewards.sigma_pose_offset_hip_yaw
                                           * error_pose_offset)
        return reward_pose_offset

    # ----------------------------------------------

    def _reward_dof_tor_ankle_feet_lift_up(self):
        left_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 0, 2].unsqueeze(1) - self.measured_heights, dim=1)
        right_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 1, 2].unsqueeze(1) - self.measured_heights, dim=1)

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

    def _reward_feet_speed_xy_close_to_ground(self):
        left_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 0, 2].unsqueeze(1) - self.measured_heights, dim=1)
        right_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 1, 2].unsqueeze(1) - self.measured_heights, dim=1)

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
            - self.measured_heights, dim=1)
        right_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 1, 2].unsqueeze(1)
            - self.measured_heights, dim=1)

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
        feet_air_time_error = self.feet_air_time - self.cfg.rewards.feet_air_time_target
        feet_air_time_error = torch.abs(feet_air_time_error)

        reward_feet_air_time = torch.exp(self.cfg.rewards.sigma_feet_air_time
                                         * feet_air_time_error)
        reward_feet_air_time *= self.feet_first_contact
        reward_feet_air_time = torch.sum(reward_feet_air_time, dim=1)
        reward_feet_air_time *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command

        return reward_feet_air_time

    def _reward_feet_air_height(self):
        left_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 0, 2].unsqueeze(1)
            - self.measured_heights,
            dim=1)
        right_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 1, 2].unsqueeze(1)
            - self.measured_heights,
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
        error_feet_air_height = torch.stack([error_feet_air_height_left_foot,
                                             error_feet_air_height_right_foot], dim=1)

        feet_air_time_mid_error = self.feet_air_time - self.cfg.rewards.feet_air_time_target / 2
        feet_air_time_mid_error = torch.abs(feet_air_time_mid_error)

        reward_feet_air_height = torch.sum(feet_air_time_mid_error * error_feet_air_height, dim=1)
        reward_feet_air_height = torch.exp(self.cfg.rewards.sigma_feet_air_height * reward_feet_air_height)
        reward_feet_air_height *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command

        return reward_feet_air_height

    def _reward_feet_air_force(self):
        reward_feet_air_force_left_foot = self.avg_feet_contact_force[:, 0]
        reward_feet_air_force_right_foot = self.avg_feet_contact_force[:, 1]

        reward_feet_air_force = torch.stack((reward_feet_air_force_left_foot,
                                             reward_feet_air_force_right_foot), dim=1)

        feet_air_time_mid_error = self.feet_air_time - self.cfg.rewards.feet_air_time_target / 2
        feet_air_time_mid_error = torch.abs(feet_air_time_mid_error)

        reward_feet_air_force = feet_air_time_mid_error * reward_feet_air_force
        reward_feet_air_force = torch.sum(reward_feet_air_force, dim=1)
        reward_feet_air_force = torch.exp(self.cfg.rewards.sigma_feet_air_force * reward_feet_air_force)
        reward_feet_air_force *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command

        return reward_feet_air_force

    def _reward_feet_land_time(self):
        feet_land_time_error = (self.feet_land_time - self.cfg.rewards.feet_land_time_max) \
                               * (self.feet_land_time > self.cfg.rewards.feet_land_time_max)

        reward_feet_land_time = 1 - torch.exp(self.cfg.rewards.sigma_feet_land_time
                                              * feet_land_time_error)
        reward_feet_land_time = torch.sum(reward_feet_land_time, dim=1)
        reward_feet_land_time *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command

        return reward_feet_land_time

    def _reward_on_the_air(self):
        jumping_error = torch.sum(self.feet_contact, dim=1) == 0

        # use exponential to make the reward more sparse
        reward_jumping = jumping_error
        return reward_jumping

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

        reward_left_foot_f = 1 - torch.exp(self.cfg.rewards.sigma_feet_stumble * error_left_foot_f)
        reward_right_foot_f = 1 - torch.exp(self.cfg.rewards.sigma_feet_stumble * error_right_foot_f)

        reward_feet_stumble = reward_left_foot_f + reward_right_foot_f
        return reward_feet_stumble
