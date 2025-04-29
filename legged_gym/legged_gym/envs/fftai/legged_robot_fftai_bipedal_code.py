import torch

from isaacgym.torch_utils import *

from legged_gym.envs.fftai.legged_robot_fftai_code import LeggedRobotFFTAI
from legged_gym.envs.fftai.legged_robot_fftai_bipedal_config import LeggedRobotFFTAIBipedalCfg


class LeggedRobotFFTAIBipedal(LeggedRobotFFTAI):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg: LeggedRobotFFTAIBipedalCfg = cfg

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_cfg(self, cfg: LeggedRobotFFTAIBipedalCfg):
        super()._init_cfg(cfg)

    # ----------------------------------------------

    def _create_envs_get_indices(self, body_names, env_handle, actor_handle):
        """
        Creates a list of indices for different bodies of the robot.

        主要针对的机器人的连杆结构进行了索引的创建
        """
        base_name = [s for s in body_names if self.cfg.asset.base_name in s]
        torso_name = [s for s in body_names if self.cfg.asset.torso_name in s]
        chest_name = [s for s in body_names if self.cfg.asset.chest_name in s]
        forehead_indices = [s for s in body_names if self.cfg.asset.forehead_name in s]

        self.base_indices = torch.zeros(len(base_name), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(base_name)):
            self.base_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, base_name[j])

        self.torso_indices = torch.zeros(len(torso_name), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(torso_name)):
            self.torso_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, torso_name[j])

        self.chest_indices = torch.zeros(len(chest_name), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(chest_name)):
            self.chest_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, chest_name[j])

        self.forehead_indices = torch.zeros(len(forehead_indices), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(forehead_indices)):
            self.forehead_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, forehead_indices[j])

        # ----------------------------------------------

        imu_name = [s for s in body_names if self.cfg.asset.imu_name in s]

        self.imu_indices = torch.zeros(len(imu_name), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(imu_name)):
            self.imu_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, imu_name[j])

        # ----------------------------------------------

        waist_names = [s for s in body_names if self.cfg.asset.waist_name in s]
        waist_roll_names = [s for s in body_names if self.cfg.asset.waist_roll_name in s]
        waist_pitch_names = [s for s in body_names if self.cfg.asset.waist_pitch_name in s]
        waist_yaw_names = [s for s in body_names if self.cfg.asset.waist_yaw_name in s]

        self.waist_indices = torch.zeros(len(waist_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(waist_names)):
            self.waist_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, waist_names[j])

        self.waist_roll_indices = torch.zeros(len(waist_roll_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(waist_roll_names)):
            self.waist_roll_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, waist_roll_names[j])

        self.waist_pitch_indices = torch.zeros(len(waist_pitch_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(waist_pitch_names)):
            self.waist_pitch_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, waist_pitch_names[j])

        self.waist_yaw_indices = torch.zeros(len(waist_yaw_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(waist_yaw_names)):
            self.waist_yaw_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, waist_yaw_names[j])

        # ----------------------------------------------

        head_names = [s for s in body_names if self.cfg.asset.head_name in s]
        head_roll_names = [s for s in body_names if self.cfg.asset.head_roll_name in s]
        head_pitch_names = [s for s in body_names if self.cfg.asset.head_pitch_name in s]
        head_yaw_names = [s for s in body_names if self.cfg.asset.head_yaw_name in s]

        self.head_indices = torch.zeros(len(head_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(head_names)):
            self.head_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, head_names[j])

        self.head_roll_indices = torch.zeros(len(head_roll_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(head_roll_names)):
            self.head_roll_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, head_roll_names[j])

        self.head_pitch_indices = torch.zeros(len(head_pitch_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(head_pitch_names)):
            self.head_pitch_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, head_pitch_names[j])

        self.head_yaw_indices = torch.zeros(len(head_yaw_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(head_yaw_names)):
            self.head_yaw_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, head_yaw_names[j])

        # ----------------------------------------------

        thigh_names = [s for s in body_names if self.cfg.asset.thigh_name in s]
        shank_names = [s for s in body_names if self.cfg.asset.shank_name in s]
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        sole_names = [s for s in body_names if self.cfg.asset.sole_name in s]

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

        # ----------------------------------------------

        upper_arm_names = [s for s in body_names if self.cfg.asset.upper_arm_name in s]
        lower_arm_names = [s for s in body_names if self.cfg.asset.lower_arm_name in s]
        hand_names = [s for s in body_names if self.cfg.asset.hand_name in s]
        end_effector_names = [s for s in body_names if self.cfg.asset.end_effector_name in s]

        self.upper_arm_indices = torch.zeros(len(upper_arm_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(upper_arm_names)):
            self.upper_arm_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, upper_arm_names[j])

        self.lower_arm_indices = torch.zeros(len(lower_arm_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(lower_arm_names)):
            self.lower_arm_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, lower_arm_names[j])

        self.hand_indices = torch.zeros(len(hand_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(hand_names)):
            self.hand_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, hand_names[j])

        self.end_effector_indices = torch.zeros(len(end_effector_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(end_effector_names)):
            self.end_effector_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, end_effector_names[j])

        # ----------------------------------------------

        payload_names = [s for s in body_names if self.cfg.asset.payload_name in s]

        self.payload_indices = torch.zeros(len(payload_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(payload_names)):
            self.payload_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, payload_names[j])

        # ----------------------------------------------

        arm_base_names = [s for s in body_names if self.cfg.asset.arm_base_name in s]
        arm_end_names = [s for s in body_names if self.cfg.asset.arm_end_name in s]

        self.arm_base_indices = torch.zeros(len(arm_base_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(arm_base_names)):
            self.arm_base_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, arm_base_names[j])

        self.arm_end_indices = torch.zeros(len(arm_end_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(arm_end_names)):
            self.arm_end_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, arm_end_names[j])

        # ----------------------------------------------

        print("##############################################")
        print("\033[1m", "Body Indices:", "\033[0m")
        print("----------------------------------------------")
        print("self.base_indices: " + str(self.base_indices))
        print("self.torso_indices: " + str(self.torso_indices))
        print("self.chest_indices: " + str(self.chest_indices))
        print("self.forehead_indices: " + str(self.forehead_indices))
        print("----------------------------------------------")
        print("self.imu_indices: " + str(self.imu_indices))
        print("----------------------------------------------")
        print("self.waist_indices: " + str(self.waist_indices))
        print("self.waist_roll_indices: " + str(self.waist_roll_indices))
        print("self.waist_pitch_indices: " + str(self.waist_pitch_indices))
        print("self.waist_yaw_indices: " + str(self.waist_yaw_indices))
        print("----------------------------------------------")
        print("self.head_indices: " + str(self.head_indices))
        print("self.head_roll_indices: " + str(self.head_roll_indices))
        print("self.head_pitch_indices: " + str(self.head_pitch_indices))
        print("self.head_yaw_indices: " + str(self.head_yaw_indices))
        print("----------------------------------------------")
        print("self.thigh_indices: " + str(self.thigh_indices))
        print("self.shank_indices: " + str(self.shank_indices))
        print("self.feet_indices: " + str(self.feet_indices))
        print("self.sole_indices: " + str(self.sole_indices))
        print("----------------------------------------------")
        print("self.upper_arm_indices: " + str(self.upper_arm_indices))
        print("self.lower_arm_indices: " + str(self.lower_arm_indices))
        print("self.hand_indices: " + str(self.hand_indices))
        print("self.end_effector_indices: " + str(self.end_effector_indices))
        print("----------------------------------------------")
        print("self.payload_indices: " + str(self.payload_indices))
        print("----------------------------------------------")
        print("self.arm_base_indices: " + str(self.arm_base_indices))
        print("self.arm_end_indices: " + str(self.arm_end_indices))
        print("##############################################")
        print("\033[0m")

    def _init_buffers_others(self):
        super()._init_buffers_others()

        # env_ids for different commands
        self.env_ids_of_walk_command = torch.arange(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)

        # contact
        self.contact_forces_limit = self.robot_mass \
                                    * 9.81 \
                                    * self.cfg.rewards.contact_force_limit_ratio  # change based on robot_mass

    # ----------------------------------------------

    def _init_buffers_joint_indices(self):

        # get joint indices
        waist_names = self.cfg.asset.waist_name
        waist_roll_names = self.cfg.asset.waist_roll_name
        waist_pitch_names = self.cfg.asset.waist_pitch_name
        waist_yaw_names = self.cfg.asset.waist_yaw_name

        head_names = self.cfg.asset.head_name
        head_roll_names = self.cfg.asset.head_roll_name
        head_pitch_names = self.cfg.asset.head_pitch_name
        head_yaw_names = self.cfg.asset.head_yaw_name

        hip_names = self.cfg.asset.hip_name
        hip_roll_names = self.cfg.asset.hip_roll_name
        hip_pitch_names = self.cfg.asset.hip_pitch_name
        hip_yaw_names = self.cfg.asset.hip_yaw_name
        knee_names = self.cfg.asset.knee_name
        knee_pitch_names = self.cfg.asset.knee_pitch_name
        ankle_names = self.cfg.asset.ankle_name
        ankle_roll_names = self.cfg.asset.ankle_roll_name
        ankle_pitch_names = self.cfg.asset.ankle_pitch_name

        shoulder_names = self.cfg.asset.shoulder_name
        shoulder_pitch_names = self.cfg.asset.shoulder_pitch_name
        shoulder_roll_names = self.cfg.asset.shoulder_roll_name
        shoulder_yaw_names = self.cfg.asset.shoulder_yaw_name
        elbow_names = self.cfg.asset.elbow_name
        wrist_names = self.cfg.asset.wrist_name
        wrist_roll_names = self.cfg.asset.wrist_roll_name
        wrist_pitch_names = self.cfg.asset.wrist_pitch_name
        wrist_yaw_names = self.cfg.asset.wrist_yaw_name

        upper_Limb_joint_names = self.cfg.asset.upper_limb_joint_names
        lower_Limb_joint_names = self.cfg.asset.lower_limb_joint_names
        main_body_joint_names = self.cfg.asset.main_body_joint_names

        self.waist_indices = []
        self.waist_roll_indices = []
        self.waist_pitch_indices = []
        self.waist_yaw_indices = []

        self.head_indices = []
        self.head_roll_indices = []
        self.head_pitch_indices = []
        self.head_yaw_indices = []

        self.hip_indices = []
        self.hip_roll_indices = []
        self.hip_pitch_indices = []
        self.hip_yaw_indices = []
        self.knee_indices = []
        self.knee_pitch_indices = []
        self.ankle_indices = []
        self.ankle_roll_indices = []
        self.ankle_pitch_indices = []

        self.shoulder_indices = []
        self.shoulder_roll_indices = []
        self.shoulder_pitch_indices = []
        self.shoulder_yaw_indices = []
        self.elbow_indices = []
        self.wrist_indices = []
        self.wrist_roll_indices = []
        self.wrist_pitch_indices = []
        self.wrist_yaw_indices = []

        self.upper_Limb_joint_indices = []
        self.lower_Limb_joint_indices = []
        self.main_body_joint_indices = []

        for i in range(self.num_dofs):
            name = self.dof_names[i]

            if waist_names in name:
                self.waist_indices.append(i)

            if waist_roll_names in name:
                self.waist_roll_indices.append(i)

            if waist_pitch_names in name:
                self.waist_pitch_indices.append(i)

            if waist_yaw_names in name:
                self.waist_yaw_indices.append(i)

            if head_names in name:
                self.head_indices.append(i)

            if head_roll_names in name:
                self.head_roll_indices.append(i)

            if head_pitch_names in name:
                self.head_pitch_indices.append(i)

            if head_yaw_names in name:
                self.head_yaw_indices.append(i)

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

            if knee_pitch_names in name:
                self.knee_pitch_indices.append(i)

            if ankle_names in name:
                self.ankle_indices.append(i)

            if ankle_roll_names in name:
                self.ankle_roll_indices.append(i)

            if ankle_pitch_names in name:
                self.ankle_pitch_indices.append(i)

            if shoulder_names in name:
                self.shoulder_indices.append(i)

            if shoulder_roll_names in name:
                self.shoulder_roll_indices.append(i)

            if shoulder_pitch_names in name:
                self.shoulder_pitch_indices.append(i)

            if shoulder_yaw_names in name:
                self.shoulder_yaw_indices.append(i)

            if elbow_names in name:
                self.elbow_indices.append(i)

            if wrist_names in name:
                self.wrist_indices.append(i)

            if wrist_roll_names in name:
                self.wrist_roll_indices.append(i)

            if wrist_pitch_names in name:
                self.wrist_pitch_indices.append(i)

            if wrist_yaw_names in name:
                self.wrist_yaw_indices.append(i)

            for j in range(len(upper_Limb_joint_names)):
                if upper_Limb_joint_names[j] in name:
                    self.upper_Limb_joint_indices.append(i)

            for j in range(len(lower_Limb_joint_names)):
                if lower_Limb_joint_names[j] in name:
                    self.lower_Limb_joint_indices.append(i)

            for j in range(len(main_body_joint_names)):
                if main_body_joint_names[j] in name:
                    self.main_body_joint_indices.append(i)

        print("##############################################")
        print("\033[1m", "Joint Indices:", "\033[0m")
        print("----------------------------------------------")
        print("self.waist_indices: " + str(self.waist_indices))
        print("self.waist_roll_indices: " + str(self.waist_roll_indices))
        print("self.waist_pitch_indices: " + str(self.waist_pitch_indices))
        print("self.waist_yaw_indices: " + str(self.waist_yaw_indices))
        print("----------------------------------------------")
        print("self.head_indices: " + str(self.head_indices))
        print("self.head_roll_indices: " + str(self.head_roll_indices))
        print("self.head_pitch_indices: " + str(self.head_pitch_indices))
        print("self.head_yaw_indices: " + str(self.head_yaw_indices))
        print("----------------------------------------------")
        print("self.hip_indices: " + str(self.hip_indices))
        print("self.hip_roll_indices: " + str(self.hip_roll_indices))
        print("self.hip_pitch_indices: " + str(self.hip_pitch_indices))
        print("self.hip_yaw_indices: " + str(self.hip_yaw_indices))
        print("self.knee_indices: " + str(self.knee_indices))
        print("self.knee_pitch_indices: " + str(self.knee_pitch_indices))
        print("self.ankle_indices: " + str(self.ankle_indices))
        print("self.ankle_roll_indices: " + str(self.ankle_roll_indices))
        print("self.ankle_pitch_indices: " + str(self.ankle_pitch_indices))
        print("----------------------------------------------")
        print("self.shoulder_indices: " + str(self.shoulder_indices))
        print("self.shoulder_pitch_indices: " + str(self.shoulder_pitch_indices))
        print("self.shoulder_roll_indices: " + str(self.shoulder_roll_indices))
        print("self.shoulder_yaw_indices: " + str(self.shoulder_yaw_indices))
        print("self.elbow_indices: " + str(self.elbow_indices))
        print("self.wrist_indices: " + str(self.wrist_indices))
        print("self.wrist_roll_indices: " + str(self.wrist_roll_indices))
        print("self.wrist_pitch_indices: " + str(self.wrist_pitch_indices))
        print("self.wrist_yaw_indices: " + str(self.wrist_yaw_indices))
        print("----------------------------------------------")
        print("self.upper_Limb_joint_indices: " + str(self.upper_Limb_joint_indices))
        print("self.lower_Limb_joint_indices: " + str(self.lower_Limb_joint_indices))
        print("self.main_body_joint_indices: " + str(self.main_body_joint_indices))
        print("##############################################")
        print("\033[0m")

        # ----------------------------------------------

        self.position_control_indices = []
        self.velocity_control_indices = []
        self.torque_control_indices = []
        self.unknown_control_indices = []

        for i in range(self.num_dofs):
            control_type = self.control_types[i]

            if control_type == 1:
                self.position_control_indices.append(i)
            elif control_type == 2:
                self.velocity_control_indices.append(i)
            elif control_type == 3:
                self.torque_control_indices.append(i)
            else:
                self.unknown_control_indices.append(i)

        print("##############################################")
        print("\033[1m", "Control Indices:", "\033[0m")
        print("----------------------------------------------")
        print("self.position_control_indices: " + str(self.position_control_indices))
        print("self.velocity_control_indices: " + str(self.velocity_control_indices))
        print("self.torque_control_indices: " + str(self.torque_control_indices))
        print("self.unknown_control_indices: " + str(self.unknown_control_indices))
        print("##############################################")
        print("\033[0m")

    # ----------------------------------------------

    def get_left_foot_height(self):
        left_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 0, 2:3]
            - self.cfg.asset.foot_thickness
            - self.measured_heights,
            dim=1).unsqueeze(1)

        return left_foot_height

    def get_right_foot_height(self):
        right_foot_height = torch.mean(
            self.rigid_body_states[:, self.feet_indices][:, 1, 2:3]
            - self.cfg.asset.foot_thickness
            - self.measured_heights,
            dim=1).unsqueeze(1)

        return right_foot_height

    def get_left_foot_height_target(self):
        left_foot_height_target = 0.0

        return left_foot_height_target

    def get_right_foot_height_target(self):
        right_foot_height_target = 0.0

        return right_foot_height_target

    def get_left_foot_height_max(self):
        left_foot_height_max = 0.1

        return left_foot_height_max

    def get_right_foot_height_max(self):
        right_foot_height_max = 0.1

        return right_foot_height_max

    # ==========================================================================================================================
    # Reward functions

    def _reward_stand_still_dof_pos_waist_joint(self):
        """
        Penalize not standing still
        """
        error_stand_still_pos = torch.abs(self.dof_pos[:, self.waist_indices]
                                          - self.default_dof_pos_tenors[:, self.waist_indices])
        error_stand_still_pos = torch.sum(error_stand_still_pos, dim=1)  # dims 2->1
        reward_stand_still_pos = torch.exp(self.cfg.rewards.sigma_stand_still_dof_pos
                                           * error_stand_still_pos)

        # ----------------------------

        """
        Jason 2024-03-23:
        Only apply the reward to the environment that is in the stand state
        """
        selector_stand_still = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)  # dims 1
        selector_stand_still[self.env_ids_of_stand_command] = 1

        # ----------------------------

        reward_stand_still_pos *= selector_stand_still

        return reward_stand_still_pos

    def _reward_stand_still_foot_distance(self):
        """
        Reward for tracking the foot distance match with the stand command,
        when the gait is in the stand state (two feet on the ground).
        """

        left_foot_pos_in_world_frame = self.rigid_body_states[:, self.feet_indices][:, 0, 0:3]
        right_foot_pos_in_world_frame = self.rigid_body_states[:, self.feet_indices][:, 1, 0:3]

        left_foot_pos_to_base_in_world_frame = left_foot_pos_in_world_frame - self.root_states[:, 0:3]
        right_foot_pos_to_base_in_world_frame = right_foot_pos_in_world_frame - self.root_states[:, 0:3]

        left_foot_pos = quat_rotate_inverse(self.root_states[:, 3:7], left_foot_pos_to_base_in_world_frame)
        right_foot_pos = quat_rotate_inverse(self.root_states[:, 3:7], right_foot_pos_to_base_in_world_frame)

        error_foot_distance_y = torch.abs(left_foot_pos[:, 1:2] - right_foot_pos[:, 1:2])
        error_foot_distance_y = torch.abs(error_foot_distance_y - self.cfg.rewards.stand_still_foot_distance)
        error_foot_distance_y = torch.sum(error_foot_distance_y, dim=1)  # dims 2->1

        reward_stand_still_foot_distance = torch.exp(self.cfg.rewards.sigma_stand_still_foot_distance
                                                     * error_foot_distance_y)

        # ----------------------------

        condition_left_foot_contact = self.feet_contact[:, 0]  # dims 1
        condition_left_foot_contact_trig = self.feet_contact_trig[:, 0]  # dims 1
        condition_right_foot_contact = self.feet_contact[:, 1]  # dims 1
        condition_right_foot_contact_trig = self.feet_contact_trig[:, 1]  # dims 1
        condition_both_feet_contact = condition_left_foot_contact * condition_right_foot_contact_trig \
                                      + condition_left_foot_contact_trig * condition_right_foot_contact

        # ----------------------------

        """
        Jason 2024-03-23:
        Only apply the reward to the environment that is in the stand state
        """
        selector_stand = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        selector_stand[self.env_ids_of_stand_command] = 1

        # ----------------------------

        reward_stand_still_foot_distance *= condition_both_feet_contact
        reward_stand_still_foot_distance *= selector_stand

        return reward_stand_still_foot_distance

    # ----------------------------------------------

    def _reward_limits_dof_pos_without_ankle(self):
        """
        Reward for pass the dof position limits
        """

        # ----------------------------------------------
        # get all controllable joints
        related_indexes = list(range(self.num_actions))

        # remove ankle joints, because when step on the ground, the position is not controllable
        related_indexes = [i for i in related_indexes if i not in self.ankle_indices]
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

    def _reward_limits_dof_vel_without_ankle(self):
        """
        Reward for pass the dof velocity limits
        """

        # ----------------------------------------------
        # get all controllable joints
        related_indexes = list(range(self.num_actions))

        # remove ankle joints, because when step on the ground, the velocity is not controllable
        related_indexes = [i for i in related_indexes if i not in self.ankle_indices]
        # ----------------------------------------------

        error_limits_dof_vel = (torch.abs(self.dof_vel[:, related_indexes])
                                - self.soft_dof_vel_limits[related_indexes]).clip(min=0.)
        error_limits_dof_vel = torch.sum(error_limits_dof_vel, dim=1)  # dims 2->1

        reward_limits_dof_vel = 1 - torch.exp(self.cfg.rewards.sigma_limits_dof_vel
                                              * error_limits_dof_vel)

        return reward_limits_dof_vel

    # ----------------------------------------------

    def _reward_feet_distance_too_close(self):
        """
        Penalty for the distance between the feet being too close.

        Jason 2025-02-16:
        这个奖赏是为了防止左右脚的交叉运动，防止互相踩脚
        """
        left_foot_pos_in_world_frame = self.rigid_body_states[:, self.feet_indices][:, 0, 0:3]
        right_foot_pos_in_world_frame = self.rigid_body_states[:, self.feet_indices][:, 1, 0:3]

        left_foot_pos_to_base_in_world_frame = left_foot_pos_in_world_frame - self.root_states[:, 0:3]
        right_foot_pos_to_base_in_world_frame = right_foot_pos_in_world_frame - self.root_states[:, 0:3]

        left_foot_pos = quat_rotate_inverse(self.root_states[:, 3:7], left_foot_pos_to_base_in_world_frame)
        right_foot_pos = quat_rotate_inverse(self.root_states[:, 3:7], right_foot_pos_to_base_in_world_frame)

        foot_distance_x = torch.abs(left_foot_pos[:, 0:1] - right_foot_pos[:, 0:1])
        foot_distance_y = torch.abs(left_foot_pos[:, 1:2] - right_foot_pos[:, 1:2])
        foot_distance_z = torch.abs(left_foot_pos[:, 2:3] - right_foot_pos[:, 2:3])

        foot_distance = torch.sqrt(foot_distance_x ** 2 + foot_distance_y ** 2 + foot_distance_z ** 2)

        error_foot_distance_too_close = torch.abs(foot_distance - self.cfg.rewards.feet_distance_too_close) \
                                        * (foot_distance < self.cfg.rewards.feet_distance_too_close)
        error_foot_distance_too_close = torch.sum(error_foot_distance_too_close, dim=1)  # dims 2->1

        reward_feet_distance_too_close = 1 - torch.exp(self.cfg.rewards.sigma_feet_distance_too_close
                                                       * error_foot_distance_too_close)

        return reward_feet_distance_too_close

    def _reward_feet_distance_y_too_close(self):
        """
        Penalty for the distance between the feet in the y direction being too close.

        Jason 2025-02-16:
        这个奖赏是为了防止左右脚的交叉运动，防止互相踩脚 y 侧方向
        """
        left_foot_pos_in_world_frame = self.rigid_body_states[:, self.feet_indices][:, 0, 0:3]
        right_foot_pos_in_world_frame = self.rigid_body_states[:, self.feet_indices][:, 1, 0:3]

        left_foot_pos_to_base_in_world_frame = left_foot_pos_in_world_frame - self.root_states[:, 0:3]
        right_foot_pos_to_base_in_world_frame = right_foot_pos_in_world_frame - self.root_states[:, 0:3]

        left_foot_pos = quat_rotate_inverse(self.root_states[:, 3:7], left_foot_pos_to_base_in_world_frame)
        right_foot_pos = quat_rotate_inverse(self.root_states[:, 3:7], right_foot_pos_to_base_in_world_frame)

        foot_distance_y = torch.abs(left_foot_pos[:, 1:2] - right_foot_pos[:, 1:2])

        error_foot_distance_y_too_close = torch.abs(foot_distance_y - self.cfg.rewards.feet_distance_y_too_close) \
                                          * (foot_distance_y < self.cfg.rewards.feet_distance_y_too_close)
        error_foot_distance_y_too_close = torch.sum(error_foot_distance_y_too_close, dim=1)

        reward_feet_distance_y_too_close = 1 - torch.exp(self.cfg.rewards.sigma_feet_distance_y_too_close
                                                         * error_foot_distance_y_too_close)

        return reward_feet_distance_y_too_close

    # ----------------------------------------------

    def _reward_feet_speed_xy_close_to_ground(self):
        """
        Reward for keeping the feet speed close to the ground.
        """
        left_foot_height = self.get_left_foot_height()
        right_foot_height = self.get_right_foot_height()

        left_foot_height_max = self.get_left_foot_height_max()
        right_foot_height_max = self.get_right_foot_height_max()

        error_left_foot_close_to_ground = \
            torch.abs(left_foot_height - left_foot_height_max / 4) \
            * (left_foot_height < left_foot_height_max / 4) \
            / (left_foot_height_max / 4)
        error_right_foot_close_to_ground = \
            torch.abs(right_foot_height - right_foot_height_max / 4) \
            * (right_foot_height < right_foot_height_max / 4) \
            / (right_foot_height_max / 4)

        error_left_foot_speed_xy_close_to_ground = \
            torch.norm(self.avg_feet_speed_xyz[:, 0, 0:2], dim=1).unsqueeze(1) \
            * error_left_foot_close_to_ground
        error_right_foot_speed_xy_close_to_ground = \
            torch.norm(self.avg_feet_speed_xyz[:, 1, 0:2], dim=1).unsqueeze(1) \
            * error_right_foot_close_to_ground

        error_feet_speed_xy_close_to_ground = error_left_foot_speed_xy_close_to_ground + \
                                              error_right_foot_speed_xy_close_to_ground
        error_feet_speed_xy_close_to_ground = torch.sum(error_feet_speed_xy_close_to_ground, dim=1)  # dims 2->1

        reward_feet_speed_xy_close_to_ground = torch.exp(self.cfg.rewards.sigma_feet_speed_xy_close_to_ground
                                                         * error_feet_speed_xy_close_to_ground)
        return reward_feet_speed_xy_close_to_ground

    def _reward_feet_force_z_close_to_ground(self):
        """
        Reward for keeping the feet close to the ground.

        Jason 2024-11-12:
        Too high penalty, will cause the robot try to use its foot edge to contact with the ground.
        过高的接近地面 z 轴接触力惩罚会导致机器人尝试把脚侧向抬高，翘起来走路。
        """
        left_foot_height = self.get_left_foot_height()
        right_foot_height = self.get_right_foot_height()

        left_foot_height_max = self.get_left_foot_height_max()
        right_foot_height_max = self.get_right_foot_height_max()

        """
        Jason 2025-01-01:
        这里计算的 left_foot_height - left_foot_height_max / 4，可能会导致机器人尝试抬高脚面，减少高度差，从而降低 error。
        """
        error_left_foot_close_to_ground = \
            torch.abs(left_foot_height - left_foot_height_max / 4) \
            * (left_foot_height < left_foot_height_max / 4) \
            / (left_foot_height_max / 4)
        error_right_foot_close_to_ground = \
            torch.abs(right_foot_height - right_foot_height_max / 4) \
            * (right_foot_height < right_foot_height_max / 4) \
            / (right_foot_height_max / 4)

        error_left_foot_force_z_close_to_ground = \
            torch.clip(self.contact_forces[:, self.feet_indices][:, 0, 2:3]
                       - self.contact_forces_limit
                       * self.cfg.rewards.feet_force_z_close_to_ground_contact_force_limit_ratio,
                       min=0) \
            * error_left_foot_close_to_ground
        error_right_foot_force_z_close_to_ground = \
            torch.clip(self.contact_forces[:, self.feet_indices][:, 1, 2:3]
                       - self.contact_forces_limit
                       * self.cfg.rewards.feet_force_z_close_to_ground_contact_force_limit_ratio,
                       min=0) \
            * error_right_foot_close_to_ground

        error_feet_force_z_close_to_ground = error_left_foot_force_z_close_to_ground + \
                                             error_right_foot_force_z_close_to_ground
        error_feet_force_z_close_to_ground = torch.sum(error_feet_force_z_close_to_ground, dim=1)  # dims 2->1

        reward_feet_force_z_close_to_ground = 1 - torch.exp(self.cfg.rewards.sigma_feet_force_z_close_to_ground
                                                            * error_feet_force_z_close_to_ground)

        return reward_feet_force_z_close_to_ground

    # ----------------------------------------------

    def _reward_feet_air_time(self):
        """
        Reward for keeping feet in the air for a certain amount of time
        """

        """
        Jason 2025-01-06:
        When the foot first contact with the ground, the air time will be reset to 0.
        So, we need to use the last air time to calculate the reward.
        """
        left_foot_air_time = self.feet_air_time_last[:, 0:1]  # dims 2
        right_foot_air_time = self.feet_air_time_last[:, 1:2]  # dims 2

        error_left_foot_air_time = torch.abs(left_foot_air_time - self.cfg.rewards.feet_air_time_target) \
                                   / self.cfg.rewards.feet_air_time_target  # dims 2
        error_right_foot_air_time = torch.abs(right_foot_air_time - self.cfg.rewards.feet_air_time_target) \
                                    / self.cfg.rewards.feet_air_time_target  # dims 2

        error_left_foot_air_time = torch.sum(error_left_foot_air_time, dim=1)  # dims 2->1
        error_right_foot_air_time = torch.sum(error_right_foot_air_time, dim=1)  # dims 2->1

        reward_left_foot_air_time = torch.exp(self.cfg.rewards.sigma_feet_air_time
                                              * error_left_foot_air_time)
        reward_right_foot_air_time = torch.exp(self.cfg.rewards.sigma_feet_air_time
                                               * error_right_foot_air_time)

        # ----------------------------

        condition_left_foot_first_contact_ground = self.feet_contact_trig[:, 0]  # dims 1
        condition_right_foot_first_contact_ground = self.feet_contact_trig[:, 1]  # dims 1

        reward_left_foot_air_time *= condition_left_foot_first_contact_ground
        reward_right_foot_air_time *= condition_right_foot_first_contact_ground

        """
        Jason 2024-03-23:
        Only apply the reward to the environment that is in the stand state
        """
        selector_move = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        selector_move[self.env_ids_of_off_command] = 0
        selector_move[self.env_ids_of_stand_command] = 0

        reward_left_foot_air_time *= selector_move
        reward_right_foot_air_time *= selector_move

        # ----------------------------

        reward_feet_air_time = reward_left_foot_air_time + reward_right_foot_air_time

        return reward_feet_air_time

    # ----------------------------------------------
