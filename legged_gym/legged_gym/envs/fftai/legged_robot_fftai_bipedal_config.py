import torch

from legged_gym.envs.fftai.legged_robot_fftai_config import (
    LeggedRobotFFTAICfg,
    LeggedRobotFFTAICfgPPO,
)


class LeggedRobotFFTAIBipedalCfg(LeggedRobotFFTAICfg):
    class asset(LeggedRobotFFTAICfg.asset):
        name = "FFTAIBipedal"

        # for both joint and link name
        base_name = 'base'
        torso_name = 'torso'  # humanoid pelvis part
        chest_name = 'waist'  # humanoid chest part
        forehead_name = 'head'  # humanoid head part

        # imu
        imu_name = 'imu'

        # waist (for link and joint, with _link and _joint)
        waist_name = 'waist'
        waist_yaw_name = "waist_yaw"
        waist_roll_name = "waist_roll"
        waist_pitch_name = "waist_pitch"

        # head (for link and joint, with _link and _joint)
        head_name = 'head'
        head_roll_name = "head_roll"
        head_pitch_name = "head_pitch"
        head_yaw_name = "head_yaw"

        # leg (for link, with _link)
        thigh_name = 'thigh'
        shank_name = 'shank'
        foot_name = 'foot'
        sole_name = 'sole'

        # arm (for link, with _link)
        upper_arm_name = 'upper_arm'
        lower_arm_name = 'lower_arm'
        hand_name = 'hand'
        end_effector_name = 'end_effector'

        # payload
        payload_name = 'payload'

        # leg (for joint, with _joint)
        hip_name = 'hip'
        hip_roll_name = "hip_roll"
        hip_pitch_name = "hip_pitch"
        hip_yaw_name = "hip_yaw"
        knee_name = 'knee'
        knee_pitch_name = "knee_pitch"
        ankle_name = 'ankle'
        ankle_roll_name = "ankle_roll"
        ankle_pitch_name = "ankle_pitch"

        # arm (for joint, with _joint)
        shoulder_name = 'shoulder'
        shoulder_roll_name = "shoulder_roll"
        shoulder_pitch_name = "shoulder_pitch"
        shoulder_yaw_name = "shoulder_yaw"
        elbow_name = 'elbow'
        wrist_name = 'wrist'
        wrist_roll_name = "wrist_roll"
        wrist_pitch_name = "wrist_pitch"
        wrist_yaw_name = "wrist_yaw"

        """
        Jason 2024-11-04:
        不同的大类关节名称，用于不同的约束条件。
        upper_limb_joint_names: 上肢关节名称
        lower_limb_joint_names: 下肢关节名称
        main_body_joint_names: 主体关节名称 (腰部 + 下肢)
        """
        upper_limb_joint_names = [shoulder_name, elbow_name, wrist_name]
        lower_limb_joint_names = [hip_name, knee_name, ankle_name]
        main_body_joint_names = [waist_name, hip_name, knee_name, ankle_name]

        # for arm reaching
        arm_base_name = 'arm_base'
        arm_end_name = 'arm_end'

    class rewards(LeggedRobotFFTAICfg.rewards):
        # ---------------------------------------------------------------
        # Reward related constants
        # 脚接触地面的力的限制比例（相对于重力）
        contact_force_limit_ratio = 1.1

        stand_still_foot_distance = 0.20  # unit: m

        feet_distance_too_close = max(stand_still_foot_distance - 0.10, 0.10)  # m
        feet_distance_y_too_close = max(stand_still_foot_distance / 4.0 * 3.0, 0.10)  # unit: m

        feet_force_z_close_to_ground_contact_force_limit_ratio = 1.0

        feet_air_time_target = 0.4  # unit: s

        # ---------------------------------------------------------------
        # Reward coefficients
        sigma_stand_still_foot_distance = -10.0 * torch.e

        sigma_feet_distance_too_close = -10.0 * torch.e
        sigma_feet_distance_y_too_close = -10.0 * torch.e

        sigma_feet_speed_xy_close_to_ground = -10.0
        sigma_feet_force_z_close_to_ground = -0.01 * torch.e

        sigma_feet_air_time = -1.0 * torch.e

        class scales(LeggedRobotFFTAICfg.rewards.scales):
            pass


class LeggedRobotFFTAIBipedalCfgPPO(LeggedRobotFFTAICfgPPO, LeggedRobotFFTAIBipedalCfg):
    pass
