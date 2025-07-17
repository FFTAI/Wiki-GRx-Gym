import math
import numpy
import torch

from legged_gym.envs.fftai.legged_robot_fftai_bipedal_config import (
    LeggedRobotFFTAIBipedalCfg,
    LeggedRobotFFTAIBipedalCfgPPO,
)


class N1Cfg(LeggedRobotFFTAIBipedalCfg):
    class sim(LeggedRobotFFTAIBipedalCfg.sim):
        dt = 0.002  # simulation time step [s]

    class env(LeggedRobotFFTAIBipedalCfg.env):
        # NVIDIA 4090 has 16384 CUDA cores
        num_envs = 4096

        # episode length in seconds
        episode_length_s = 20

        num_obs = 121
        num_pri_obs = num_obs
        num_actions = (6 + 6 + 1 + 5 + 5)  # 6 left leg, 6 right leg, 1 waist, 2 left arm, 5 right arm, 5 right arm

        obs_profile = "default"

    class terrain(LeggedRobotFFTAIBipedalCfg.terrain):
        mesh_type = "plane"  # "heightfield" # none, plane, heightfield or trimesh

        # terrain types:
        # [
        # smooth plane, rough plane,
        # smooth slope, rough slope,
        # smooth wave, rough wave,
        # smooth stairs, rough stairs,
        # stones, discrete,
        # gap, pit,
        # ]
        terrain_proportions = [
            1.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
        ]

        terrain_length = (1.0 / 2.0) * 20 * 0.5
        terrain_width = (1.0 / 2.0) * 20 * 0.5

    class asset(LeggedRobotFFTAIBipedalCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/N1/urdf/N1_raw.urdf"
        name = "N1"

        # for link name
        foot_name = "foot_pitch"

        penalize_contacts_on = []
        terminate_contacts_on = [
            LeggedRobotFFTAIBipedalCfg.asset.imu_name,
            LeggedRobotFFTAIBipedalCfg.asset.torso_name,
            LeggedRobotFFTAIBipedalCfg.asset.waist_name,
            LeggedRobotFFTAIBipedalCfg.asset.thigh_name,
            LeggedRobotFFTAIBipedalCfg.asset.upper_arm_name,
            LeggedRobotFFTAIBipedalCfg.asset.lower_arm_name,
            LeggedRobotFFTAIBipedalCfg.asset.hand_name,
            LeggedRobotFFTAIBipedalCfg.asset.end_effector_name,
            LeggedRobotFFTAIBipedalCfg.asset.payload_name,
        ]

        disable_gravity = False
        collapse_fixed_joints = False  # 显示 fixed joint 的信息

        fix_base_link = False

        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False

        foot_thickness = 0.035 + 0.0125

    class dof(LeggedRobotFFTAIBipedalCfg.dof):
        """"""
        """
        friction: 关节的摩擦系数，无量纲
        armature: 关节的转子惯性，单位是 kg*m^2
        mechanism: 传动机构的摩擦系数，无量纲
        """
        friction = {
            # left leg
            "left_hip_pitch_joint": 0.02,
            "left_hip_roll_joint": 0.02,
            "left_hip_yaw_joint": 0.02,
            "left_knee_pitch_joint": 0.02,
            "left_ankle_roll_joint": 0.02,
            "left_ankle_pitch_joint": 0.02,

            # right leg
            "right_hip_pitch_joint": 0.02,
            "right_hip_roll_joint": 0.02,
            "right_hip_yaw_joint": 0.02,
            "right_knee_pitch_joint": 0.02,
            "right_ankle_roll_joint": 0.02,
            "right_ankle_pitch_joint": 0.02,

            # waist
            "waist_yaw_joint": 0.02,

            # left arm
            "left_shoulder_pitch_joint": 0.02,
            "left_shoulder_roll_joint": 0.02,
            "left_shoulder_yaw_joint": 0.02,
            "left_elbow_pitch_joint": 0.02,
            "left_wrist_yaw_joint": 0.02,

            # right arm
            "right_shoulder_pitch_joint": 0.02,
            "right_shoulder_roll_joint": 0.02,
            "right_shoulder_yaw_joint": 0.02,
            "right_elbow_pitch_joint": 0.02,
            "right_wrist_yaw_joint": 0.02,
        }
        armature = {
            # left leg
            "left_hip_pitch_joint": 0.12109824,
            "left_hip_roll_joint": 0.167592,
            "left_hip_yaw_joint": 0.167592,
            "left_knee_pitch_joint": 0.12109824,
            "left_ankle_roll_joint": 0.0312822,
            "left_ankle_pitch_joint": 0.0312822,

            # right leg
            "right_hip_pitch_joint": 0.12109824,
            "right_hip_roll_joint": 0.167592,
            "right_hip_yaw_joint": 0.167592,
            "right_knee_pitch_joint": 0.12109824,
            "right_ankle_roll_joint": 0.0312822,
            "right_ankle_pitch_joint": 0.0312822,

            # waist
            "waist_yaw_joint": 0.167592,

            # left arm
            "left_shoulder_pitch_joint": 0.167592,
            "left_shoulder_roll_joint": 0.0312822,
            "left_shoulder_yaw_joint": 0.0312822,
            "left_elbow_pitch_joint": 0.0312822,
            "left_wrist_yaw_joint": 0.0312822,

            # right arm
            "right_shoulder_pitch_joint": 0.167592,
            "right_shoulder_roll_joint": 0.0312822,
            "right_shoulder_yaw_joint": 0.0312822,
            "right_elbow_pitch_joint": 0.0312822,
            "right_wrist_yaw_joint": 0.0312822,
        }
        mechanism = {
            # left leg
            "left_hip_pitch_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_pitch_joint": 0.01,  # 连杆传动机构
            "left_ankle_roll_joint": 0.01,  # 连杆传动机构
            "left_ankle_pitch_joint": 0.01,  # 连杆传动机构

            # right leg
            "right_hip_pitch_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_pitch_joint": 0.01,  # 连杆传动机构
            "right_ankle_roll_joint": 0.01,  # 连杆传动机构
            "right_ankle_pitch_joint": 0.01,  # 连杆传动机构

            # waist
            "waist_yaw_joint": 0.0,

            # left arm
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,

            # right arm
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        }

    class init_state(LeggedRobotFFTAIBipedalCfg.init_state):
        pos = [0.0, 0.0, 0.70]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            # left leg
            "left_hip_pitch_joint": -numpy.deg2rad(14.0),
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_pitch_joint": +numpy.deg2rad(29.5),
            "left_ankle_roll_joint": 0.0,
            "left_ankle_pitch_joint": -numpy.deg2rad(13.7),

            # right leg
            "right_hip_pitch_joint": -numpy.deg2rad(14.0),
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_pitch_joint": +numpy.deg2rad(29.5),
            "right_ankle_roll_joint": 0.0,
            "right_ankle_pitch_joint": -numpy.deg2rad(13.7),

            # waist
            "waist_yaw_joint": 0.0,

            # left arm
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,

            # right arm
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        }

    class commands(LeggedRobotFFTAIBipedalCfg.commands):
        curriculum = False
        curriculum_chg_lin_vel_x = 0.50  # additional linear velocity in x direction
        curriculum_chg_lin_vel_y = 0.50  # additional linear velocity in y direction
        curriculum_chg_ang_vel_yaw = 0.50  # additional angular velocity around z axis
        curriculum_min_lin_vel_x = -0.50  # maximum linear velocity in x direction
        curriculum_min_lin_vel_y = -0.50  # maximum linear velocity in y direction
        curriculum_min_ang_vel_yaw = -1.00  # maximum angular velocity around z axis
        curriculum_max_lin_vel_x = +0.50  # maximum linear velocity in x direction
        curriculum_max_lin_vel_y = +0.50  # maximum linear velocity in y direction
        curriculum_max_ang_vel_yaw = +1.00  # maximum angular velocity around z axis

        command_profile = "base_velocity"
        num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw (in heading mode ang_vel_yaw is recomputed from heading error)
        resample_command_interval_s = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        gait_patterns = [
            "stand", "walk",
        ]

        class ranges(LeggedRobotFFTAIBipedalCfg.commands.ranges):
            lin_vel_x = [-0.50, 0.75]  # min max [m/s]
            lin_vel_y = [-0.50, 0.50]  # min max [m/s]
            ang_vel_yaw = [-1.00, 1.00]  # min max [rad/s]

    class control(LeggedRobotFFTAIBipedalCfg.control):
        action_names = [
            # leg
            "hip_pitch",
            "hip_roll",
            "hip_yaw",
            "knee_pitch",
            "ankle_roll",
            "ankle_pitch",

            # waist
            "waist_yaw",

            # arm
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow_pitch",
            "wrist_yaw",
        ]

        action_scale = {
            # leg
            "hip_pitch": 1,
            "hip_roll": 1,
            "hip_yaw": 1,
            "knee_pitch": 1,
            "ankle_roll": 1,
            "ankle_pitch": 1,

            # waist
            "waist_yaw": 1,

            # arm
            "shoulder_pitch": 1,
            "shoulder_roll": 1,
            "shoulder_yaw": 1,
            "elbow_pitch": 1,
            "wrist_yaw": 1,
        }

        control_type = {
            # leg
            "hip_pitch": 1,
            "hip_roll": 1,
            "hip_yaw": 1,
            "knee_pitch": 1,
            "ankle_roll": 1,
            "ankle_pitch": 1,

            # waist
            "waist_yaw": 1,

            # arm
            "shoulder_pitch": 1,
            "shoulder_roll": 1,
            "shoulder_yaw": 1,
            "elbow_pitch": 1,
            "wrist_yaw": 1,
        }

        # PD Drive parameters:
        stiffness = {
            # leg
            "hip_pitch": 180.0,
            "hip_roll": 120.0,
            "hip_yaw": 90.0,
            "knee_pitch": 120.0,
            "ankle_roll": 45.0,
            "ankle_pitch": 45.0,

            # waist
            "waist_yaw": 90.0,

            # arm
            "shoulder_pitch": 90.0,
            "shoulder_roll": 45.0,
            "shoulder_yaw": 45.0,
            "elbow_pitch": 45.0,
            "wrist_yaw": 45.0,
        }  # [N*m/rad]
        damping = {
            # leg
            "hip_pitch": 10.0,
            "hip_roll": 10.0,
            "hip_yaw": 8.0,
            "knee_pitch": 8.0,
            "ankle_roll": 2.5,
            "ankle_pitch": 2.5,

            # waist
            "waist_yaw": 8.0,

            # arm
            "shoulder_pitch": 8.0,
            "shoulder_roll": 2.5,
            "shoulder_yaw": 2.5,
            "elbow_pitch": 2.5,
            "wrist_yaw": 2.5,
        }

        dof_pos_offset_scale = {
            # leg
            "hip_pitch": 1,
            "hip_roll": 1,
            "hip_yaw": 1,
            "knee_pitch": 1,
            "ankle_roll": 1,
            "ankle_pitch": 1,

            # waist
            "waist_yaw": 1,

            # arm
            "shoulder_pitch": 1,
            "shoulder_roll": 1,
            "shoulder_yaw": 1,
            "elbow_pitch": 1,
            "wrist_yaw": 1,
        }

        dof_vel_scale = {
            # leg
            "hip_pitch": 1,
            "hip_roll": 1,
            "hip_yaw": 1,
            "knee_pitch": 1,
            "ankle_roll": 1,
            "ankle_pitch": 1,

            # waist
            "waist_yaw": 1,

            # arm
            "shoulder_pitch": 1,
            "shoulder_roll": 1,
            "shoulder_yaw": 1,
            "elbow_pitch": 1,
            "wrist_yaw": 1,
        }

        dof_tor_scale = {
            # leg
            "hip_pitch": 1,
            "hip_roll": 1,
            "hip_yaw": 1,
            "knee_pitch": 1,
            "ankle_roll": 1,
            "ankle_pitch": 1,

            # waist
            "waist_yaw": 1,

            # arm
            "shoulder_pitch": 1,
            "shoulder_roll": 1,
            "shoulder_yaw": 1,
            "elbow_pitch": 1,
            "wrist_yaw": 1,
        }

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class rewards(LeggedRobotFFTAIBipedalCfg.rewards):
        only_positive_rewards = False

        gait_cycle_period = 0.8  # gait cycle period [s]

        """
        机器人相关参数信息：
        robot_mass: 机器人的质量
        """
        robot_mass = 39.727  # kg

        """
        奖赏设计的一些期望值：
        base_height_target: 期望的机器人身体高度，可以通过设置 fix_base_link = True 来确认
        stand_still_foot_distance: 站立阶段脚的横向距离，可以通过设置 fix_base_link = True 来确认
        """
        base_height_target = 0.68  # unit: m
        stand_still_foot_distance = 0.24  # unit: m

        """
        运动约束相关：
        soft_dof_pos_limit: 关节位置限制
        soft_dof_vel_limit: 关节速度限制
        soft_dof_tor_limit: 关节力矩限制
        """
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.95
        soft_dof_tor_limit = 0.95

        feet_distance_too_close = max(stand_still_foot_distance - 0.10, 0.10)  # unit: m
        feet_distance_y_too_close = max(stand_still_foot_distance / 2.0, 0.10)  # unit: m

        base_height_offset_range_limit = 0.01  # unit: m

        # 脚接触地面的力的限制比例（相对于重力）
        contact_force_limit_ratio = 1.0

        # ---------------------------------------------------------------

        class scales(LeggedRobotFFTAIBipedalCfg.rewards.scales):
            termination = 0.0

    class noise(LeggedRobotFFTAIBipedalCfg.noise):
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales(LeggedRobotFFTAIBipedalCfg.noise.noise_scales):
            """ """
            """
            全部为基本国际单位制
            action: 动作 rad
            dof_pos: 关节位置 rad
            dof_vel: 关节速度 rad/s
            lin_vel: 线速度 m/s
            ang_vel: 角速度 rad/s
            gravity: 重力 m/s^2
            height_measurements: 高度 m
            """
            action = 0.00  # rad
            lin_vel = 0.10  # m/s
            ang_vel = 0.05  # rad/s, 0.05 rad/s -> 2.87 deg/s
            gravity = 0.03  # m/s^2
            dof_pos = 0.04  # rad, 0.04 rad -> 2.3 deg
            dof_vel = 0.20  # rad/s, 0.20 rad/s -> 11.5 deg/s
            height_measurements = 0.05  # m

    class normalization(LeggedRobotFFTAIBipedalCfg.normalization):
        class obs_scales(LeggedRobotFFTAIBipedalCfg.normalization.obs_scales):
            action = 1.00
            lin_vel = 1.00  # map 1.0 m/s -> 1.0
            ang_vel = 1.00  # map 1.0 rad/s -> 1.0
            gravity = 1.00
            dof_pos = 1.00
            dof_vel = 0.10  # map 6.28 rad/s -> 0.628
            height_measurements = 5.0  # map 0.2 m -> 1.0

        actions_max = numpy.array([
            2.618, 1.571, 1.571, 2.356, 0.436, 0.785,  # left leg
            2.618, 0.262, 1.571, 2.356, 0.436, 0.785,  # right leg
            2.618,  # waist
            2.967, 2.792, 1.833, 1.658, 1.833,  # left arm
            2.967, 0.175, 1.833, 1.658, 1.833,  # right arm
        ])
        actions_min = numpy.array([
            -2.618, -0.262, -1.571, -0.087, -0.436, -0.785,  # left leg
            -2.618, -1.571, -1.571, -0.087, -0.436, -0.785,  # right leg
            -2.618,  # waist
            -2.967, -0.175, -1.833, -0.314, 1.833,  # left arm
            -2.967, -2.792, -1.833, -0.314, -1.833,  # right arm
        ])

        clip_observations = 100.0

        clip_actions_max = \
            actions_max \
            + numpy.array([
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # left leg
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # right leg
                1.0,  # waist
                1.0, 1.0, 1.0, 1.0, 1.0,  # left arm
                1.0, 1.0, 1.0, 1.0, 1.0,  # right arm
            ])
        clip_actions_min = \
            actions_min \
            - numpy.array([
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # left leg
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # right leg
                1.0,  # waist
                1.0, 1.0, 1.0, 1.0, 1.0,  # left arm
                1.0, 1.0, 1.0, 1.0, 1.0,  # right arm
            ])

    class mirror:
        enable_mirror = False
        observations_coefficient = \
            numpy.ones(121)
        observations_exchange = \
            numpy.array([])
        actions_coefficient = \
            numpy.ones(23)
        actions_exchange = \
            numpy.array([])


class N1CfgPPO(LeggedRobotFFTAIBipedalCfgPPO, N1Cfg):
    runner_class_name = "OnPolicyRunner"

    class runner(LeggedRobotFFTAIBipedalCfgPPO.runner):
        experiment_name = "N1"
        num_steps_per_env = 64

        run_name = ""
        max_iterations = 5000
        save_interval = 100

    class algorithm(LeggedRobotFFTAIBipedalCfgPPO.algorithm):
        class_name = "PPO"

        # training params
        num_learning_epochs = 8
        num_mini_batches = 25  # mini batch size = num_envs*nsteps / num_mini_batches
        learning_rate = 1.e-4
        learning_rate_min = 1.e-5
        learning_rate_max = 1.e-3
        schedule = "adaptive"  # could be adaptive, fixed
        desired_kl = 0.05

        # storage class
        storage_class = "RolloutStorage"

    class policy(LeggedRobotFFTAIBipedalCfgPPO.policy):
        class_name = "ActorCriticMLP"

        # policy params
        actor_hidden_dims = [1024, 512, 256, 128]
        critic_hidden_dims = [1024, 512, 256, 128]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        init_weights = False

        fixed_std = False
        init_noise_std = [0.2] * N1Cfg.env.num_actions

        decay_std = False
        decay_ratio = 1 - 2.0e-6
        decay_std_min = 0.1
