import numpy

from legged_gym.envs.fftai.legged_robot_fftai_config import LeggedRobotFFTAICfg, LeggedRobotFFTAICfgPPO


class GR1T2Cfg(LeggedRobotFFTAICfg):
    class env(LeggedRobotFFTAICfg.env):
        num_envs = 8192  # NVIDIA 4090 has 16384 CUDA cores
        episode_length_s = 20  # episode length in seconds

        num_obs = 121
        num_actions = 32

    class terrain(LeggedRobotFFTAICfg.terrain):
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh
        border_size = 25  # [m]
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]

        # 1mx1m rectangle (without center line)
        measure_heights = True
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    class commands(LeggedRobotFFTAICfg.commands):
        curriculum = False
        curriculum_profile = 'None'

        num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges(LeggedRobotFFTAICfg.commands.ranges):
            lin_vel_x = [-0.75, 0.75]  # min max [m/s]
            lin_vel_y = [-0.50, 0.50]  # min max [m/s]
            ang_vel_yaw = [-1.00, 1.00]  # min max [rad/s]

    class init_state(LeggedRobotFFTAICfg.init_state):
        pos = [0.0, 0.0, 0.95]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            # left leg
            'l_hip_roll': 0.0,
            'l_hip_yaw': 0.,
            'l_hip_pitch': -0.2618,
            'l_knee_pitch': 0.5236,
            'l_ankle_pitch': -0.2618,
            'l_ankle_roll': 0.0,

            # right leg
            'r_hip_roll': -0.,
            'r_hip_yaw': 0.,
            'r_hip_pitch': -0.2618,
            'r_knee_pitch': 0.5236,
            'r_ankle_pitch': -0.2618,
            'r_ankle_roll': 0.0,

            # waist
            'waist_yaw': 0.0,
            'waist_pitch': 0.0,
            'waist_roll': 0.0,

            # head
            'head_yaw': 0.0,
            'head_pitch': 0.0,
            'head_roll': 0.0,

            # left arm
            'l_shoulder_pitch': 0.0,
            'l_shoulder_roll': 0.2,
            'l_shoulder_yaw': 0.0,
            'l_elbow_pitch': -0.3,
            'l_wrist_yaw': 0.0,
            'l_wrist_roll': 0.0,
            'l_wrist_pitch': 0.0,

            # right arm
            'r_shoulder_pitch': 0.0,
            'r_shoulder_roll': -0.2,
            'r_shoulder_yaw': 0.0,
            'r_elbow_pitch': -0.3,
            'r_wrist_yaw': 0.0,
            'r_wrist_roll': 0.0,
            'r_wrist_pitch': 0.0
        }

    class control(LeggedRobotFFTAICfg.control):
        # PD Drive parameters:
        stiffness = {
            'hip_roll': 251.625, 'hip_yaw': 362.5214, 'hip_pitch': 200,
            'knee_pitch': 200,
            'ankle_pitch': 10.9805, 'ankle_roll': 0.25,  # 'ankleRoll': 0.0,
            'waist_yaw': 362.5214, 'waist_pitch': 362.5214, 'waist_roll': 362.5214,
            'head_yaw': 10.0, 'head_pitch': 10.0, 'head_roll': 10.0,
            'shoulder_pitch': 92.85, 'shoulder_roll': 92.85, 'shoulder_yaw': 112.06,
            'elbow_pitch': 112.06,
            'wrist_yaw': 10.0, 'wrist_roll': 10.0, 'wrist_pitch': 10.0
        }  # [N*m/rad]
        damping = {
            'hip_roll': 14.72, 'hip_yaw': 10.0833, 'hip_pitch': 11,
            'knee_pitch': 11,
            'ankle_pitch': 0.5991, 'ankle_roll': 0.01,
            'waist_yaw': 10.0833, 'waist_pitch': 10.0833, 'waist_roll': 10.0833,
            'head_yaw': 1.0, 'head_pitch': 1.0, 'head_roll': 1.0,
            'shoulder_pitch': 2.575, 'shoulder_roll': 2.575, 'shoulder_yaw': 3.1,
            'elbow_pitch': 3.1,
            'wrist_yaw': 1.0, 'wrist_roll': 1.0, 'wrist_pitch': 1.0
        }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1.0

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 20

    class asset(LeggedRobotFFTAICfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/GR1T2/urdf/GR1T2.urdf'
        name = "GR1T2"

        # for both joint and link name
        torso_name = 'torso'  # humanoid pelvis part
        chest_name = 'waist_roll'  # humanoid chest part
        forehead_name = 'head_pitch'  # humanoid head part

        # imu
        imu_name = 'imu'

        # waist
        waist_name = 'waist'
        waist_yaw_name = 'waist_yaw'
        waist_roll_name = 'waist_roll'
        waist_pitch_name = 'waist_pitch'

        # head
        head_name = 'head'
        head_roll_name = 'head_roll'
        head_pitch_name = 'head_pitch'

        # for link name
        thigh_name = 'thigh'
        shank_name = 'shank'
        foot_name = 'foot_roll'
        sole_name = 'sole'
        upper_arm_name = 'upper_arm'
        lower_arm_name = 'lower_arm'
        hand_name = 'hand'

        # for joint name
        hip_name = 'hip'
        hip_roll_name = 'hip_roll'
        hip_yaw_name = 'hip_yaw'
        hip_pitch_name = 'hip_pitch'
        knee_name = 'knee'
        ankle_name = 'ankle'
        ankle_pitch_name = 'ankle_pitch'
        ankle_roll_name = 'ankle_roll'
        shoulder_name = 'shoulder'
        shoulder_pitch_name = 'shoulder_pitch'
        shoulder_roll_name = 'shoulder_roll'
        shoulder_yaw_name = 'shoulder_yaw'
        elbow_name = 'elbow'
        wrist_name = 'wrist'
        wrist_yaw_name = 'wrist_yaw'
        wrist_roll_name = 'wrist_roll'
        wrist_pitch_name = 'wrist_pitch'

        # for arm reaching
        arm_base_name = 'arm_base'
        arm_end_name = 'arm_end'

        penalize_contacts_on = []
        terminate_after_contacts_on = ['waist', 'thigh', 'shoulder', 'elbow', 'hand']

        disable_gravity = False
        collapse_fixed_joints = False  # 显示 fixed joint 的信息
        fix_base_link = False

        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False

    class rewards(LeggedRobotFFTAICfg.rewards):
        tracking_sigma = 1.0  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.95
        max_contact_force = 500.

        only_positive_rewards = False
        base_height_target = 0.90  # 期望的机器人身体高度
        swing_feet_height_target = 0.10  # 期望的脚抬起高度

        feet_stumble_ratio = 5.0  # ratio = fxy / fz

        # ---------------------------------------------------------------

        feet_air_time_target = 0.5  # 期望的脚空中时间
        feet_land_time_max = 1.5  # 最大的脚着地时间

        # ---------------------------------------------------------------

        sigma_collision = -1.0
        sigma_stand_still = -1.0

        sigma_cmd_diff_lin_vel_x = -3.0
        sigma_cmd_diff_lin_vel_y = -3.0
        sigma_cmd_diff_lin_vel_z = -3.0
        sigma_cmd_diff_ang_vel_roll = -3.0
        sigma_cmd_diff_ang_vel_pitch = -3.0
        sigma_cmd_diff_ang_vel_yaw = -3.0
        sigma_cmd_diff_base_height = -50.0

        sigma_cmd_diff_base_orient = -20.0
        sigma_cmd_diff_torso_orient = -20.0
        sigma_cmd_diff_chest_orient = -20.0
        sigma_cmd_diff_forehead_orient = -20.0

        sigma_action_diff = -0.1
        sigma_action_diff_knee = -1.0

        sigma_dof_vel_new = -0.01
        sigma_dof_vel_new_knee = -0.05

        sigma_dof_acc_new = -0.00005
        sigma_dof_tor_new = -0.00005
        sigma_dof_tor_new_hip_roll = -0.002

        sigma_dof_tor_ankle_feet_lift_up = -1.0

        sigma_pose_offset = -0.1
        sigma_pose_offset_hip_roll = -0.1
        sigma_pose_offset_hip_yaw = -0.1

        sigma_limits_dof_pos = -1.0
        sigma_limits_dof_vel = -1.0
        sigma_limits_dof_tor = -1.0

        sigma_feet_speed_xy_close_to_ground = -10.0
        sigma_feet_speed_z_close_to_height_target = -10.0

        sigma_feet_air_time = -1.0
        sigma_feet_air_time_mid = -10.0
        sigma_feet_air_height = -20.0
        sigma_feet_air_force = -0.005
        sigma_feet_land_time = -1.0

        sigma_on_the_air = -1.0

        sigma_feet_stumble = -1.0

        # sigma ---------------------------------------------------------------

        class scales(LeggedRobotFFTAICfg.rewards.scales):
            termination = 0.0

    class noise(LeggedRobotFFTAICfg.noise):
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales(LeggedRobotFFTAICfg.noise.noise_scales):
            lin_vel = 0.10  # m/s
            ang_vel = 0.05  # rad/s
            gravity = 0.03  # m/s^2
            dof_pos = 0.04  # rad
            dof_vel = 0.20  # rad/s
            action = 0.0  # rad
            height_measurements = 0.05  # m

    class normalization(LeggedRobotFFTAICfg.normalization):
        class obs_scales(LeggedRobotFFTAICfg.normalization.obs_scales):
            action = 1.0
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 1.0
            height_measurements = 1.0

        actions_max = numpy.array([
            0.79, 0.7, 0.7, 1.92, 0.52, 0.44,  # left leg
            0.09, 0.7, 0.7, 1.92, 0.52, 0.44,  # right leg
            1.05, 1.22, 0.7,  # waist
            2.71, 0.35, 0.35,  # head
            1.92, 3.27, 2.97, 2.27, 2.97, 0.61, 0.61,  # left arm
            1.92, 0.57, 2.97, 2.27, 2.97, 0.61, 0.61,  # right arm
        ])
        actions_min = numpy.array([
            -0.09, -0.7, -1.75, -0.09, -1.05, -0.44,  # left leg
            -0.79, -0.7, -1.75, -0.09, -1.05, -0.44,  # right leg
            -1.05, -0.52, -0.7,  # waist
            -2.71, -0.35, -0.52,  # head
            -2.79, -0.57, -2.97, -2.27, -2.97, -0.61, -0.61,  # left arm
            -2.79, -3.27, -2.97, -2.27, -2.97, -0.61, -0.61,  # right arm
        ])

        clip_observations = 100.0
        clip_actions_max = actions_max + (numpy.abs(actions_max) + numpy.abs(actions_min)) * 0.01
        clip_actions_min = actions_min - (numpy.abs(actions_max) + numpy.abs(actions_min)) * 0.01

    class sim(LeggedRobotFFTAICfg.sim):
        dt = 0.001


class GR1T2CfgPPO(LeggedRobotFFTAICfgPPO, GR1T2Cfg):
    runner_class_name = 'OnPolicyRunner'

    class runner(LeggedRobotFFTAICfgPPO.runner):
        algorithm_class_name = 'PPO'
        policy_class_name = 'ActorCriticMLP'

        experiment_name = 'GR1T2'
        num_steps_per_env = 64

        run_name = ''
        max_iterations = 2000
        save_interval = 100

    class algorithm(LeggedRobotFFTAICfgPPO.algorithm):
        # training params
        num_learning_epochs = 8
        num_mini_batches = 25  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-4
        learning_rate_min = 1.e-5
        learning_rate_max = 1.e-3
        schedule = 'adaptive'  # could be adaptive, fixed
        desired_kl = 0.01

        # storage class
        storage_class = "RolloutStorage"

    class policy(LeggedRobotFFTAICfgPPO.policy):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        actor_output_activation = None  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        critic_output_activation = None  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        fixed_std = False
        init_noise_std = 0.2
