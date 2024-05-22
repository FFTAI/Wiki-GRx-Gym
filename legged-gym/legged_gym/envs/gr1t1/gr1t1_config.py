import numpy

from .legged_robot_fftai_config import LeggedRobotFFTAICfg, LeggedRobotFFTAICfgPPO


class GR1T1Cfg(LeggedRobotFFTAICfg):
    class env(LeggedRobotFFTAICfg.env):
        num_envs = 8192  # NVIDIA 4090 has 16384 CUDA cores
        episode_length_s = 20  # episode length in seconds

        size_mh = [0, 0]
        num_mh = size_mh[0] * size_mh[1]
        num_obs = 121
        num_stack = 1
        actor_num_output = 32

        encoder_profile = None
        num_encoder_input = 0
        num_encoder_output = 0

    class terrain(LeggedRobotFFTAICfg.terrain):
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh
        border_size = 25  # [m]
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]

        # 1mx1m rectangle (without center line)
        measure_heights = True
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        # 1mx1m rectangle (without center line)
        measure_heights_supervisor = True
        measured_points_x_supervisor = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        measured_points_y_supervisor = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    class commands(LeggedRobotFFTAICfg.commands):
        curriculum = False
        curriculum_profile = 'None'

        num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        heading_command = False  # if true: compute ang vel command from heading error

        resampling_time = 10.  # time before command are changed[s]
        resample_command_profiles = [
            "GR1T1-walk"
        ]
        resample_command_profiles_randomize = False
        resample_command_log = False
        resample_command_plot = False

        class ranges_walk:
            lin_vel_x = [-0.50, 0.50]  # min max [m/s]
            lin_vel_y = [-0.50, 0.50]  # min max [m/s]
            ang_vel_yaw = [-0.50, 0.50]  # min max [rad/s]

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
        decimation = 10

        # delay: Number of control action delayed @ sim DT
        delay_mean = 0.0
        delay_std = 0.0

    class asset(LeggedRobotFFTAICfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/GR1T1/urdf/GR1T1.urdf'
        name = "GR1T1"

        # for both joint and link name
        torso_name = 'base'  # humanoid pelvis part
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
        foot_name = 'foot_roll'  # 产生接触力的部分
        sole_name = 'sole'  # 限定脚掌范围的部分
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
        terminate_after_base_projected_gravity_greater_than = 0.7  # [m/s^2]
        terminate_after_base_lin_vel_greater_than = None  # [m/s]
        terminate_after_base_ang_vel_greater_than = None  # [rad/s]

        disable_gravity = False
        collapse_fixed_joints = False  # 显示 fixed joint 的信息
        fix_base_link = False

        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False

    class domain_rand(LeggedRobotFFTAICfg.domain_rand):
        # randomize friction and restitution
        randomize_friction = True
        friction_range = [0.5, 1.0]
        restitution_range = [0.0, 0.5]

        # randomize mass
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]  # unit : kg

        # randomize inertias
        randomize_base_com = True
        added_com_range_x = [-0.1, 0.1]  # unit : m
        added_com_range_y = [-0.1, 0.1]  # unit : m
        added_com_range_z = [-0.1, 0.1]  # unit : m

        # randomize motor strength
        randomize_motor_strength = True
        motor_strength = [0.90, 1.1]

        # randomize observations
        randomize_obs_lin_vel = False
        obs_lin_vel = [0.8, 1.2]

        # randomize external forces
        randomize_impulse_push_robots = True
        impulse_push_interval_s = 5.5
        impulse_push_max_vel_xy = 0.5

        # randomize init velocity
        randomize_init_dof_pos = True
        randomize_init_velocity = True

        # Jason 2024-03-01:
        # Should be False at the first training,
        # because falling down will make all robots to be stand command.
        # randomize stand command
        randomize_stand_command = False

    class rewards(LeggedRobotFFTAICfg.rewards):
        tracking_sigma = 1.0  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.95
        max_contact_force = 500.

        only_positive_rewards = False
        base_height_target = 0.90  # 期望的机器人身体高度
        swing_feet_height_target = 0.10  # 期望的脚抬起高度
        swing_contact_force_limit = 55.0 * 9.81 * 1.1  # 55 kg
        swing_height_offset_target = swing_feet_height_target / 4.0  # over half of the foot height can get reward
        swing_feet_distance_offset_target = [0.00, 0.22]

        feet_stumble_ratio = 5.0  # ratio = fxy / fz
        foot_contact_threshold = 1.0  # unit: N

        # ---------------------------------------------------------------

        feet_air_time_target = 0.5  # 期望的脚空中时间
        feet_land_time_max = 1.5  # 最大的脚着地时间

        # sigma ---------------------------------------------------------------

        # Jason 2024-03-15:
        # sigma is like the sensitivity of the reward function
        # calculate contact force sum, then reward = exp(-contact_force_sum * sigma)
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

        sigma_action_dof_pos_diff = -0.1
        sigma_action_dof_pos_diff_ankle = -0.5

        sigma_action_diff = -0.1

        sigma_dof_vel_new = -0.01
        sigma_dof_acc_new = -0.00005

        sigma_dof_tor_new = -0.00005
        sigma_dof_tor_new_hip_roll = -0.002

        sigma_dof_tor_ankle_feet_lift_up = -1.0

        sigma_pose_offset = -0.1

        sigma_pose_vel_waist = -0.1
        sigma_pose_vel_head = -0.1

        sigma_feet_speed_xy_close_to_ground = -100.0
        sigma_feet_speed_z_close_to_height_target = -20.0

        sigma_feet_air_time = -1.0
        sigma_feet_air_time_mid = -10.0
        sigma_feet_air_height = -20.0
        sigma_feet_air_force = -0.005
        sigma_feet_land_time = -1.0

        sigma_on_the_air = -1.0

        sigma_feet_force = -0.005
        sigma_feet_speed = -1.0  # -0.03
        sigma_feet_orient = -10.0

        sigma_orient_diff_feet_put_down = -100.0
        sigma_orient_diff_feet_lift_up = -100.0

        sigma_limits_actions = -10.0
        sigma_limits_dof_pos = -10.0
        sigma_limits_dof_vel = -10.0
        sigma_limits_dof_tor = -10.0

        sigma_swing_tracking = -1.0
        sigma_swing_contact_force = -0.002
        sigma_swing_height_target = -20.0
        sigma_swing_height_offset = -10.0
        sigma_swing_arm = -10.0
        sigma_swing_symmetric = -10.0
        sigma_swing_feet_distance = -10.0
        sigma_swing_orient = -10.0

        sigma_hip_yaw = -0.5

        sigma_step_on_terrain_edge = -1.0

        sigma_feet_stumble = -1.0

        sigma_follow_teacher = -1.0

        # sigma ---------------------------------------------------------------

        class scales(LeggedRobotFFTAICfg.rewards.scales):
            termination = 0.0

        class scales_walk:
            dof_vel = -0.001  # -0.01  # -0.001
            dof_acc = -1.e-7  # -1.e-6  # -1.e-7

            torques = -25.e-6  # -30.e-6
            ankle_torques = -5.0e-6  # -0.005  # -0.001# -0.005
            knee_torques = -25.e-6  # -0.0001

            feet_force = 0.8
            feet_speed = 0.6

            x_vel_diff = 1.0
            y_vel_diff = 0.5
            z_vel_diff = 0.2
            ang_vel_xy = 0.1
            ang_vel_diff = 0.4

            base_orientation_diff = 0.2
            feet_orien_diff = 0.2

            action_diff = 0.15
            ankle_action_diff = 0.6
            arm_action_diff = 1.0
            hip_yaw = 0.3
            dof_acc_diff = 0.15
            dof_pos_limits = 0.8

            swing_tracking = -0.2
            swing_contact_force = -1.0
            swing_height_target = -0.2
            swing_height_offset = 0.2
            swing_arm = 0.2
            swing_symmetric = 0.2
            feet_distance = 0.2

            arm_pose = -0.3
            torso_orientation_diff = -1.0
            torso_ang_vel_xy = -0.1

    class noise(LeggedRobotFFTAICfg.noise):
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales(LeggedRobotFFTAICfg.noise.noise_scales):
            lin_vel = 0.05  # m/s
            ang_vel = 0.04  # rad/s
            gravity = 0.02  # m/s^2
            dof_pos = 0.03  # rad
            dof_vel = 0.20  # rad/s
            action = 0.0  # rad
            height_measurements = 0.02  # m

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


class GR1T1CfgPPO(LeggedRobotFFTAICfgPPO, GR1T1Cfg):
    runner_class_name = 'OnPolicyRunner'

    class runner(LeggedRobotFFTAICfgPPO.runner):
        # policy_class_name = 'ActorCriticRecurrent'
        algorithm_class_name = 'PPO'
        policy_class_name = 'ActorCriticMLP'

        log_root = 'synology'
        experiment_name = 'GR1T1'
        run_name = ''
        num_steps_per_env = 50
        max_iterations = 500  # number of policy updates -> 500 * 1.0s = 500s data collection
        save_interval = 100  # check for potential saves every this many iterations

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

        set_std = False
        set_noise_std = 0.2
