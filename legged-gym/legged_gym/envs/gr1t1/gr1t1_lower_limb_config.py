import numpy

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GR1T1LowerLimbCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096  # 4096, number of instances trained in parallel
        num_actions = 32  # number of actuators on robot

        size_mh = [11, 11]

        # IMPORTANT, num_observations should match the dimension of following items (subject to robot configurations) that included in observation instance
        # | `base_lin_vel`              | 3    |
        # | `base_ang_vel`              | 3    |
        # | `projected_gravity`         | 3    |
        # | `commands`                  | 3(4) |
        # | `dof_pos`                   | 32   |
        # | `dof_vel`                   | 32   |
        # | `action`                    | 32   |
        # | `num_measured_heights`      | 11*11|    meshgrid defined in `terrian` if `measure_heights` is `True`
        num_obs = 39
        num_pri_obs = 168
        actor_num_output = 10

        episode_length_s = 20  # episode length in seconds

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh

        curriculum = True
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        max_init_terrain_level = num_rows - 1  # maximum initial terrain level

        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.5, 0.4, 0.0, 0.1, 0.0]
        terrain_length = 10.
        terrain_width = 10.
        slope_treshold = 0.5  # slopes above this threshold will be corrected to vertical surfaces

        # 1mx1m rectangle (without center line)
        measure_heights = True
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    class init_state(LeggedRobotCfg.init_state):
        # pos = [0.0 0.0 0.95]
        pos = [0.0, 0.0, 0.95]  # x, y, z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            # left leg
            'l_hip_roll': 0.0,
            'l_hip_yaw': 0.,
            'l_hip_pitch': -0.5236,
            'l_knee_pitch': 1.0472,
            'l_ankle_pitch': -0.5236,
            'l_ankle_roll': 0.0,

            # right leg
            'r_hip_roll': -0.,
            'r_hip_yaw': 0.,
            'r_hip_pitch': -0.5236,
            'r_knee_pitch': 1.0472,
            'r_ankle_pitch': -0.5236,
            'r_ankle_roll': 0.0,
        }

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        heading_command = False  # if true: compute ang vel command from heading error

        resample_command_profiles = [
            "GR1T1-walk"
        ]

        ranges_swing_feet_height = [0.05, 0.05]  # min max [m]

        class ranges_walk:
            lin_vel_x = [-0.50, 0.50]  # min max [m/s]
            lin_vel_y = [-0.50, 0.50]  # min max [m/s]
            ang_vel_yaw = [-0.50, 0.50]  # min max [rad/s]

    class control(LeggedRobotCfg.control):
        # Notice:
        # Because we need to set the std of the actions,
        # We need to define the calculation of stiffness here based on the torque we want to output.
        # std = 10 deg = 0.1745 rad
        # PD Drive parameters:
        stiffness = {
            # Notice:
            # The joint with higher power output should be more stiff
            # The foot should has lower stiffness to simulate the compliance
            'hip_roll': 57,
            'hip_yaw': 43,
            'hip_pitch': 114,
            'knee_pitch': 114,
            'ankle_pitch': 15.3,
            'ankle_roll': 0.25,
        }  # [N*m/rad]
        damping = {
            'hip_roll': 5.7,
            'hip_yaw': 4.3,
            'hip_pitch': 11.4,
            'knee_pitch': 11.4,
            'ankle_pitch': 1.5,
            'ankle_roll': 0.01,
        }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1.0

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 20

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/gr1t1/urdf/GR1T1_lower_limb.urdf'
        name = 'gr1t1'

        # for both joint and link name
        torso_name = 'base'  # humanoid pelvis part
        chest_name = 'waist_roll'  # humanoid chest part
        forehead_name = 'head_pitch'  # humanoid head part

        waist_name = 'waist'
        waist_roll_name = 'waist_roll'
        waist_pitch_name = 'waist_pitch'
        head_name = 'head'
        head_roll_name = 'head_roll'
        head_pitch_name = 'head_pitch'

        # for link name
        thigh_name = 'thigh'
        shank_name = 'shank'
        foot_name = 'foot_roll'  # foot_pitch is not used
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
        shoulder_name = 'shoulder'
        shoulder_pitch_name = 'shoulder_pitch'
        shoulder_roll_name = 'shoulder_roll'
        shoulder_yaw_name = 'shoulder_yaw'
        elbow_name = 'elbow'
        wrist_name = 'wrist'
        wrist_yaw_name = 'wrist_yaw'
        wrist_roll_name = 'wrist_roll'
        wrist_pitch_name = 'wrist_pitch'

        terminate_after_contacts_on = ['waist', 'thigh', 'shoulder', 'elbow', 'hand']
        terminate_after_base_projected_gravity_greater_than = 1.0  # [m/s^2]
        terminate_after_base_lin_vel_greater_than = None  # [m/s]
        terminate_after_base_ang_vel_greater_than = None  # [rad/s]

        flip_visual_attachments = False
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.3, 1.0]
        restitution_range = [0.0, 1.0]

        # randomize mass
        randomize_base_mass = False
        randomize_thigh_mass = False
        randomize_shank_mass = False
        randomize_torso_mass = False
        randomize_upper_arm_mass = False
        randomize_lower_arm_mass = False
        added_mass_range = [-0.05, 0.05]

        # randomize inertias
        randomize_base_com = False
        added_com_range_x = [-0.05, 0.05]
        added_com_range_y = [-0.02, 0.02]
        added_com_range_z = [-0.05, 0.05]

        # randomize motor strength
        randomize_motor_strength = False
        motor_strength = [0.7, 1.4]

        # randomize external forces
        push_robots = True
        push_interval_s = 3.5
        max_push_vel_xy = 0.5

        apply_forces = False
        continue_time_s = 0.5
        max_ex_forces = [-200.0, 200.0]
        max_ex_torques = [-0.0, 0.0]

        # randomize observations
        randomize_obs_lin_vel = False
        obs_lin_vel = [0.8, 1.2]

        randomize_init_velocity = True

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.8  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9

        feet_height_target = 0.2
        swing_height_offset_target = 0.05
        swing_feet_distance_offset_target = [0.00, 0.25]
        max_contact_force = 500.  # forces above this value are penalized

        base_height_target = 0.85  # 期望的机器人身体高度

        feet_air_time_target = 0.5  # 期望的脚空中时间
        feet_land_time_max = 1.0  # 最大的脚着地时间

        class scales(LeggedRobotCfg.rewards.scales):
            # 对于 termination 和 collision 的惩罚
            termination = -0.0
            collision = -0.0
            stand_still = -1.0

            # tracking_lin_vel 和 tracking_ang_vel 是奖赏函数中的两个基本项，分别对应机器人的线速度和角速度
            # 尝试方式：设置一个较大的值，然后观察机器人的行为，如果开始运动了，则适当减小这个值，直到机器人停止运动，然后再适当增大这个值
            # cmd_diff_ang_vel_roll 和 cmd_diff_ang_vel_roll 身体姿态保持, 非水平地面上不能使用该值
            # orientation = -0.2  # -3.0 ~ -5.0 均可
            # 经验：
            # 1. -2.0 可能导致学出来的模型无法 sim2real，身体侧倾不足，导致机器人抬腿高度不足
            # 2. 因为使用的 IMU 的性能有限，如果身体姿态倾斜角度不够大，IMU 的检测数据也不一定准确
            cmd_diff_lin_vel_x = 2.0
            cmd_diff_lin_vel_y = 0.5
            cmd_diff_lin_vel_z = 0.1
            cmd_diff_ang_vel_roll = 0.1
            cmd_diff_ang_vel_pitch = 0.1
            cmd_diff_ang_vel_yaw = 1.0
            cmd_diff_base_height = 0.5

            cmd_diff_base_orient = 0.25
            cmd_diff_torso_orient = 0.0
            cmd_diff_chest_orient = 0.5
            cmd_diff_forehead_orient = 0.0

            # action_rate 惩罚的是机器人的动作
            # 如果设置太大，会导致机器人不敢动
            # 尝试方式：设置一个较大的值，然后观察机器人的行为，如果机器人不敢动，则逐渐调小（该方法有问题，不能这么做!）
            action_diff = -2.0
            action_diff_knee = -0.2

            action_diff_diff = -1.0

            # dof_vel, dof_acc, torques 惩罚的是机器人的动作
            # dof_acc 的惩罚是为了防止机器人动作过快，从而可以防止指令突变
            # 如果设置太大，会导致机器人不敢动
            # 尝试方式：设置一个较大的值，然后观察机器人的行为，如果机器人不敢动，则逐渐调小
            dof_vel_new = -0.2  # 速度惩罚会影响学习速度
            dof_acc_new = -0.2  # 加速度惩罚会影响学习速度
            dof_tor_new = -0.2  # 力矩惩罚会影响学习速度，和抬脚高度
            dof_tor_new_hip_roll = -1.0

            dof_tor_ankle_feet_lift_up = -0.5

            pose_offset = -1.0

            limits_dof_pos = -100.0
            limits_dof_vel = -100.0
            limits_dof_tor = -100.0

            feet_speed_xy_close_to_ground = -10.0
            feet_speed_z_close_to_height_target = 0.0

            feet_air_time = 4.0
            feet_air_height = 4.0  # 1.0
            feet_air_force = 1.0
            feet_land_time = -1.0

            # 惩罚四只脚都离地的情况
            on_the_air = -1.0

            # 惩罚踢到竖直面的情况

            feet_stumble = -0.2

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            lin_vel = 0.10  # m/s
            ang_vel = 0.05  # rad/s
            gravity = 0.03  # m/s^2
            dof_pos = 0.04  # rad
            dof_vel = 0.20  # rad/s
            action = 0.0  # rad
            height_measurements = 0.05  # m

    class normalization(LeggedRobotCfg.normalization):
        actions_max = numpy.array([
            0.79, 0.7, 0.7, 1.92, 0.52,  # left leg
            0.09, 0.7, 0.7, 1.92, 0.52,  # right leg
        ])
        actions_min = numpy.array([
            -0.09, -0.7, -1.75, -0.09, -1.05,  # left leg
            -0.79, -0.7, -1.75, -0.09, -1.05,  # right leg
        ])

        clip_observations = 100.0
        clip_actions_max = actions_max + 60 / 180 * numpy.pi / 3  # Note: allow output rated torque
        clip_actions_min = actions_min - 60 / 180 * numpy.pi / 3  # Note: allow output rated torque

    class sim(LeggedRobotCfg.sim):
        dt = 0.001


class GR1T1LowerLimbCfgPPO(LeggedRobotCfgPPO, GR1T1LowerLimbCfg):
    runner_class_name = 'OnPolicyRunner'

    class runner(LeggedRobotCfgPPO.runner):
        algorithm_class_name = 'PPO'
        policy_class_name = 'ActorCriticMLP'

        log_root = 'synology'
        experiment_name = 'GR1T1'
        num_steps_per_env = 64

        run_name = 'gr1t1_lower_limb'
        max_iterations = 2000  # number of policy updates -> 1500 * 1.0s = 1500s data collection
        save_interval = 100

        # load and resume
        resume = False
        load_run = -1
        checkpoint = -1  # 1000 # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt

    class algorithm(LeggedRobotCfgPPO.algorithm):
        desired_kl = 0.03

        # storage class
        storage_class = "RolloutStorage"

        # actor-critic
        learning_rate = 1e-4

    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        actor_output_activation = None  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        critic_output_activation = None  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        fixed_std = False
        init_noise_std = 0.2
