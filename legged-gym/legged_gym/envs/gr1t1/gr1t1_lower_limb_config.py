import numpy

from legged_gym.envs.gr1t1.gr1t1_config import GR1T1Cfg, GR1T1CfgPPO


class GR1T1LowerLimbCfg(GR1T1Cfg):
    class env(GR1T1Cfg.env):
        num_envs = 8192  # NVIDIA 4090 has 16384 CUDA cores

        num_obs = 39
        num_pri_obs = 168
        num_actions = 10

    class terrain(GR1T1Cfg.terrain):
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

    class commands(GR1T1Cfg.commands):
        class ranges_walk:
            lin_vel_x = [-0.50, 0.50]  # min max [m/s]
            lin_vel_y = [-0.50, 0.50]  # min max [m/s]
            ang_vel_yaw = [-0.50, 0.50]  # min max [rad/s]

    class init_state(GR1T1Cfg.init_state):
        pos = [0.0, 0.0, 0.95]  # x, y, z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            # left leg
            'l_hip_roll': 0.0,
            'l_hip_yaw': 0.,
            'l_hip_pitch': -0.2618,
            'l_knee_pitch': 0.5236,
            'l_ankle_pitch': -0.2618,

            # right leg
            'r_hip_roll': -0.,
            'r_hip_yaw': 0.,
            'r_hip_pitch': -0.2618,
            'r_knee_pitch': 0.5236,
            'r_ankle_pitch': -0.2618,
        }

    class control(GR1T1Cfg.control):
        # PD Drive parameters:
        stiffness = {
            'hip_roll': 57,
            'hip_yaw': 43,
            'hip_pitch': 114,
            'knee_pitch': 114,
            'ankle_pitch': 15.3,
        }  # [N*m/rad]
        damping = {
            'hip_roll': stiffness['hip_roll'] / 10,
            'hip_yaw': stiffness['hip_yaw'] / 10,
            'hip_pitch': stiffness['hip_pitch'] / 10,
            'knee_pitch': stiffness['knee_pitch'] / 10,
            'ankle_pitch': stiffness['ankle_pitch'] / 10,
        }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1.0

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 20

    class asset(GR1T1Cfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/gr1t1/urdf/GR1T1_lower_limb.urdf'
        name = 'GR1T1'

    class rewards(GR1T1Cfg.rewards):
        base_height_target = 0.88  # 期望的机器人身体高度
        swing_feet_height_target = 0.10  # 期望的脚抬高度

        # ---------------------------------------------------------------

        feet_air_time_target = 0.5  # 期望的脚空中时间
        feet_land_time_max = 1.0  # 最大的脚着地时间

        # ---------------------------------------------------------------

        class scales(GR1T1Cfg.rewards.scales):
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

    class normalization(GR1T1Cfg.normalization):
        actions_max = numpy.array([
            0.79, 0.7, 0.7, 1.92, 0.52,  # left leg
            0.09, 0.7, 0.7, 1.92, 0.52,  # right leg
        ])
        actions_min = numpy.array([
            -0.09, -0.7, -1.75, -0.09, -1.05,  # left leg
            -0.79, -0.7, -1.75, -0.09, -1.05,  # right leg
        ])

        clip_observations = 100.0
        clip_actions_max = actions_max + 60 / 180 * numpy.pi / 3
        clip_actions_min = actions_min - 60 / 180 * numpy.pi / 3

    class sim(GR1T1Cfg.sim):
        dt = 0.001


class GR1T1LowerLimbCfgPPO(GR1T1CfgPPO, GR1T1LowerLimbCfg):
    runner_class_name = 'OnPolicyRunner'

    class runner(GR1T1CfgPPO.runner):
        algorithm_class_name = 'PPO'
        policy_class_name = 'ActorCriticMLP'

        experiment_name = 'GR1T1'
        num_steps_per_env = 64

        run_name = 'gr1t1_lower_limb'
        max_iterations = 2000
        save_interval = 100

    class algorithm(GR1T1CfgPPO.algorithm):
        desired_kl = 0.03

        # actor-critic
        learning_rate = 1e-4

    class policy(GR1T1CfgPPO.policy):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        actor_output_activation = None  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        critic_output_activation = None  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        fixed_std = False
        init_noise_std = 0.2
