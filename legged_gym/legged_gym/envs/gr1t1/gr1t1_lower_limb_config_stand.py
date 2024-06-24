import numpy

from legged_gym.envs.gr1t1.gr1t1_config import GR1T1Cfg, GR1T1CfgPPO


class GR1T1LowerLimbCfg(GR1T1Cfg):
    class env(GR1T1Cfg.env):
        num_envs = 8192  # NVIDIA 4090 has 16384 CUDA cores

        num_obs = 39
        num_pri_obs = 168
        num_actions = 10

        num_stack = 10
        use_stack = True

    class terrain(GR1T1Cfg.terrain):
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh

        curriculum = True
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        max_init_terrain_level = num_rows - 1  # maximum initial terrain level

        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.4, 0.4, 0.0, 0.2, 0.0]
        terrain_length = 10.
        terrain_width = 10.
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class commands(GR1T1Cfg.commands):
        class ranges(GR1T1Cfg.commands.ranges):
            lin_vel_x = [-0.0, 0.0]  # min max [m/s]
            lin_vel_y = [-0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-0.0, 0.0]  # min max [rad/s]

    class control(GR1T1Cfg.control):
        # PD Drive parameters:
        stiffness = {
            'hip_roll': 40,
            'hip_yaw': 45,
            'hip_pitch': 130,
            'knee_pitch': 130,
            'ankle_pitch': 18,
        }  # [N*m/rad]
        damping = {
            'hip_roll': stiffness['hip_roll'] / 10 * 2.5,
            'hip_yaw': stiffness['hip_yaw'] / 10 * 7.5,
            'hip_pitch': stiffness['hip_pitch'] / 10 * 2.5,
            'knee_pitch': stiffness['knee_pitch'] / 10 * 2.5,
            'ankle_pitch': stiffness['ankle_pitch'] / 10 * 2.5,
        }

    class asset(GR1T1Cfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/GR1T1/urdf/GR1T1_lower_limb.urdf'

    class rewards(GR1T1Cfg.rewards):
        base_height_target = 0.85  # 期望的机器人身体高度
        swing_feet_height_target = 0.05  # 期望的脚抬高度

        # ---------------------------------------------------------------

        feet_air_time_target = 0.5  # 期望的脚空中时间
        feet_land_time_max = 1.0  # 最大的脚着地时间

        # ---------------------------------------------------------------

        class scales(GR1T1Cfg.rewards.scales):
            termination = -0.0
            collision = -0.0
            stand_still = 2.0

            cmd_diff_lin_vel_x = 3.0
            cmd_diff_lin_vel_y = 1.0
            cmd_diff_lin_vel_z = 0.1
            cmd_diff_ang_vel_roll = 0.1
            cmd_diff_ang_vel_pitch = 0.1
            cmd_diff_ang_vel_yaw = 2.0
            cmd_diff_base_height = 0.5

            cmd_diff_base_orient = 0.25
            cmd_diff_torso_orient = 0.0
            cmd_diff_chest_orient = 0.5
            cmd_diff_forehead_orient = 0.0

            action_diff = -10.0
            action_diff_knee = -1.0
            action_diff_diff = -2.0

            dof_vel_new = -0.2
            dof_acc_new = -0.2
            dof_tor_new = -0.2

            dof_tor_ankle_feet_lift_up = -0.5

            pose_offset = 0.0
            pose_offset_hip_roll = 0.0

            limits_dof_pos = -100.0
            limits_dof_vel = -100.0
            limits_dof_tor = -10.0

            feet_speed_xy_close_to_ground = 0.2
            feet_speed_z_close_to_height_target = 0.0

            on_the_air = -1.0

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
        clip_actions_max = actions_max + 1 / 3
        clip_actions_min = actions_min - 1 / 3


class GR1T1LowerLimbCfgPPO(GR1T1CfgPPO, GR1T1LowerLimbCfg):
    class runner(GR1T1CfgPPO.runner):
        run_name = 'gr1t1_lower_limb_stand'
        max_iterations = 1000

    class algorithm(GR1T1CfgPPO.algorithm):
        learning_rate_min = 4.e-5
        desired_kl = 0.08

    class policy(GR1T1CfgPPO.policy):
        fixed_std = False
        init_noise_std = 0.2

        decay_std = True
        decay_ratio = 1 - 6.0e-6
        decay_std_min = 0.05
