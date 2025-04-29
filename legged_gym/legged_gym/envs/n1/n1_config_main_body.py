import numpy

from legged_gym.envs.n1.n1_config import (
    N1Cfg as N1BaseCfg,
    N1CfgPPO as N1BaseCfgPPO,
)


class N1MainBodyCfg(N1BaseCfg):
    class env(N1BaseCfg.env):
        # episode length in seconds
        episode_length_s = 20

        num_obs = 48
        num_pri_obs = 181
        num_actions = (6 + 6 + 1)

        use_stack = True
        num_stack = 5

    class asset(N1BaseCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/N1/urdf/N1_main_body_raw.urdf"

    class terrain(N1BaseCfg.terrain):
        mesh_type = "trimesh"  # "plane" or "trimesh"

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
            0.5, 0.5,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
        ]

    class rewards(N1BaseCfg.rewards):
        class scales(N1BaseCfg.rewards.scales):
            stand_still_dof_pos_waist_joint = 0.50
            stand_still_foot_distance = 0.25

            """
            base related
            """
            cmd_diff_base_lin_vel_x = 1.00
            cmd_diff_base_lin_vel_y = 0.50
            cmd_diff_base_ang_vel_yaw = 0.75

            base_lin_vel_z = 0.25

            base_height_offset_range = 0.50

            base_flat_orient = 0.25
            torso_flat_orient = 0.25

            """
            dof related
            """
            action_diff = -5.00
            action_diff_diff = -1.10

            dof_pos_offset = 0.50
            # dof_vel = -0.20
            dof_acc = -0.25
            dof_tor = -0.05

            limits_dof_pos_without_ankle = -10.00
            limits_dof_vel_without_ankle = -5.00
            limits_dof_tor = -1.00

            """
            feet related
            """
            feet_speed_xy_close_to_ground = 0.20
            feet_force_z_close_to_ground = -0.20
            feet_stumble = -0.20
            feet_distance_too_close = -0.50

            feet_air_time = 2.00  # two feet -> 1.0

    class normalization(N1BaseCfg.normalization):
        actions_max = numpy.array([
            2.618, 1.571, 1.571, 2.356, 0.436, 0.785,  # left leg
            2.618, 0.262, 1.571, 2.356, 0.436, 0.785,  # right leg
            2.618,  # waist
        ])
        actions_min = numpy.array([
            -2.618, -0.262, -1.571, -0.087, -0.436, -0.785,  # left leg
            -2.618, -1.571, -1.571, -0.087, -0.436, -0.785,  # right leg
            -2.618,  # waist
        ])

        clip_observations = 100.0

        clip_actions_max = \
            actions_max \
            + numpy.array([
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # left leg
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # right leg
                1.0,  # waist
            ])
        clip_actions_min = \
            actions_min \
            - numpy.array([
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # left leg
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # right leg
                1.0,  # waist
            ])

    class mirror(N1BaseCfg.mirror):
        """"""
        """
        做镜像处理时，主要是需要将:
        - base y, roll, yaw 的值进行反向处理，即取负值。
        - dof roll, yaw 的值进行反向处理，即取负值。
        - dof_pos_offset, dof_vel, actions 的值进行左右对调处理。
        """
        enable_mirror = True
        observations_coefficient = \
            numpy.array(
                [
                    # commands
                    1.0, -1.0, -1.0,  # x, y, yaw
                    # base related
                    -1.0, 1.0, -1.0,  # base_ang_vel, roll, pitch, yaw
                    1.0, -1.0, 1.0,  # base_projected_gravity, x, y, z
                    # dof related
                    1.0, -1.0, -1.0, 1.0, -1.0, 1.0,  # dof_pos_offset (left leg)
                    1.0, -1.0, -1.0, 1.0, -1.0, 1.0,  # dof_pos_offset (right leg)
                    -1.0,  # dof_pos_offset (waist)
                    1.0, -1.0, -1.0, 1.0, -1.0, 1.0,  # dof_vel (left leg)
                    1.0, -1.0, -1.0, 1.0, -1.0, 1.0,  # dof_vel (right leg)
                    -1.0,  # dof_vel (waist)
                    # action related
                    1.0, -1.0, -1.0, 1.0, -1.0, 1.0,  # actions (left leg)
                    1.0, -1.0, -1.0, 1.0, -1.0, 1.0,  # actions (right leg)
                    -1.0,  # actions (waist)
                ]
            )
        observations_exchange = \
            numpy.array(
                [
                    # commands
                    # base related
                    *[(9 + i, 9 + 6 + i) for i in range(6)],  # dof related (dof_pos_offset)
                    *[(22 + i, 22 + 6 + i) for i in range(6)],  # dof related (dof_vel)
                    *[(35 + i, 35 + 6 + i) for i in range(6)],  # action related
                ]
            )
        actions_coefficient = \
            numpy.array(
                [
                    1.0, -1.0, -1.0, 1.0, -1.0, 1.0,  # actions (left leg)
                    1.0, -1.0, -1.0, 1.0, -1.0, 1.0,  # actions (right leg)
                    -1.0,  # actions (waist)
                ]
            )
        actions_exchange = \
            numpy.array([
                *[(0 + i, 0 + 6 + i) for i in range(6)],  # left leg <-> right leg
            ])


class N1MainBodyCfgPPO(N1BaseCfgPPO, N1MainBodyCfg):
    runner_class_name = "OnPolicyRunnerMirror"

    class algorithm(N1BaseCfgPPO.algorithm):
        class_name = "PPOMirror"

    class policy(N1BaseCfgPPO.policy):
        init_noise_std = [0.2] * N1MainBodyCfg.env.num_actions
