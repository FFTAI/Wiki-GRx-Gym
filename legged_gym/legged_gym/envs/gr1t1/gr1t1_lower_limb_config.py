import numpy

from legged_gym.envs.gr1t1.gr1t1_config import (
    GR1T1Cfg,
    GR1T1CfgPPO,
)


class GR1T1LowerLimbCfg(GR1T1Cfg):
    class env(GR1T1Cfg.env):
        num_envs = 8192  # NVIDIA 4090 has 16384 CUDA cores

        num_obs = 39
        num_pri_obs = 168
        num_actions = 10

    class terrain(GR1T1Cfg.terrain):
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh

    class control(GR1T1Cfg.control):
        # PD Drive parameters:
        stiffness = {
            'hip_roll': 48 / numpy.deg2rad(30),
            'hip_yaw': 66 / numpy.deg2rad(30),
            'hip_pitch': 130 / numpy.deg2rad(30),
            'knee_pitch': 130 / numpy.deg2rad(30),
            'ankle_pitch': 15 / numpy.deg2rad(30),
        }  # [N*m/rad]
        damping = {
            'hip_roll': stiffness['hip_roll'] / 10 * 0.5,
            'hip_yaw': stiffness['hip_yaw'] / 10 * 0.5,
            'hip_pitch': stiffness['hip_pitch'] / 10 * 0.5,
            'knee_pitch': stiffness['knee_pitch'] / 10 * 0.5,
            'ankle_pitch': stiffness['ankle_pitch'] / 10 * 0.5,
        }

    class asset(GR1T1Cfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/GR1T1/urdf/GR1T1_lower_limb.urdf'

    class rewards(GR1T1Cfg.rewards):
        class scales(GR1T1Cfg.rewards.scales):
            termination = -0.0
            collision = -0.0
            stand_still = -10.0

            cmd_diff_lin_vel_x = 2.0
            cmd_diff_lin_vel_y = 0.5
            cmd_diff_lin_vel_z = 0.1
            cmd_diff_ang_vel_roll = 0.1
            cmd_diff_ang_vel_pitch = 0.1
            cmd_diff_ang_vel_yaw = 1.0
            cmd_diff_base_height = 0.5

            cmd_diff_base_orient = 0.25
            cmd_diff_torso_orient = 0.5
            cmd_diff_forehead_orient = 0.0

            action_diff = -5.0
            action_diff_diff = -1.0

            dof_acc_new = -0.25
            dof_tor_new = -0.05
            dof_tor_ankle_feet_lift_up = -0.5

            pose_offset = -1.0

            limits_dof_pos = -10.00
            limits_dof_vel = -5.00
            limits_dof_tor = -1.00

            feet_speed_xy_close_to_ground = 0.50
            feet_speed_z_close_to_height_target = 0.0

            feet_air_time = 4.0
            feet_air_height = 4.0  # 1.0
            feet_air_force = 1.0
            feet_land_time = -1.0

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
        clip_actions_max = \
            actions_max \
            + numpy.deg2rad(numpy.array([
                30, 30, 30, 30, 30,  # left leg
                30, 30, 30, 30, 30,  # right leg
            ]))
        clip_actions_min = \
            actions_min \
            - numpy.deg2rad(numpy.array([
                30, 30, 30, 30, 30,  # left leg
                30, 30, 30, 30, 30,  # right leg
            ]))


class GR1T1LowerLimbCfgPPO(GR1T1CfgPPO, GR1T1LowerLimbCfg):
    class runner(GR1T1CfgPPO.runner):
        run_name = 'gr1t1_lower_limb'
        max_iterations = 4000

    class algorithm(GR1T1CfgPPO.algorithm):
        desired_kl = 0.03

    class policy(GR1T1CfgPPO.policy):
        pass
