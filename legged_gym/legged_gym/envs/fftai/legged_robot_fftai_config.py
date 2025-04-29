import numpy
import torch

from legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)


class LeggedRobotFFTAICfg(LeggedRobotCfg):
    class sim(LeggedRobotCfg.sim):
        dt = 0.002

        class physx(LeggedRobotCfg.sim.physx):
            num_position_iterations = 4
            num_velocity_iterations = 0

    class env(LeggedRobotCfg.env):
        num_obs = 1
        num_actions = 1

    class asset(LeggedRobotCfg.asset):
        foot_thickness = 0.0  # Must be the same as in the URDF file

    class rewards(LeggedRobotCfg.rewards):
        # ---------------------------------------------------------------
        # Reward related constants
        feet_stumble_ratio = 5.0  # ratio = fxy / fz

        base_height_offset_range_limit = 0.01  # unit: m

        # ---------------------------------------------------------------
        # Reward coefficients
        sigma_collision = -1.0 * torch.e

        sigma_stand_still = -1.0 * torch.e
        sigma_stand_still_dof_pos = -1.0 * torch.e
        sigma_stand_still_dof_vel = -1.0 * torch.e

        sigma_cmd_diff_base_lin_vel_x = -1.0 * torch.e * (1.0 / 0.50)
        sigma_cmd_diff_base_lin_vel_y = -1.0 * torch.e * (1.0 / 1.00)
        sigma_cmd_diff_base_ang_vel_yaw = -1.0 * torch.e * (1.0 / 3.00)

        sigma_base_lin_vel_xy = -1.0 * torch.e
        sigma_base_lin_vel_z = -1.0 * torch.e
        sigma_base_avg_lin_vel_z = -1.0 * torch.e
        sigma_base_ang_vel_roll = -1.0 * torch.e
        sigma_base_ang_vel_pitch = -1.0 * torch.e

        sigma_base_height_offset = -10.0 * torch.e
        sigma_base_orient_offset = -1.0 * torch.e

        sigma_base_flat_orient = -5.0 * torch.e
        sigma_torso_flat_orient = -5.0 * torch.e

        sigma_action_diff = -0.1
        sigma_action_diff_diff = -1.0

        sigma_dof_pos_offset = -0.1
        sigma_dof_vel = -0.01
        sigma_dof_acc = -0.001 * torch.e  # reward calculate: 0 ~ 1.0
        sigma_dof_tor = -0.01 * torch.e  # reward calculate: 0 ~ 1.0

        sigma_limits_action = -1.0
        sigma_limits_dof_pos = -1.0
        sigma_limits_dof_vel = -10.0
        sigma_limits_dof_tor = -0.1

        sigma_feet_stumble = -1.0

    class normalization(LeggedRobotCfg.normalization):
        actions_max = numpy.array([
            1.0, 1.0,
        ])
        actions_min = numpy.array([
            -1.0, -1.0,
        ])

        clip_observations = 100.0
        clip_actions_max = \
            actions_max \
            + numpy.array([
                1.0, 1.0,
            ])
        clip_actions_min = \
            actions_min \
            - numpy.array([
                1.0, 1.0,
            ])

    class mirror:
        enable_mirror = False
        observations_coefficient = \
            numpy.ones(0)
        observations_exchange = \
            numpy.array([])
        actions_coefficient = \
            numpy.ones(0)
        actions_exchange = \
            numpy.array([])


class LeggedRobotFFTAICfgPPO(LeggedRobotCfgPPO, LeggedRobotFFTAICfg):
    pass
