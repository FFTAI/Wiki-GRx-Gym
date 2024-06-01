import math
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class LeggedRobotFFTAICfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_obs = 1
        num_actions = 1

    class rewards(LeggedRobotCfg.rewards):
        sigma_action_diff = -0.1
        sigma_action_diff_diff = -1.0

    class sim(LeggedRobotCfg.sim):
        dt = 0.001

        class physx(LeggedRobotCfg.sim.physx):
            num_position_iterations = 4
            num_velocity_iterations = 0


class LeggedRobotFFTAICfgPPO(LeggedRobotCfgPPO):
    pass
