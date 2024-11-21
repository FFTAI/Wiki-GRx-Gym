from legged_gym.envs.gr1t1.gr1t1_lower_limb_config import (
    GR1T1LowerLimbCfg as LeggedRobotFFTAICfg,
    GR1T1LowerLimbCfgPPO as LeggedRobotFFTAICfgPPO,
)


class GR1T2LowerLimbCfg(LeggedRobotFFTAICfg):
    class asset(LeggedRobotFFTAICfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/GR1T2/urdf/GR1T2_lower_limb.urdf'


class GR1T2LowerLimbCfgPPO(LeggedRobotFFTAICfgPPO, GR1T2LowerLimbCfg):
    class runner(LeggedRobotFFTAICfgPPO.runner):
        run_name = 'gr1t2_lower_limb'
