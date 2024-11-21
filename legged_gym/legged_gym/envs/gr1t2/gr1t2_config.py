from legged_gym.envs.gr1t1.gr1t1_config import (
    GR1T1Cfg as LeggedRobotFFTAICfg,
    GR1T1CfgPPO as LeggedRobotFFTAICfgPPO,
)


class GR1T2Cfg(LeggedRobotFFTAICfg):
    class asset(LeggedRobotFFTAICfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/GR1T2/urdf/GR1T2.urdf'


class GR1T2CfgPPO(LeggedRobotFFTAICfgPPO, GR1T2Cfg):
    class runner(LeggedRobotFFTAICfgPPO.runner):
        run_name = 'gr1t2'
