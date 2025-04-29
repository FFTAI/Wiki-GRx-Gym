from legged_gym.utils.task_registry import task_registry

# ------------------------------------------------------------

from legged_gym.envs.n1.n1_code import N1
from legged_gym.envs.n1.n1_config_main_body import N1MainBodyCfg, N1MainBodyCfgPPO

task_registry.register("N1", N1, N1MainBodyCfg(), N1MainBodyCfgPPO(), )
