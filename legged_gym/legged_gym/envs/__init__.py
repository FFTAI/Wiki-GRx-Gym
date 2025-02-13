# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

# Fourier Intelligence GRx Robots
from .fftai.legged_robot_fftai import LeggedRobotFFTAI
from .fftai.legged_robot_fftai_config import (
    LeggedRobotFFTAICfg,
    LeggedRobotFFTAICfgPPO,
)
from .gr1t1.gr1t1 import GR1T1
from .gr1t1.gr1t1_lower_limb_config import (
    GR1T1LowerLimbCfg as GR1T1Cfg,
    GR1T1LowerLimbCfgPPO as GR1T1CfgPPO,
)
from .gr1t2.gr1t2 import GR1T2
from .gr1t2.gr1t2_lower_limb_config import (
    GR1T2LowerLimbCfg as GR1T2Cfg,
    GR1T2LowerLimbCfgPPO as GR1T2CfgPPO,
)

from legged_gym.utils.task_registry import task_registry

# register Fourier Intelligence GRx robots
task_registry.register("GR1T1", GR1T1, GR1T1Cfg(), GR1T1CfgPPO())
task_registry.register("GR1T2", GR1T2, GR1T2Cfg(), GR1T2CfgPPO())
