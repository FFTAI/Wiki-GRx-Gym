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

import math
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class LeggedRobotFFTAICfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        obs_profile = ''
        size_mh = [0, 0]
        num_mh = 0
        num_obs = 1
        num_stack = 1
        actor_num_output = 1

        # encoder
        encoder_profile = None
        num_encoder_input = 0
        num_encoder_output = 0

    class control(LeggedRobotCfg.control):
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

        # delay: Number of control action delayed @ sim DT
        delay_mean = 0
        delay_std = 0

    class rewards(LeggedRobotCfg.rewards):
        sigma_action_diff_diff = -1.0
        sigma_action_diff_diff_hip_roll = -1.0

    class sim(LeggedRobotCfg.sim):
        dt = 0.001

        class physx(LeggedRobotCfg.sim.physx):
            num_position_iterations = 4  # 4
            num_velocity_iterations = 4  # 0


class LeggedRobotFFTAICfgPPO(LeggedRobotCfgPPO):
    pass
