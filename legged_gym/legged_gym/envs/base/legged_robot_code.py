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
from re import match

import numpy as np
from sympy.physics.units import action

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import os
import random
import numpy

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.gym_math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.base.base_task_code import BaseTask
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

# log colors
GRAY = "\033[90m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"


class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): "cuda" or "cpu"
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """

        # 标志位置 False
        self.init_done = False

        self.cfg: LeggedRobotCfg = cfg
        self.height_samples = None
        self.debug_viz = True  # show measure height square

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self._parse_cfg()

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._init_buffers()
        self._prepare_reward_function()

        # 标志位置 True
        self.init_done = True

    def _init_cfg(self, cfg: LeggedRobotCfg):
        super()._init_cfg(cfg)

    def _parse_cfg(self):

        print("##############################################")
        print("Robot Parse Config")

        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)

        if self.cfg.commands.curriculum:
            self.command_ranges["lin_vel_x"] = [
                -self.cfg.commands.curriculum_chg_lin_vel_x,
                +self.cfg.commands.curriculum_chg_lin_vel_x,
            ]
            self.command_ranges["lin_vel_y"] = [
                -self.cfg.commands.curriculum_chg_lin_vel_y,
                +self.cfg.commands.curriculum_chg_lin_vel_y,
            ]
            self.command_ranges["ang_vel_yaw"] = [
                -self.cfg.commands.curriculum_chg_ang_vel_yaw,
                +self.cfg.commands.curriculum_chg_ang_vel_yaw,
            ]

        print("self.command_ranges = \n", self.command_ranges)

        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            self.cfg.terrain.curriculum = False

        self._init_episode_length()

        self.resample_command_interval = int(self.cfg.commands.resample_command_interval_s / self.dt)

        self.cfg.domain_rand.drag_interval = np.ceil(self.cfg.domain_rand.drag_interval_s / self.dt)
        self.cfg.domain_rand.drag_keep = np.ceil(self.cfg.domain_rand.drag_keep_s / self.dt)

    def _init_episode_length(self, episode_length_s=None):
        """
        Initialize the episode length in simulation steps

        Args:
            episode_length_s (float): Episode length in seconds
        """
        if episode_length_s is None:
            episode_length_s = self.cfg.env.episode_length_s
        else:
            episode_length_s = episode_length_s

        self.max_episode_length_s = episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        print(f"{GREEN}##############################################{RESET}")
        print(f"{GREEN}Robot Init Episode Length{RESET}")
        print(f"{GREEN}self.max_episode_length_s = {self.max_episode_length_s}{RESET}")
        print(f"{GREEN}self.max_episode_length = {self.max_episode_length}{RESET}")
        print(f"{GREEN}##############################################{RESET}")

    def _init_buffers(self):
        """
        Initialize torch tensors which will contain simulation states and processed quantities
        """
        print("##############################################")
        print("Robot Init Buffers")

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # FIXME: This code can only handle one env with one actor situation
        # get robot mass (from env, after mass randomization)
        self.robot_mass = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            env_handle = self.env_handles[i]
            actor_handle = self.actor_handles[i]

            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)

            robot_mass = 0
            for j in range(len(body_props)):
                robot_mass += body_props[j].mass

            self.robot_mass[i] = robot_mass

        # create some wrapper tensors for different slices
        self.all_actors_root_states = gymtorch.wrap_tensor(actor_root_state)
        self.all_actors_dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        self.all_actors_net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)
        self.all_actors_rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor)

        print("self.all_actors_root_states.shape = \n", self.all_actors_root_states.shape)
        print("self.all_actors_dof_states.shape = \n", self.all_actors_dof_states.shape)
        print("self.all_actors_net_contact_forces.shape = \n", self.all_actors_net_contact_forces.shape)
        print("self.all_actors_rigid_body_states.shape = \n", self.all_actors_rigid_body_states.shape)

        self.robot_actor_root_states = self.all_actors_root_states[0: self.num_envs]
        self.robot_actor_dof_states = self.all_actors_dof_states[0: self.num_envs * self.num_dofs]
        self.robot_actor_net_contact_forces = self.all_actors_net_contact_forces[0: self.num_envs * self.num_bodies]
        self.robot_actor_rigid_body_states = self.all_actors_rigid_body_states[0: self.num_envs * self.num_bodies]

        print("self.num_all_envs = \n", self.num_all_envs)
        print("self.num_envs = \n", self.num_envs)
        print("self.num_dofs = \n", self.num_dofs)
        print("self.num_bodies = \n", self.num_bodies)
        print("self.robot_actor_root_states.shape = \n", self.robot_actor_root_states.shape)
        print("self.robot_actor_dof_states.shape = \n", self.robot_actor_dof_states.shape)
        print("self.robot_actor_net_contact_forces.shape = \n", self.robot_actor_net_contact_forces.shape)
        print("self.robot_actor_rigid_body_states.shape = \n", self.robot_actor_rigid_body_states.shape)

        self.root_states = self.all_actors_root_states[0: self.num_envs]
        self.dof_states = self.all_actors_dof_states[0: self.num_envs * self.num_dofs]
        self.contact_forces = self.robot_actor_net_contact_forces.view(self.num_envs, -1, 3)
        self.rigid_body_states = self.robot_actor_rigid_body_states.view(self.num_envs, -1, 13)

        print("self.root_states.shape = \n", self.root_states.shape)
        print("self.dof_states.shape = \n", self.dof_states.shape)
        print("self.contact_forces.shape = \n", self.contact_forces.shape)
        print("self.rigid_body_states.shape = \n", self.rigid_body_states.shape)

        self.dof_pos = self.dof_states.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.last_dof_pos = torch.zeros_like(self.dof_pos)

        self.dof_vel = self.dof_states.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        self.dof_acc = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_pos_offset = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)

        self.dof_tor = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_tor = torch.zeros_like(self.dof_tor)

        self.base_pos = self.root_states[:, 0:3]  # in world frame
        self.base_quat = self.root_states[:, 3:7]  # in world frame
        self.base_heading = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}

        self.obs_noise_scale_vec = self.compute_obs_noise_scale_vec()

        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))

        # actions
        self.actions_untreated = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.clip_actions_max = torch.tensor(+self.cfg.normalization.clip_actions).to(torch.float32).to(self.device)
        self.clip_actions_min = torch.tensor(-self.cfg.normalization.clip_actions).to(torch.float32).to(self.device)

        self.action_indices = torch.zeros(self.num_actions, dtype=torch.int, device=self.device, requires_grad=False)
        self.action_scales = torch.ones(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.action_scales_inv = torch.ones(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        # ufc = under of control
        self.ufc_indices = torch.zeros(self.num_actions, dtype=torch.int, device=self.device, requires_grad=False)

        # ofc = out of control
        self.ofc_indices = torch.zeros(self.num_dofs - self.num_actions, dtype=torch.int, device=self.device, requires_grad=False)
        self.ofc_dof_pos = torch.zeros(self.num_envs, self.num_dofs - self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        # dof
        self.control_types = torch.zeros(self.num_dofs, dtype=torch.int, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)

        self.dof_pos_offset_scales = torch.ones(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_vel_scales = torch.ones(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_tor_scales = torch.ones(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)

        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)

        # commands
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_base_heading = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)  # 期望的 heading

        self._init_commands_scale()

        # base related buffers
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        self.base_heights_offset = torch.zeros(self.num_envs,
                                               1,
                                               dtype=torch.float, device=self.device, requires_grad=False)
        self.surround_heights_offset = torch.zeros(self.num_envs,
                                                   len(self.cfg.terrain.measured_points_x)
                                                   * len(self.cfg.terrain.measured_points_y),
                                                   dtype=torch.float, device=self.device, requires_grad=False)

        self.base_projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.torso_projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # ----------------------------------------------------------------------------------------------------

        print("self.dof_names:")
        for name in self.dof_names:
            print(f"  - {name}")
        print("")

        # ----------------------------------------------------------------------------------------------------
        # ACTION related buffers
        self._init_buffers_actions()

        # ----------------------------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------------------
        # DOF related buffers
        self._init_buffers_dofs()

        # ----------------------------------------------------------------------------------------------------
        # get joint indices
        self._init_buffers_joint_indices()

        # get measure height
        self._init_buffers_measure_heights()

        # get command curriculum counts
        self._init_buffers_curriculum_commands()

        # others
        self._init_buffers_others()

        # ----------------------------------------------------------------------------------------------------

    def _init_commands_scale(self):
        """
        Initialize the command scale vector
        """
        if self.cfg.commands.command_profile == "not_use":
            self.commands_scale = torch.tensor(data=[
                0.0,
            ], device=self.device, requires_grad=False, )

        elif self.cfg.commands.command_profile == "base_velocity":
            self.commands_scale = torch.tensor(data=[
                self.obs_scales.lin_vel,
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
            ], device=self.device, requires_grad=False, )

        else:
            self.commands_scale = torch.ones(self.cfg.commands.num_commands,
                                             device=self.device, requires_grad=False)

    def _init_buffers_actions(self):
        """
        Initialize the action related buffers
        """

        action_index = 0

        for i in range(self.num_dofs):
            name = self.dof_names[i]

            # action name
            action_name_found = False

            for dof_name in self.cfg.control.action_names:
                if dof_name in name:
                    action_name_found = True

            if action_name_found:
                self.action_indices[action_index] = i
                action_index += 1

        print(f"{CYAN}self.action_indexes:{RESET}")
        for i, index in enumerate(self.action_indices):
            print(f"{CYAN}  - {self.dof_names[self.action_indices[i]]}: {index}{RESET}")
        print(f"{CYAN}{RESET}")

        all_indices = set(range(self.num_dofs))
        action_indices = set(self.action_indices.tolist())
        ofc_indices = all_indices - action_indices

        self.ufc_indices = torch.tensor(list(action_indices), dtype=torch.int, device=self.device, requires_grad=False)
        self.ofc_indices = torch.tensor(list(ofc_indices), dtype=torch.int, device=self.device, requires_grad=False)

        print(f"{CYAN}self.ufc_indices:{RESET}")
        for i, index in enumerate(self.ufc_indices):
            print(f"{CYAN}  - {self.dof_names[self.ufc_indices[i]]}: {index}{RESET}")
        print(f"{CYAN}{RESET}")

        print(f"{CYAN}self.ofc_indices:{RESET}")
        for i, index in enumerate(self.ofc_indices):
            print(f"{CYAN}  - {self.dof_names[self.ofc_indices[i]]}: {index}{RESET}")
        print(f"{CYAN}{RESET}")

        action_index = 0

        for i in range(self.num_dofs):
            name = self.dof_names[i]

            # action scales
            action_scale_found = False
            action_scale = 0

            for dof_name in self.cfg.control.action_scale.keys():
                if dof_name in name:
                    action_scale_found = True
                    action_scale = self.cfg.control.action_scale[dof_name]
                    break

            if action_scale_found:
                self.action_scales[action_index] = action_scale
                action_index += 1
            else:
                print(
                    f"{RED}Action scale of joint {name} were not defined!{RESET}"
                )

        print("self.action_scales:")
        for i, scale in enumerate(self.action_scales):
            print(f"  - {self.dof_names[self.action_indices[i]]}: {scale}")
        print("")

        """
        Jason 2025-03-04:
        action_scales_inv 为 action_scales 的倒数，主要为方便一些除以 action_scales 的计算
        - 如果 action_scales 为 0.0，则 action_scales_inv 为 0.0
        - 如果 action_scales 不为 0.0，则 action_scales_inv 为 1.0 / action_scales
        """
        for i, scale in enumerate(self.action_scales):
            if scale == 0.0:
                self.action_scales_inv[i] = 0.0
            else:
                self.action_scales_inv[i] = 1.0 / scale

        print("self.action_scales_inv:")
        for i, scale in enumerate(self.action_scales_inv):
            print(f"  - {self.dof_names[self.action_indices[i]]}: {scale}")
        print("")

    def _init_buffers_dofs(self):
        """
        Initialize the DOF related buffers
        """

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.num_dofs):
            name = self.dof_names[i]

            # default joint angles
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle

            # control type
            control_type_found = False

            for dof_name in self.cfg.control.control_type.keys():
                if dof_name in name:
                    self.control_types[i] = self.cfg.control.control_type[dof_name]
                    control_type_found = True

            if not control_type_found:
                self.control_types[i] = 0.0
                print(
                    f"{RED}Control type of joint {name} were not defined, setting them to zero(0){RESET}"
                )

            # PD gains
            stiffness_found = False
            damping_found = False

            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    stiffness_found = True

            if not stiffness_found:
                self.p_gains[i] = 0.0
                if self.control_types[i] in [1, 2]:
                    print(
                        f"{RED}"
                        f"P gain of joint {name} were not defined, setting them to zero(0)"
                        f"{RESET}"
                    )

            for dof_name in self.cfg.control.damping.keys():
                if dof_name in name:
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    damping_found = True

            if not damping_found:
                self.d_gains[i] = 0.0
                if self.control_types[i] in [1, 2]:
                    print(
                        f"{RED}"
                        f"D gain of joint {name} were not defined, setting them to zero(0)"
                        f"{RESET}"
                    )

            # dof_pos_offset scales
            dof_pos_offset_scale_found = False

            for dof_name in self.cfg.control.dof_pos_offset_scale.keys():
                if dof_name in name:
                    self.dof_pos_offset_scales[i] = self.cfg.control.dof_pos_offset_scale[dof_name]
                    dof_pos_offset_scale_found = True

            if not dof_pos_offset_scale_found:
                self.dof_pos_offset_scales[i] = 1.0
                print(
                    f"{RED}Pos offset scale of joint {name} were not defined, setting them to one(1){RESET}"
                )

            # dof_vel scales
            dof_vel_scale_found = False

            for dof_name in self.cfg.control.dof_vel_scale.keys():
                if dof_name in name:
                    self.dof_vel_scales[i] = self.cfg.control.dof_vel_scale[dof_name]
                    dof_vel_scale_found = True

            if not dof_vel_scale_found:
                self.dof_vel_scales[i] = 1.0
                print(
                    f"{RED}Dof vel scale of joint {name} were not defined, setting them to one(1){RESET}"
                )

            # dof_tor scales
            dof_tor_scale_found = False

            for dof_name in self.cfg.control.dof_tor_scale.keys():
                if dof_name in name:
                    self.dof_tor_scales[i] = self.cfg.control.dof_tor_scale[dof_name]
                    dof_tor_scale_found = True

            if not dof_tor_scale_found:
                self.dof_tor_scales[i] = 1.0
                print(
                    f"{RED}Dof tor scale of joint {name} were not defined, setting them to one(1){RESET}"
                )

        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)  # add env dim

        print("self.control_types:")
        for i, control_type in enumerate(self.control_types):
            print(f"  - {self.dof_names[i]}: {control_type}")
        print("")

        print("self.p_gains:")
        for i, gain in enumerate(self.p_gains):
            print(f"  - {self.dof_names[i]}: {gain}")
        print("")

        print("self.d_gains:")
        for i, gain in enumerate(self.d_gains):
            print(f"  - {self.dof_names[i]}: {gain}")
        print("")

        print("self.default_dof_pos:")
        for i, pos in enumerate(self.default_dof_pos[0]):
            print(f"  - {self.dof_names[i]}: {pos}")
        print("")

        print("self.dof_pos_offset_scales:")
        for i, scale in enumerate(self.dof_pos_offset_scales):
            print(f"  - {self.dof_names[i]}: {scale}")
        print("")

        print("self.dof_vel_scales:")
        for i, scale in enumerate(self.dof_vel_scales):
            print(f"  - {self.dof_names[i]}: {scale}")
        print("")

        print("self.dof_tor_scales:")
        for i, scale in enumerate(self.dof_tor_scales):
            print(f"  - {self.dof_names[i]}: {scale}")
        print("")

    def _init_buffers_joint_indices(self):
        pass

    def _init_buffers_measure_heights(self):
        # measured height
        if self.cfg.terrain.measure_heights:
            self.height_points, self.num_height_points = self._init_height_points()
        self.measured_heights = 0

    def _init_buffers_curriculum_commands(self):
        if self.cfg.commands.curriculum:
            self.curriculum_count_cmd_diff_base_lin_vel_x = 0
            self.curriculum_count_cmd_diff_base_lin_vel_y = 0
            self.curriculum_count_cmd_diff_base_ang_vel_yaw = 0
            self.curriculum_count_max_cmd_diff_base_lin_vel_x = 1000
            self.curriculum_count_max_cmd_diff_base_lin_vel_y = 1000
            self.curriculum_count_max_cmd_diff_base_ang_vel_yaw = 1000

    def _init_buffers_others(self):
        pass

    # ---------------------------------------

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,
                                       self.sim_params)

        mesh_type = self.cfg.terrain.mesh_type

        if mesh_type in ["heightfield", "trimesh"]:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)

        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. "
                             "Allowed types are [None, plane, heightfield, trimesh]")

        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ---------------------------------------

    def clip_actions(self, actions):
        actions_clipped = torch.clip(actions, self.clip_actions_min, self.clip_actions_max).to(self.device)
        return actions_clipped

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        """
        Jason 2025-03-08:
        clone() make sure that the actions_untreated is not changed.
        """
        self.actions_untreated = actions.clone()

        """
        Jason 2024-11-22:
        这里的 action 和 self.actions 是不一样的。
        - actions 在传入之前，已经被 OnPolicyRunner 完成了数据的记录了。
        - self.actions 只是一个 buffer，用于传入 observations 和 rewards 的计算。
        """
        self.actions = self.clip_actions(actions)

        self.before_physics_step()
        self.render()
        self.during_physics_step()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and extras
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        if self.pri_obs_buf is not None:
            self.pri_obs_buf = torch.clip(self.pri_obs_buf, -clip_obs, clip_obs)

        # ------------------------------
        # output

        step_return = (
            self.get_observations(),
            self.get_privileged_observations(),
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

        return step_return

    def before_physics_step(self):
        pass

    def during_physics_step(self):
        self._during_physics_step_before_sim()

        for deci in range(self.cfg.control.decimation):
            # select actions based on control delay
            compute_torques_actions = torch.where(deci > self.control_delay,
                                                  self.actions,
                                                  self.last_actions)

            # computer torques based on controller actions
            self.dof_tor = self._compute_torques(compute_torques_actions).view(self.dof_tor.shape)

            self._during_physics_step_in_sim()

        self._during_physics_step_after_sim()

    def _during_physics_step_before_sim(self):
        # add control delay randomness
        if self.cfg.domain_rand.randomize_control_delay:
            control_delay_range_lower = self.cfg.domain_rand.control_delay_s_range[0] / self.sim_params.dt
            control_delay_range_upper = self.cfg.domain_rand.control_delay_s_range[1] / self.sim_params.dt

            self.control_delay = \
                torch_rand_sqrt_float(
                    lower=control_delay_range_lower,
                    upper=control_delay_range_upper,
                    shape=(self.num_envs, 1),
                    device=self.device,
                )

            # for value lower than 0.5, set to 0.0
            self.control_delay = \
                torch.where(self.control_delay < 0.5, torch.zeros_like(self.control_delay), self.control_delay)

        else:
            self.control_delay = \
                torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

    def _during_physics_step_in_sim(self):
        # global self.dof_tor -> dof_tor_output
        dof_tor_output = \
            torch.zeros(self.num_all_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        dof_tor_output[0: self.num_envs] = self.dof_tor

        # actor movement
        env_ids_int32 = torch.arange(start=0, end=self.num_envs, step=1, device=self.device).to(dtype=torch.int32)

        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(dof_tor_output),
                                                        gymtorch.unwrap_tensor(env_ids_int32),
                                                        len(env_ids_int32))

        # simulate
        self.gym.simulate(self.sim)

        if self.device != "cpu":
            self.gym.fetch_results(self.sim, True)

        # refresh state (should be called only once in one step)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def _during_physics_step_after_sim(self):
        # compute some quantities
        self.dof_acc = (self.dof_vel - self.last_dof_vel) / self.dt

    def post_physics_step(self):
        """
        Post physics step:
            - check terminations
            - compute observations and rewards
            - calls self._post_physics_step_callback() for common computations
            - calls self._draw_debug_vis() if needed
        """
        # refresh state (should be called only once in one step)
        # self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_net_contact_force_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)

        # update counter
        self.common_step_counter += 1
        self.episode_length_buf += 1

        # update state
        self.post_physics_step_update_state()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()

        # reset some environments
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids=reset_env_ids)

        # prepare updated states (root states and dof states)
        self.prepare_update_states()

        # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self.compute_observations()

        # record last values
        self.record_last_values()

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return reset_env_ids

    def post_physics_step_update_state(self):
        self.base_pos[:] = self.root_states[:, 0:3]  # in world frame
        self.base_quat[:] = self.root_states[:, 3:7]  # in world frame

        base_forward = quat_apply(self.base_quat, self.forward_vec)
        self.base_heading[:] = torch.atan2(base_forward[:, 1], base_forward[:, 0]).unsqueeze(1)  # in world frame

        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])  # in local frame
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])  # in local frame

        self.base_projected_gravity[:] = quat_rotate_inverse(
            self.base_quat,
            self.gravity_vec
        )
        self.torso_projected_gravity[:] = quat_rotate_inverse(
            self.rigid_body_states[:, self.torso_indices][:, 0, 3:7],
            self.gravity_vec
        )

        self.dof_pos_offset[:] = self.dof_pos - self.default_dof_pos
        self.dof_vel[:] = self.dof_vel[:]

        # using time to resample commands
        if self.resample_command_interval > 0:
            resample_commands_env_ids = \
                ((self.episode_length_buf
                  % self.resample_command_interval
                  == 0).nonzero(as_tuple=False).flatten())
            self._resample_commands(resample_commands_env_ids)

        self._auto_heading()

        # measure height
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        # drag robots
        if (
                self.cfg.domain_rand.drag_robots
                and ((self.common_step_counter % self.cfg.domain_rand.drag_interval) >
                     (self.cfg.domain_rand.drag_interval - self.cfg.domain_rand.drag_keep))
        ):
            drag_forces, drag_torques = self._drag_robots()

    def check_termination(self):
        """
        Check if environments need to be reset
        """
        # detect contact forces at termination links
        self.reset_buf = \
            torch.any(
                torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
                dim=1,
            )

        # detect base tilt too much (roll and pitch)
        self.reset_buf |= \
            torch.abs(self.base_projected_gravity[:, 2]) \
            < self.cfg.asset.terminate_project_gravity_less_than

        # no terminal reward for time-outs
        self.time_out_buf = \
            self.episode_length_buf > self.max_episode_length

        self.reset_buf |= self.time_out_buf

    def compute_reward(self):
        """
        Compute rewards:
            Calls each reward function which had a non-zero scale
            (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]

            self.rew_buf += rew
            self.episode_sums[name] += rew

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def reset_idx(self, env_ids):
        """
        Reset some environments.
            Calls
                self._reset_states(env_ids)
                self._reset_actions(env_ids)
                self._reset_others(env_ids)
                self._resample_commands(env_ids)
                self.reset_last_values(env_ids)
                self._reset_observations(env_ids)
                self.reset_buf[env_ids] = 1

            [Optional] calls
                self._update_terrain_curriculum(env_ids)
                self._update_command_curriculum(env_ids)
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        # ----------------------------------

        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum:
            self._update_command_curriculum(env_ids)

        # reset robot states
        self._reset_states(env_ids)

        # reset actions
        self._reset_actions(env_ids)

        # reset others
        self._reset_others(env_ids)

        # reset commands
        self._resample_commands(env_ids)

        # reset buffers
        self.reset_last_values(env_ids)

        # reset observations
        self._reset_observations(env_ids)

        # reset episode length
        if self.cfg.domain_rand.randomize_reset_episode_length:
            self.episode_length_buf[env_ids] = torch.randint_like(self.episode_length_buf[env_ids],
                                                                  high=int(self.max_episode_length))
        else:
            self.episode_length_buf[env_ids] = 0

        # reset flag
        self.reset_buf[env_ids] = 1

        # ----------------------------------

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.

        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

        if self.cfg.commands.curriculum:
            self.extras["episode"]["min_command_x"] = self.command_ranges["lin_vel_x"][0]
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            self.extras["episode"]["min_command_y"] = self.command_ranges["lin_vel_y"][0]
            self.extras["episode"]["max_command_y"] = self.command_ranges["lin_vel_y"][1]
            self.extras["episode"]["min_command_yaw"] = self.command_ranges["ang_vel_yaw"][0]
            self.extras["episode"]["max_command_yaw"] = self.command_ranges["ang_vel_yaw"][1]

        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def prepare_update_states(self):
        """
        Prepare states for the next step
        """
        pass

    def compute_observations(self):
        """
        Computes observations
        """
        # compute observations variables
        self.compute_observation_variables()

        # compute observations profile
        self.compute_observation_profile()

        # calculate observation noise
        self.compute_observation_noise()

        # calculate observation stack
        self.compute_observation_stack()

    def compute_observation_variables(self):
        self.dof_pos_offset = \
            self.dof_pos - self.default_dof_pos

        self.base_heights_offset = \
            torch.mean(
                torch.clip(
                    self.root_states[:, 2:3]
                    - self.cfg.rewards.base_height_target
                    - self.measured_heights,
                    min=-1.0,
                    max=1.0),
                dim=1).unsqueeze(1)

        self.surround_heights_offset = \
            torch.clip(self.root_states[:, 2:3]
                       - self.cfg.rewards.base_height_target
                       - self.measured_heights,
                       min=-1.0,
                       max=1.0)

    def compute_observation_profile(self):
        self.obs_buf = torch.cat(
            (
                # base related
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.base_projected_gravity,
                self.commands[:, :3] * self.commands_scale,

                # dof related
                self.dof_pos_offset * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions * self.obs_scales.action,
            ), dim=-1)

    def compute_observation_noise(self):
        # add noise if needed, only add noise to the actor observations
        if self.cfg.noise.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.obs_noise_scale_vec

    def compute_observation_stack(self):
        # stack obs_buf to obs_stack
        if self.actor_obs_use_stack:
            self.obs_stack = torch.cat((self.obs_stack[:, self.num_obs:], self.obs_buf), dim=1)

            self.obs_stack_num_stacked += 1
            self.obs_stack_num_stacked = torch.clamp(self.obs_stack_num_stacked, 0, self.num_stack)

    def compute_obs_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        self.noise_scales = self.cfg.noise.noise_scales
        self.noise_level = self.cfg.noise.noise_level

        obs_noise_scale_vec = self.compute_obs_noise_scale_vec_profile()

        print("obs_noise_scale_vec = ", obs_noise_scale_vec, "\n")
        return obs_noise_scale_vec

    def compute_obs_noise_scale_vec_profile(self):
        noise_vec = torch.zeros_like(self.obs_buf[0])

        return noise_vec

    def reset_last_values(self, env_ids):
        self.last_dof_pos[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.last_dof_tor[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0

    def record_last_values(self):
        self.last_dof_pos[:] = self.dof_pos[:]  # auto update
        self.last_dof_vel[:] = self.dof_vel[:]  # auto update
        self.last_dof_tor[:] = self.dof_tor[:]
        self.last_actions[:] = self.actions[:]

    # ------------- Callbacks --------------

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0],
                                                    friction_range[1],
                                                    (num_buckets, 1), device="cpu")

                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id == 0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                restitution_buckets = torch_rand_float(restitution_range[0],
                                                       restitution_range[1],
                                                       (num_buckets, 1), device="cpu")

                self.restitution_coeffs = restitution_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """

        # ---------------------------------------

        # log info
        if env_id == 0:
            print(
                f"{RED}"
                f"############################################## \n"
                f"Robot DOF properties origin: \n"
                f"props: \n {props} \n"
                f"############################################## \n"
                f"{RESET}"
            )

        # ---------------------------------------

        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_tor_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)

            self.soft_dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.soft_dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
            self.soft_dof_tor_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)

            for i in range(len(props)):
                """
                Jason 2024-11-21:
                在给 urdf 添加位置、速度、力矩限制时，需要注意以下几点：
                - 位置限制按照关节的 [结构下限, 结构上限] 设置
                - 速度限制按照关节的 [额定速度] 设置
                - 力矩限制按照关节的 [额定力矩 * 3] 设置
                - 功率限制按照关节的 [额定力矩 * 额定速度] 设置
                """
                # hard limits
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.dof_tor_limits[i] = props["effort"][i].item()

                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.soft_dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.soft_dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.soft_dof_vel_limits[i] = self.dof_vel_limits[i] * self.cfg.rewards.soft_dof_vel_limit
                self.soft_dof_tor_limits[i] = self.dof_tor_limits[i] * self.cfg.rewards.soft_dof_tor_limit

            print("self.dof_pos_limits: \n", self.dof_pos_limits)
            print("self.dof_vel_limits: \n", self.dof_vel_limits)
            print("self.dof_tor_limits: \n", self.dof_tor_limits)

            print("self.soft_dof_pos_limits: \n", self.soft_dof_pos_limits)
            print("self.soft_dof_vel_limits: \n", self.soft_dof_vel_limits)
            print("self.soft_dof_tor_limits: \n", self.soft_dof_tor_limits)

        # randomize motor friction
        multiply_motor_friction_range = self.cfg.domain_rand.multiply_motor_friction_range

        # set dof friction
        for i in range(self.num_dofs):
            name = self.dof_names[i]

            # default friction and armature
            dof_motor_friction = self.cfg.dof.friction[name]

            # randomize motor friction
            if self.cfg.domain_rand.randomize_motor_friction:
                dof_motor_friction *= np.random.uniform(multiply_motor_friction_range[0],
                                                        multiply_motor_friction_range[1])

            found = False

            for dof_name in self.cfg.dof.friction.keys():
                if dof_name in name:
                    props["friction"][i] = dof_motor_friction
                    found = True

            if not found:
                props["friction"][i] = 0.
                print(
                    f"friction of joint {name} were not defined, setting them to zero(0)"
                )

        # randomize motor armature
        multiply_motor_armature_range = self.cfg.domain_rand.multiply_motor_armature_range

        # set dof armature
        for i in range(self.num_dofs):
            name = self.dof_names[i]

            # default friction and armature
            dof_motor_armature = self.cfg.dof.armature[name]

            # randomize motor armature
            if self.cfg.domain_rand.randomize_motor_armature:
                dof_motor_armature *= np.random.uniform(multiply_motor_armature_range[0],
                                                        multiply_motor_armature_range[1])

            found = False

            for dof_name in self.cfg.dof.armature.keys():
                if dof_name in name:
                    props["armature"][i] = dof_motor_armature
                    found = True

            if not found:
                props["armature"][i] = 0.
                print(
                    f"armature of joint {name} were not defined, setting them to zero(0)"
                )

        # randomize mechanism friction
        multiply_mechanism_friction_range = self.cfg.domain_rand.multiply_mechanism_friction_range

        # set mechanism friction
        for i in range(self.num_dofs):
            name = self.dof_names[i]

            # default mechanism friction
            dof_mechanism_friction = self.cfg.dof.mechanism[name]

            # randomize mechanism friction
            if self.cfg.domain_rand.randomize_mechanism_friction:
                dof_mechanism_friction *= np.random.uniform(multiply_mechanism_friction_range[0],
                                                            multiply_mechanism_friction_range[1])

            found = False

            for dof_name in self.cfg.dof.mechanism.keys():
                if dof_name in name:
                    props["friction"][i] += dof_mechanism_friction
                    found = True

            if not found:
                props["friction"][i] = 0.
                print(
                    f"friction and armature of joint {name} were not defined, setting them to zero(0)"
                )

        # ---------------------------------------

        # log info
        if env_id == 0:
            print(
                f"{RED}"
                f"############################################## \n"
                f"Robot DOF properties refined: \n"
                f"props: \n {props} \n"
                f"############################################## \n"
                f"{RESET}"
            )

        # ---------------------------------------

        return props

    def _process_rigid_body_props(self, props, env_id):

        # randomize base mass
        multiply_base_mass_range = self.cfg.domain_rand.multiply_base_mass_range

        if self.cfg.domain_rand.randomize_base_mass:
            mass = props[0].mass
            mass *= np.random.uniform(multiply_base_mass_range[0], multiply_base_mass_range[1])

            props[0].mass = mass
            props[0].invMass = 1 / mass

        # randomize base center of mass (com)
        add_base_com_range_x = self.cfg.domain_rand.add_base_com_range_x
        add_base_com_range_y = self.cfg.domain_rand.add_base_com_range_y
        add_base_com_range_z = self.cfg.domain_rand.add_base_com_range_z

        if self.cfg.domain_rand.randomize_base_com:
            comm_vec = numpy.array([
                props[0].com.x,
                props[0].com.y,
                props[0].com.z
            ])
            comm_vec += np.array([
                np.random.uniform(add_base_com_range_x[0], add_base_com_range_x[1]),
                np.random.uniform(add_base_com_range_y[0], add_base_com_range_y[1]),
                np.random.uniform(add_base_com_range_z[0], add_base_com_range_z[1]),
            ])

            props[0].com = gymapi.Vec3(comm_vec[0], comm_vec[1], comm_vec[2])

        # randomize base inertia
        multiply_base_inertia_range = self.cfg.domain_rand.multiply_base_inertia_range

        if self.cfg.domain_rand.randomize_base_inertia:
            inertia_matrix = numpy.array([
                [props[0].inertia.x.x, props[0].inertia.x.y, props[0].inertia.x.z],
                [props[0].inertia.y.x, props[0].inertia.y.y, props[0].inertia.y.z],
                [props[0].inertia.z.x, props[0].inertia.z.y, props[0].inertia.z.z],
            ])
            inertia_matrix *= np.array([
                [np.random.uniform(multiply_base_inertia_range[0], multiply_base_inertia_range[1]), 1, 1],
                [1, np.random.uniform(multiply_base_inertia_range[0], multiply_base_inertia_range[1]), 1],
                [1, 1, np.random.uniform(multiply_base_inertia_range[0], multiply_base_inertia_range[1])],
            ])
            inertia_matrix_inv = numpy.linalg.inv(inertia_matrix)

            props[0].inertia.x = gymapi.Vec3(inertia_matrix[0, 0], inertia_matrix[0, 1], inertia_matrix[0, 2])
            props[0].inertia.y = gymapi.Vec3(inertia_matrix[1, 0], inertia_matrix[1, 1], inertia_matrix[1, 2])
            props[0].inertia.z = gymapi.Vec3(inertia_matrix[2, 0], inertia_matrix[2, 1], inertia_matrix[2, 2])
            props[0].invInertia.x = gymapi.Vec3(inertia_matrix_inv[0, 0], inertia_matrix_inv[0, 1], inertia_matrix_inv[0, 2])
            props[0].invInertia.y = gymapi.Vec3(inertia_matrix_inv[1, 0], inertia_matrix_inv[1, 1], inertia_matrix_inv[1, 2])
            props[0].invInertia.z = gymapi.Vec3(inertia_matrix_inv[2, 0], inertia_matrix_inv[2, 1], inertia_matrix_inv[2, 2])

        # ---------------------------------------

        # randomize torso mass
        multiply_torso_mass_range = self.cfg.domain_rand.multiply_torso_mass_range

        if self.cfg.domain_rand.randomize_torso_mass:
            for i in self.torso_indices:
                # if is base_link, pass
                if i == 0:
                    pass

                mass = props[i].mass
                mass *= np.random.uniform(multiply_torso_mass_range[0], multiply_torso_mass_range[1])

                props[i].mass = mass
                props[i].invMass = 1 / mass

        # randomize torso center of mass (com)
        add_torso_com_range_x = self.cfg.domain_rand.add_torso_com_range_x
        add_torso_com_range_y = self.cfg.domain_rand.add_torso_com_range_y
        add_torso_com_range_z = self.cfg.domain_rand.add_torso_com_range_z

        if self.cfg.domain_rand.randomize_torso_com:
            for i in self.torso_indices:
                # if is base_link, pass
                if i == 0:
                    pass

                comm_vec = numpy.array([
                    props[i].com.x,
                    props[i].com.y,
                    props[i].com.z
                ])
                comm_vec += np.array([
                    np.random.uniform(add_torso_com_range_x[0], add_torso_com_range_x[1]),
                    np.random.uniform(add_torso_com_range_y[0], add_torso_com_range_y[1]),
                    np.random.uniform(add_torso_com_range_z[0], add_torso_com_range_z[1]),
                ])

                props[i].com = gymapi.Vec3(comm_vec[0], comm_vec[1], comm_vec[2])

        # randomize torso inertia
        multiply_torso_inertia_range = self.cfg.domain_rand.multiply_torso_inertia_range

        if self.cfg.domain_rand.randomize_torso_inertia:
            for i in self.torso_indices:
                # if is base_link, pass
                if i == 0:
                    pass

                inertia_matrix = numpy.array([
                    [props[i].inertia.x.x, props[i].inertia.x.y, props[i].inertia.x.z],
                    [props[i].inertia.y.x, props[i].inertia.y.y, props[i].inertia.y.z],
                    [props[i].inertia.z.x, props[i].inertia.z.y, props[i].inertia.z.z],
                ])
                inertia_matrix *= np.array([
                    [np.random.uniform(multiply_torso_inertia_range[0], multiply_torso_inertia_range[1]), 1, 1],
                    [1, np.random.uniform(multiply_torso_inertia_range[0], multiply_torso_inertia_range[1]), 1],
                    [1, 1, np.random.uniform(multiply_torso_inertia_range[0], multiply_torso_inertia_range[1])],
                ])
                inertia_matrix_inv = numpy.linalg.inv(inertia_matrix)

                props[i].inertia.x = gymapi.Vec3(inertia_matrix[0, 0], inertia_matrix[0, 1], inertia_matrix[0, 2])
                props[i].inertia.y = gymapi.Vec3(inertia_matrix[1, 0], inertia_matrix[1, 1], inertia_matrix[1, 2])
                props[i].inertia.z = gymapi.Vec3(inertia_matrix[2, 0], inertia_matrix[2, 1], inertia_matrix[2, 2])
                props[i].invInertia.x = gymapi.Vec3(inertia_matrix_inv[0, 0], inertia_matrix_inv[0, 1], inertia_matrix_inv[0, 2])
                props[i].invInertia.y = gymapi.Vec3(inertia_matrix_inv[1, 0], inertia_matrix_inv[1, 1], inertia_matrix_inv[1, 2])
                props[i].invInertia.z = gymapi.Vec3(inertia_matrix_inv[2, 0], inertia_matrix_inv[2, 1], inertia_matrix_inv[2, 2])

        # ---------------------------------------

        # randomize payload mass
        multiply_payload_mass_range = self.cfg.domain_rand.multiply_payload_mass_range

        if self.cfg.domain_rand.randomize_payload_mass:
            for i in self.payload_indices:
                mass = props[i].mass
                mass *= np.random.uniform(multiply_payload_mass_range[0], multiply_payload_mass_range[1])

                props[i].mass = mass
                props[i].invMass = 1 / mass

        # randomize payload center of mass (com)
        add_payload_com_range_x = self.cfg.domain_rand.add_payload_com_range_x
        add_payload_com_range_y = self.cfg.domain_rand.add_payload_com_range_y
        add_payload_com_range_z = self.cfg.domain_rand.add_payload_com_range_z

        if self.cfg.domain_rand.randomize_payload_com:
            for i in self.payload_indices:
                comm_vec = numpy.array([
                    props[i].com.x,
                    props[i].com.y,
                    props[i].com.z
                ])
                comm_vec += np.array([
                    np.random.uniform(add_payload_com_range_x[0], add_payload_com_range_x[1]),
                    np.random.uniform(add_payload_com_range_y[0], add_payload_com_range_y[1]),
                    np.random.uniform(add_payload_com_range_z[0], add_payload_com_range_z[1]),
                ])

                props[i].com = gymapi.Vec3(comm_vec[0], comm_vec[1], comm_vec[2])

        # randomize payload inertia
        multiply_payload_inertia_range = self.cfg.domain_rand.multiply_payload_inertia_range

        if self.cfg.domain_rand.randomize_payload_inertia:
            for i in self.payload_indices:
                inertia_matrix = numpy.array([
                    [props[i].inertia.x.x, props[i].inertia.x.y, props[i].inertia.x.z],
                    [props[i].inertia.y.x, props[i].inertia.y.y, props[i].inertia.y.z],
                    [props[i].inertia.z.x, props[i].inertia.z.y, props[i].inertia.z.z],
                ])
                inertia_matrix *= np.array([
                    [np.random.uniform(multiply_payload_inertia_range[0], multiply_payload_inertia_range[1]), 1, 1],
                    [1, np.random.uniform(multiply_payload_inertia_range[0], multiply_payload_inertia_range[1]), 1],
                    [1, 1, np.random.uniform(multiply_payload_inertia_range[0], multiply_payload_inertia_range[1])],
                ])
                inertia_matrix_inv = numpy.linalg.inv(inertia_matrix)

                props[i].inertia.x = gymapi.Vec3(inertia_matrix[0, 0], inertia_matrix[0, 1], inertia_matrix[0, 2])
                props[i].inertia.y = gymapi.Vec3(inertia_matrix[1, 0], inertia_matrix[1, 1], inertia_matrix[1, 2])
                props[i].inertia.z = gymapi.Vec3(inertia_matrix[2, 0], inertia_matrix[2, 1], inertia_matrix[2, 2])
                props[i].invInertia.x = gymapi.Vec3(inertia_matrix_inv[0, 0], inertia_matrix_inv[0, 1], inertia_matrix_inv[0, 2])
                props[i].invInertia.y = gymapi.Vec3(inertia_matrix_inv[1, 0], inertia_matrix_inv[1, 1], inertia_matrix_inv[1, 2])
                props[i].invInertia.z = gymapi.Vec3(inertia_matrix_inv[2, 0], inertia_matrix_inv[2, 1], inertia_matrix_inv[2, 2])

        # ---------------------------------------

        # randomize link mass
        multiply_link_mass_range = self.cfg.domain_rand.multiply_link_mass_range

        if self.cfg.domain_rand.randomize_link_mass:
            for i in range(len(props)):
                # if is base_link or torso_link, pass
                if i == 0 or i in self.torso_indices or i in self.payload_indices:
                    pass

                # other links
                mass = props[i].mass
                mass *= np.random.uniform(multiply_link_mass_range[0], multiply_link_mass_range[1])

                props[i].mass = mass
                props[i].invMass = 1 / mass

        # randomize link center of mass (com)
        add_link_com_range_x_range = self.cfg.domain_rand.add_link_com_range_x_range
        add_link_com_range_y_range = self.cfg.domain_rand.add_link_com_range_y_range
        add_link_com_range_z_range = self.cfg.domain_rand.add_link_com_range_z_range

        if self.cfg.domain_rand.randomize_link_com:
            for i in range(1, len(props)):
                # if is base_link or torso_link, pass
                if i == 0 or i in self.torso_indices or i in self.payload_indices:
                    pass

                # other links
                comm_vec = numpy.array([
                    props[i].com.x,
                    props[i].com.y,
                    props[i].com.z
                ])
                comm_vec += np.array([
                    np.random.uniform(add_link_com_range_x_range[0], add_link_com_range_x_range[1]),
                    np.random.uniform(add_link_com_range_y_range[0], add_link_com_range_y_range[1]),
                    np.random.uniform(add_link_com_range_z_range[0], add_link_com_range_z_range[1]),
                ])

                props[i].com = gymapi.Vec3(comm_vec[0], comm_vec[1], comm_vec[2])

        # randomize link inertia
        multiply_link_inertia_range = self.cfg.domain_rand.multiply_link_inertia_range

        if self.cfg.domain_rand.randomize_link_inertia:
            for i in range(1, len(props)):
                # if is base_link or torso_link, pass
                if i == 0 or i in self.torso_indices or i in self.payload_indices:
                    pass

                # other links
                inertia_matrix = numpy.array([
                    [props[i].inertia.x.x, props[i].inertia.x.y, props[i].inertia.x.z],
                    [props[i].inertia.y.x, props[i].inertia.y.y, props[i].inertia.y.z],
                    [props[i].inertia.z.x, props[i].inertia.z.y, props[i].inertia.z.z],
                ])
                inertia_matrix *= np.array([
                    [np.random.uniform(multiply_link_inertia_range[0], multiply_link_inertia_range[1]), 1, 1],
                    [1, np.random.uniform(multiply_link_inertia_range[0], multiply_link_inertia_range[1]), 1],
                    [1, 1, np.random.uniform(multiply_link_inertia_range[0], multiply_link_inertia_range[1])],
                ])
                inertia_matrix_inv = numpy.linalg.inv(inertia_matrix)

                props[i].inertia.x = gymapi.Vec3(inertia_matrix[0, 0], inertia_matrix[0, 1], inertia_matrix[0, 2])
                props[i].inertia.y = gymapi.Vec3(inertia_matrix[1, 0], inertia_matrix[1, 1], inertia_matrix[1, 2])
                props[i].inertia.z = gymapi.Vec3(inertia_matrix[2, 0], inertia_matrix[2, 1], inertia_matrix[2, 2])
                props[i].invInertia.x = gymapi.Vec3(inertia_matrix_inv[0, 0], inertia_matrix_inv[0, 1], inertia_matrix_inv[0, 2])
                props[i].invInertia.y = gymapi.Vec3(inertia_matrix_inv[1, 0], inertia_matrix_inv[1, 1], inertia_matrix_inv[1, 2])
                props[i].invInertia.z = gymapi.Vec3(inertia_matrix_inv[2, 0], inertia_matrix_inv[2, 1], inertia_matrix_inv[2, 2])

        return props

    def _resample_commands(self, env_ids=None, command_profile=None):
        """
        Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
            command_profile (str): Command profile to be used
        """
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, step=1, device=self.device)

        if command_profile is None:
            command_profile = self.cfg.commands.command_profile

        if command_profile == "not_use":
            pass

        if command_profile == "base_velocity":
            self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                         self.command_ranges["lin_vel_x"][1],
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
                                                         self.command_ranges["lin_vel_y"][1],
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                         self.command_ranges["ang_vel_yaw"][1],
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)

            if self.cfg.commands.heading_command:
                self.commands_base_heading[env_ids] = torch_rand_float(self.command_ranges["heading"][0],
                                                                       self.command_ranges["heading"][1],
                                                                       (len(env_ids), 1),
                                                                       device=self.device).squeeze(1)

            # map commands list to separate commands
            self.commands_base_lin_vel_x = self.commands[:, 0:1]
            self.commands_base_lin_vel_y = self.commands[:, 1:2]
            self.commands_base_ang_vel_yaw = self.commands[:, 2:3]

            self._command_refinement(env_ids)

    def _command_refinement(self, env_ids=None, command_profile=None):
        """
        Refine commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
            command_profile (str): Command profile to be used
        """
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, step=1, device=self.device)

        if command_profile is None:
            command_profile = self.cfg.commands.command_profile

        if command_profile == "base_velocity":
            # if belong to stand command, set all commands to zero
            flags_of_stand_command = (torch.norm(self.commands[env_ids, 0:2], dim=1) <= 0.10) \
                                     * (torch.abs(self.commands[env_ids, 2]) <= 0.10)
            self.commands[env_ids, 0:3] *= ~flags_of_stand_command.unsqueeze(1)

    def _auto_heading(self, command_profile=None):

        if command_profile is None:
            command_profile = self.cfg.commands.command_profile

        if command_profile == "base_velocity":
            # using heading command to [auto] change the yaw command
            if self.cfg.commands.heading_command:
                self.commands[:, 2:3] = \
                    torch.clip(0.5 * wrap_to_pi(self.commands_base_heading - self.base_heading),
                               self.command_ranges["ang_vel_yaw"][0],
                               self.command_ranges["ang_vel_yaw"][1])

    def _compute_torques(self, actions):
        """
        Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor, [num_envs, num_dof]): Actions to be converted to torques

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        actions_scaled = actions * self.action_scales

        # expand actions from (num_envs, num_actions) to (num_envs, num_dofs)
        actions_expanded = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float32)

        for i in range(self.num_actions):
            action_index = self.action_indices[i]
            actions_expanded[:, action_index:action_index + 1] = actions_scaled[:, i:i + 1]

        # --------------------------------------------------

        control_target = actions_expanded

        # --------------------------------------------------

        # compute torques
        torques = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float32)

        for i in range(self.num_dofs):
            control_type = self.control_types[i]

            if control_type == 1:  # Position Control (PD controller)
                torques_P_p = self.p_gain_scales[:, i:i + 1] * self.p_gains[i] \
                              * (control_target[:, i:i + 1] + self.default_dof_pos[:, i:i + 1] - self.dof_pos[:, i:i + 1])
                torques_P_d = self.d_gain_scales[:, i:i + 1] * self.d_gains[i] \
                              * (0 - self.dof_vel[:, i:i + 1])
                torques[:, i:i + 1] = torques_P_p + torques_P_d

            elif control_type == 2:  # Velocity Control (PD controller)
                torques_V_p = 0
                torques_V_d = self.d_gain_scales[:, i:i + 1] * self.d_gains[i] \
                              * (control_target[:, i:i + 1] - self.dof_vel[:, i:i + 1])
                torques[:, i:i + 1] = torques_V_p + torques_V_d

            elif control_type == 3:  # Torque Control
                torques[:, i:i + 1] = control_target[:, i:i + 1]

            else:
                raise NameError(f"Unknown controller type: {control_type}")

        # --------------------------------------------------

        # ratio
        torques *= self.motor_strength_scales

        # clip
        torques = torch.clip(torques, min=-self.dof_tor_limits, max=+self.dof_tor_limits)

        # store torques
        self.torques = torques.clone()

        return torques

    def _reset_states(self, env_ids):
        """
        Resets states of selected environments

        Args:
            env_ids (List[int]): Environment ids
        """
        self._reset_root_states(env_ids)
        self._reset_dof_states(env_ids)

    def _reset_root_states(self, env_ids):
        """
        Resets ROOT states position and velocities of selected environments

            Sets base position to self.base_init_state + self.env_origins
            Sets base orientation to self.base_init_state
            Sets base velocity to zero

        Args:
            env_ids (List[int]): Environment ids
        """
        # base position
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, 0:3] += self.env_origins[env_ids]  # pos: x, y, z

        if self.cfg.domain_rand.randomize_init_base_position_xy:
            add_init_base_position_xy_range = self.cfg.domain_rand.add_init_base_position_xy_range

            # xy position within 1m of the center
            self.root_states[env_ids, 0:2] += torch_rand_float(lower=add_init_base_position_xy_range[0],
                                                               upper=add_init_base_position_xy_range[1],
                                                               shape=(len(env_ids), 2),
                                                               device=self.device)

        # base orientation
        self.root_states[env_ids, 3:7] = self.base_init_state[3:7]  # quat: x, y, z, w
        euler_rpy = torch.zeros(len(env_ids), 3, device=self.device)

        if self.cfg.domain_rand.randomize_init_base_orientation_yaw:
            euler_rpy = torch.zeros(len(env_ids), 3, device=self.device)
            rand_yaw = torch_rand_float(lower=-2.00 * np.pi,
                                        upper=+2.00 * np.pi,
                                        shape=(len(env_ids), 1),
                                        device=self.device)
            euler_rpy[:, 2] = rand_yaw.squeeze(1)

        # ---------------------------------------

        # base velocity
        # [7:10]: lin vel
        self.root_states[env_ids, 7:10] = 0.0

        if self.cfg.domain_rand.randomize_init_base_linear_velocity:
            self.root_states[env_ids, 7:10] = torch_rand_float(lower=-0.5,
                                                               upper=+0.5,
                                                               shape=(len(env_ids), 3),
                                                               device=self.device)

        # [10:13]: ang vel
        self.root_states[env_ids, 10:13] = 0.0

        if self.cfg.domain_rand.randomize_init_base_angular_velocity:
            self.root_states[env_ids, 10:13] = torch_rand_float(lower=-0.5,
                                                                upper=+0.5,
                                                                shape=(len(env_ids), 3),
                                                                device=self.device)

        # ---------------------------------------
        # set to simulation

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Sets actor root state buffer to values provided for given actor indices.
        # Full actor root states buffer should be provided for all actors.
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.all_actors_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32),
                                                     len(env_ids_int32))

    def _reset_dof_states(self, env_ids):
        """
        Resets DOF position and velocities of selected environments

        Positions are randomly selected within
            self.cfg.domain_rand.multiply_init_dof_pos_near_default_range x default positions.

        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environment ids
        """
        dof_pos = self.default_dof_pos \
                  * torch_rand_float(lower=1.0,
                                     upper=1.0,
                                     shape=(len(env_ids), self.num_dofs),
                                     device=self.device)

        """
        randomize_init_dof_pos_near_default: 关节初始位置附近随机化
        """
        # dof_pos randomize
        if self.cfg.domain_rand.randomize_init_dof_pos_near_default:
            # 加法
            if self.cfg.domain_rand.randomize_init_dof_pos_near_default_add:
                add_init_dof_pos_near_default_range = self.cfg.domain_rand.add_init_dof_pos_near_default_range

                dof_pos = self.default_dof_pos \
                          + torch_rand_float(lower=add_init_dof_pos_near_default_range[0],
                                             upper=add_init_dof_pos_near_default_range[1],
                                             shape=(len(env_ids), self.num_dofs),
                                             device=self.device)

            # 乘法
            if self.cfg.domain_rand.randomize_init_dof_pos_near_default_multiply:
                multiply_init_dof_pos_near_default_range = self.cfg.domain_rand.multiply_init_dof_pos_near_default_range

                dof_pos = self.default_dof_pos \
                          * torch_rand_float(lower=multiply_init_dof_pos_near_default_range[0],
                                             upper=multiply_init_dof_pos_near_default_range[1],
                                             shape=(len(env_ids), self.num_dofs),
                                             device=self.device)

        # dof_pos clip
        dof_pos = torch.clip(dof_pos,
                             self.dof_pos_limits[:, 0:1].view(1, -1),
                             self.dof_pos_limits[:, 1:2].view(1, -1))

        # 只对 env_ids 对应的环境 actor 做修改
        self.dof_pos[env_ids] = dof_pos

        # dof_vel randomize
        dof_vel = 0.0

        # 只对 env_ids 对应的环境 actor 做修改
        self.dof_vel[env_ids] = dof_vel

        # ---------------------------------------
        # set to simulation

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Sets actor root state buffer to values provided for given actor indices.
        # Full actor root states buffer should be provided for all actors.
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.all_actors_dof_states),
                                              gymtorch.unwrap_tensor(env_ids_int32),
                                              len(env_ids_int32))

    def _reset_actions(self, env_ids):
        """
        Resets actions of selected environments

        Args:
            env_ids (List[int]): Environment ids
        """
        self.actions[env_ids] = 0.0

    def _reset_others(self, env_ids):
        """
        Resets other states of selected environments

        Args:
            env_ids (List[int]): Environment ids
        """
        self.torques[env_ids] = 0.0

    def _reset_observations(self, env_ids):
        """
        Resets observations of selected environments

        Args:
            env_ids (List[int]): Environment ids
        """
        # actor observations
        self.obs_buf[env_ids] = 0.0
        self.obs_stack[env_ids] = 0.0
        self.obs_stack_num_stacked[env_ids] = 0

        # critic observations
        self.pri_obs_buf[env_ids] = 0.0

    def _drag_robots(self):
        """
        Randomly drags the robots. Emulates a force by setting a randomized base externel force.

        using apply_body_forces to set the base external force

        Output:
        - force: the force applied to the robot
        """
        # Full actor forces and torques buffer should be provided for all actors.
        all_actors_forces = torch.zeros((self.num_all_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        all_actors_torques = torch.zeros((self.num_all_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)

        # only apply force to tosro link, at x/y direction
        max_force = self.cfg.domain_rand.max_drag_force
        all_actors_forces[0: self.num_envs, self.torso_indices, 0] = \
            torch_rand_float(-max_force,
                             +max_force,
                             (self.num_envs, len(self.torso_indices)),
                             device=self.device)
        all_actors_forces[0: self.num_envs, self.torso_indices, 1] = \
            torch_rand_float(-max_force,
                             +max_force,
                             (self.num_envs, len(self.torso_indices)),
                             device=self.device)

        # Applies forces and/or torques to rigid bodies for the immediate timestep, in Newtons.
        self.gym.apply_rigid_body_force_tensors(self.sim,
                                                gymtorch.unwrap_tensor(all_actors_forces),
                                                gymtorch.unwrap_tensor(all_actors_torques),
                                                gymapi.ENV_SPACE)

        return all_actors_forces, all_actors_torques

    def _update_terrain_curriculum(self, env_ids):
        """
        Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return

        move_up, move_up_level, move_down, move_down_level = \
            self._update_terrain_curriculum_detection(env_ids=env_ids)

        self.terrain_levels[env_ids] += move_up_level * move_up - move_down_level * move_down

        # randomize move up terrain level
        if self.cfg.domain_rand.randomize_move_up_terrain_level:
            levels_to_max_terrain_level = self.max_terrain_level * torch.ones(len(env_ids), device=self.device) \
                                          - self.terrain_levels[env_ids]
            move_up_levels = (levels_to_max_terrain_level
                              * torch.rand_like(levels_to_max_terrain_level)).to(dtype=torch.int32) \
                             * move_up

            self.terrain_levels[env_ids] = torch.where(
                torch.rand_like(levels_to_max_terrain_level) < self.cfg.domain_rand.move_up_terrain_level_prob,
                self.terrain_levels[env_ids] + move_up_levels,
                self.terrain_levels[env_ids]
            )

        # robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0)
        )  # (the minimum level is zero)

        self.env_origins[env_ids] = \
            self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _update_terrain_curriculum_detection(self, env_ids):
        """
        Detects if the robot should move up or down in the terrain curriculum
        """
        # 1. detect based on distance from origin
        distance = torch.norm(self.root_states[env_ids, 0:2] - self.env_origins[env_ids, 0:2], dim=1)  # dims 2->1

        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        move_up_level = 1

        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, 0:2], dim=1)
                     * self.max_episode_length_s * 0.5) \
                    * ~move_up
        move_down_level = 1

        return move_up, move_up_level, move_down, move_down_level

    def _update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """

        if (torch.mean(self.episode_sums["cmd_diff_base_lin_vel_x"][env_ids]) / self.max_episode_length) \
                > (0.75 * self.reward_scales["cmd_diff_base_lin_vel_x"]):
            self.curriculum_count_cmd_diff_base_lin_vel_x += 1

            # Only if continuously above 75% for a certain number of episodes
            if self.curriculum_count_cmd_diff_base_lin_vel_x > self.curriculum_count_max_cmd_diff_base_lin_vel_x:
                self.curriculum_count_cmd_diff_base_lin_vel_x = 0

                self.command_ranges["lin_vel_x"][0] = \
                    np.clip(self.command_ranges["lin_vel_x"][0] - self.cfg.commands.curriculum_chg_lin_vel_x,
                            self.cfg.commands.curriculum_min_lin_vel_x,
                            0.0)
                self.command_ranges["lin_vel_x"][1] = \
                    np.clip(self.command_ranges["lin_vel_x"][1] + self.cfg.commands.curriculum_chg_lin_vel_x,
                            0.0,
                            self.cfg.commands.curriculum_max_lin_vel_x)
        else:
            self.curriculum_count_cmd_diff_base_lin_vel_x -= 1

            if self.curriculum_count_cmd_diff_base_lin_vel_x < 0:
                self.curriculum_count_cmd_diff_base_lin_vel_x = 0

        if (torch.mean(self.episode_sums["cmd_diff_base_lin_vel_y"][env_ids]) / self.max_episode_length) \
                > (0.70 * self.reward_scales["cmd_diff_base_lin_vel_y"]):
            self.curriculum_count_cmd_diff_base_lin_vel_y += 1

            # Only if continuously above 75% for a certain number of episodes
            if self.curriculum_count_cmd_diff_base_lin_vel_y > self.curriculum_count_max_cmd_diff_base_lin_vel_y:
                self.curriculum_count_cmd_diff_base_lin_vel_y = 0

                self.command_ranges["lin_vel_y"][0] = \
                    np.clip(self.command_ranges["lin_vel_y"][0] - self.cfg.commands.curriculum_chg_lin_vel_y,
                            self.cfg.commands.curriculum_min_lin_vel_y,
                            0.0)
                self.command_ranges["lin_vel_y"][1] = \
                    np.clip(self.command_ranges["lin_vel_y"][1] + self.cfg.commands.curriculum_chg_lin_vel_y,
                            0.0,
                            self.cfg.commands.curriculum_max_lin_vel_y)
        else:
            self.curriculum_count_cmd_diff_base_lin_vel_y -= 1

            if self.curriculum_count_cmd_diff_base_lin_vel_y < 0:
                self.curriculum_count_cmd_diff_base_lin_vel_y = 0

        if (torch.mean(self.episode_sums["cmd_diff_base_ang_vel_yaw"][env_ids]) / self.max_episode_length) \
                > (0.65 * self.reward_scales["cmd_diff_base_ang_vel_yaw"]):
            self.curriculum_count_cmd_diff_base_ang_vel_yaw += 1

            # Only if continuously above 75% for a certain number of episodes
            if self.curriculum_count_cmd_diff_base_ang_vel_yaw > self.curriculum_count_max_cmd_diff_base_ang_vel_yaw:
                self.curriculum_count_cmd_diff_base_ang_vel_yaw = 0

                self.command_ranges["ang_vel_yaw"][0] = \
                    np.clip(self.command_ranges["ang_vel_yaw"][0] - self.cfg.commands.curriculum_chg_ang_vel_yaw,
                            self.cfg.commands.curriculum_min_ang_vel_yaw,
                            0.0)
                self.command_ranges["ang_vel_yaw"][1] = \
                    np.clip(self.command_ranges["ang_vel_yaw"][1] + self.cfg.commands.curriculum_chg_ang_vel_yaw,
                            0.0,
                            self.cfg.commands.curriculum_max_ang_vel_yaw)
        else:
            self.curriculum_count_cmd_diff_base_ang_vel_yaw -= 1

            if self.curriculum_count_cmd_diff_base_ang_vel_yaw < 0:
                self.curriculum_count_cmd_diff_base_ang_vel_yaw = 0

    # ----------------------------------------

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {}
        for name in self.reward_scales.keys():
            self.episode_sums[name] = torch.zeros(self.num_envs,
                                                  dtype=torch.float, device=self.device, requires_grad=False)

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution

        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.cfg.terrain.horizontal_scale
        hf_params.row_scale = self.cfg.terrain.horizontal_scale
        hf_params.vertical_scale = self.cfg.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows

        hf_params.transform.p.x = -self.cfg.terrain.border_size
        hf_params.transform.p.y = -self.cfg.terrain.border_size
        hf_params.transform.p.z = 0.0

        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim,
                                 self.terrain.heightsamples,
                                 hf_params)
        self.height_samples = \
            torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                          self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.cfg.terrain.border_size
        tm_params.transform.p.y = -self.cfg.terrain.border_size
        tm_params.transform.p.z = 0.0

        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution

        self.gym.add_triangle_mesh(self.sim,
                                   self.terrain.vertices.flatten(order="C"),
                                   self.terrain.triangles.flatten(order="C"),
                                   tm_params)
        self.height_samples = \
            torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                          self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        print("##############################################")
        print("Creating environments")

        self.robot_asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self.robot_asset_root = os.path.dirname(self.robot_asset_path)
        self.robot_asset_file = os.path.basename(self.robot_asset_path)

        print("self.robot_asset_path:", self.robot_asset_path)
        print("self.robot_asset_root:", self.robot_asset_root)
        print("self.robot_asset_file:", self.robot_asset_file)
        print("")

        self.robot_asset_options = gymapi.AssetOptions()
        self.robot_asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        self.robot_asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        self.robot_asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        self.robot_asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        self.robot_asset_options.fix_base_link = self.cfg.asset.fix_base_link
        self.robot_asset_options.density = self.cfg.asset.density
        self.robot_asset_options.angular_damping = self.cfg.asset.angular_damping
        self.robot_asset_options.linear_damping = self.cfg.asset.linear_damping
        self.robot_asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        self.robot_asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        self.robot_asset_options.armature = self.cfg.asset.armature
        self.robot_asset_options.thickness = self.cfg.asset.thickness
        self.robot_asset_options.disable_gravity = self.cfg.asset.disable_gravity

        print("self.robot_asset_options: \n", self.robot_asset_options)
        print("asset_options.collapse_fixed_joints: \n", self.robot_asset_options.collapse_fixed_joints)
        print("")

        self.robot_asset = self.gym.load_asset(self.sim,
                                               self.robot_asset_root,
                                               self.robot_asset_file,
                                               self.robot_asset_options)

        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        self.num_dofs = self.gym.get_asset_dof_count(self.robot_asset)

        print("self.num_bodies:", self.num_bodies)
        print("self.num_dofs:", self.num_dofs)
        print("")

        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)

        # save body names from the asset
        self.body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.num_bodies = len(self.body_names)

        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.num_dofs = len(self.dof_names)

        print("self.num_bodies:", self.num_bodies)
        print("self.body_names:")
        for name in self.body_names:
            print(f"  - {name}")
        print("")

        print("self.num_dofs:", self.num_dofs)
        print("self.dof_names:")
        for name in self.dof_names:
            print(f"  - {name}")
        print("")

        self.motor_strength_scales = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gain_scales = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gain_scales = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)

        # set up the initial state of the robot
        base_init_state_list = self.cfg.init_state.pos + \
                               self.cfg.init_state.rot + \
                               self.cfg.init_state.lin_vel + \
                               self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)

        print("self.base_init_state: \n", self.base_init_state)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[0:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)

        self.env_handles = []
        self.actor_handles = []

        for env_index in range(self.num_envs):
            # 虚拟环境计数器
            self.num_all_envs += 1

            # ----------------------------------

            # 对刚体表面属性进行随机化处理：摩擦系数、弹性系数
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, env_index)
            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)

            # ----------------------------------

            # create env instance
            env_handle = \
                self.gym.create_env(self.sim,  # Simulation Handle.
                                    env_lower,  # lower bounds of environment space
                                    env_upper,  # upper bounds of environment space
                                    int(np.sqrt(self.num_envs)))  # Number of environments to tile in a row

            # ----------------------------------

            # update the origin of the actor
            pos = self.env_origins[env_index].clone()
            pos[0:2] += torch_rand_float(lower=-1.0,
                                         upper=+1.0,
                                         shape=(2, 1),
                                         device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            # ----------------------------------

            actor_handle = self.gym.create_actor(env_handle,
                                                 self.robot_asset,
                                                 start_pose,
                                                 self.cfg.asset.name,  # name of the actor
                                                 env_index,  # collision group
                                                 self.cfg.asset.self_collisions,  # disable self collision
                                                 0)  # segmentation ID used in segmentation camera sensors

            # ----------------------------------

            if env_index == 0:
                actor_name = self.gym.get_actor_name(env_handle, actor_handle)
                actor_body_names = self.gym.get_actor_rigid_body_names(env_handle, actor_handle)
                actor_body_dict = self.gym.get_actor_rigid_body_dict(env_handle, actor_handle)
                actor_joint_names = self.gym.get_actor_dof_names(env_handle, actor_handle)
                actor_joint_dict = self.gym.get_actor_dof_dict(env_handle, actor_handle)
                actor_dof_names = self.gym.get_actor_dof_names(env_handle, actor_handle)
                actor_dof_dict = self.gym.get_actor_dof_dict(env_handle, actor_handle)

                print("##############################################")
                print("actor_info:")

                print("actor_name:")
                print(str(actor_name))
                print("")

                print("actor_body_names:")
                for name in actor_body_names:
                    print(f"  - {name}")
                print("")

                print("actor_body_dict:")
                for key, value in actor_body_dict.items():
                    print(f"  - {key}: {value}")
                print("")

                print("actor_joint_names:")
                for name in actor_joint_names:
                    print(f"  - {name}")
                print("")

                print("actor_joint_dict:")
                for key, value in actor_joint_dict.items():
                    print(f"  - {key}: {value}")
                print("")

                print("actor_dof_names:")
                for name in actor_dof_names:
                    print(f"  - {name}")
                print("")

                print("actor_dof_dict:")
                for key, value in actor_dof_dict.items():
                    print(f"  - {key}: {value}")
                print("")
                print("##############################################")

            # ----------------------------------

            dof_props = self._process_dof_props(dof_props_asset, env_index)

            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)

            if self.cfg.domain_rand.randomize_motor_strength:
                multiply_motor_strength_range = self.cfg.domain_rand.multiply_motor_strength_range
                self.motor_strength_scales[env_index, :] = torch_rand_float(lower=multiply_motor_strength_range[0],
                                                                            upper=multiply_motor_strength_range[1],
                                                                            shape=(1, self.num_dofs),
                                                                            device=self.device)

            if self.cfg.domain_rand.randomize_motor_stiffness:
                multiply_motor_stiffness_range = self.cfg.domain_rand.multiply_motor_stiffness_range
                self.p_gain_scales[env_index, :] = torch_rand_float(lower=multiply_motor_stiffness_range[0],
                                                                    upper=multiply_motor_stiffness_range[1],
                                                                    shape=(1, self.num_dofs),
                                                                    device=self.device)

            if self.cfg.domain_rand.randomize_motor_damping:
                multiply_motor_damping_range = self.cfg.domain_rand.multiply_motor_damping_range
                self.d_gain_scales[env_index, :] = torch_rand_float(lower=multiply_motor_damping_range[0],
                                                                    upper=multiply_motor_damping_range[1],
                                                                    shape=(1, self.num_dofs),
                                                                    device=self.device)

            # ----------------------------------

            if env_index == 0:
                self._create_envs_get_indices(self.body_names, env_handle, actor_handle)

            # ----------------------------------

            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)

            # 对刚体内在属性进行随机化处理：质量、惯量
            body_props = self._process_rigid_body_props(body_props, env_index)

            # ----------------------------------

            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

            self.env_handles.append(env_handle)
            self.actor_handles.append(actor_handle)

            # ----------------------------------

        # create penalty contacts
        self._create_envs_penalized_contacts(self.body_names)

        # create terminated contacts
        self._create_envs_terminated_contacts(self.body_names)

        print("##############################################")

    def _create_envs_get_indices(self, body_names, env_handle, actor_handle):
        """
        Creates a list of indices for different bodies of the robot.
        """
        torso_name = [s for s in body_names if self.cfg.asset.torso_name in s]
        chest_names = [s for s in body_names if self.cfg.asset.chest_name in s]
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        forehead_indices = [s for s in body_names if self.cfg.asset.forehead_name in s]
        payload_names = [s for s in body_names if self.cfg.asset.payload_name in s]

        self.torso_indices = torch.zeros(len(torso_name), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(torso_name)):
            self.torso_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, torso_name[j])

        self.chest_indices = torch.zeros(len(chest_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(chest_names)):
            self.chest_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, chest_names[j])

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for j in range(len(feet_names)):
            self.feet_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, feet_names[j])

        self.forehead_indices = torch.zeros(len(forehead_indices), dtype=torch.long, device=self.device,
                                            requires_grad=False)
        for j in range(len(forehead_indices)):
            self.forehead_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle,
                                                                             forehead_indices[j])

        self.payload_indices = torch.zeros(len(payload_names), dtype=torch.long, device=self.device,
                                           requires_grad=False)
        for j in range(len(payload_names)):
            self.payload_indices[j] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, payload_names[j])

        print("self.torso_indices: " + str(self.torso_indices))
        print("self.chest_indices: " + str(self.chest_indices))
        print("self.feet_indices: " + str(self.feet_indices))
        print("self.forehead_indices: " + str(self.forehead_indices))
        print("self.payload_indices: " + str(self.payload_indices))

    def _create_envs_penalized_contacts(self, body_names):
        """
        Creates a list of penalized contacts.
        """
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names),
                                                     dtype=torch.long,
                                                     device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.env_handles[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        print("self.penalised_contact_indices: \n", str(self.penalised_contact_indices))

    def _create_envs_terminated_contacts(self, body_names):
        """ Creates a list of terminated contacts.
        """
        termination_contact_names = []
        for name in self.cfg.asset.terminate_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names),
                                                       dtype=torch.long,
                                                       device=self.device,
                                                       requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.env_handles[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])

        print("self.termination_contact_indices: \n", str(self.termination_contact_indices))

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1

            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = \
                torch.div(torch.arange(self.num_envs, device=self.device),
                          (self.num_envs / self.cfg.terrain.num_cols),
                          rounding_mode="floor").to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

        else:
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # clear all lines
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # draw height lines
        if self.cfg.terrain.measure_heights:
            sphere_geom = gymutil.WireframeSphereGeometry(
                radius=0.02,
                num_lats=4,
                num_lons=4,
                pose=None,
                color=(1, 1, 0)
            )

            for i in range(self.num_envs):
                base_pos = (self.root_states[i, :3]).cpu().numpy()
                heights = self.measured_heights[i].cpu().numpy()
                height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                               self.height_points[i]).cpu().numpy()

                for j in range(heights.shape[0]):
                    x = height_points[j, 0] + base_pos[0]
                    y = height_points[j, 1] + base_pos[1]
                    z = heights[j]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.env_handles[i], sphere_pose)

    def _init_height_points(self, points_x=None, points_y=None):
        """ Returns points at which the height measurments are sampled (in base frame)

        Args:
            points_x (List[float], optional): x coordinates of the points. Defaults to None.
            points_y (List[float], optional): y coordinates of the points. Defaults to None.

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
            int: number of height points
        """
        if points_x is None:
            points_x = self.cfg.terrain.measured_points_x

        if points_y is None:
            points_y = self.cfg.terrain.measured_points_y

        x = torch.tensor(points_x, device=self.device, requires_grad=False)
        y = torch.tensor(points_y, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        num_height_points = grid_x.numel()
        height_points = torch.zeros(self.num_envs, num_height_points, 3, device=self.device, requires_grad=False)
        height_points[:, :, 0] = grid_x.flatten()
        height_points[:, :, 1] = grid_y.flatten()

        return height_points, num_height_points

    def _get_heights(self, env_ids=None, height_points=None, num_height_points=None, frame_type="local"):
        """
        Samples heights of the terrain at required points around each robot.
        The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.
            height_points (torch.Tensor, optional): Points at which to measure the height. Defaults to None.
            num_height_points (int, optional): Number of height points. Defaults to None.
            frame_type (str, optional): Frame in which to measure the height. Defaults to "local". Can be "local" or "world".

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, step=1, device=self.device)

        if height_points is None:
            height_points = self.height_points

        if num_height_points is None:
            num_height_points = self.num_height_points

        if self.cfg.terrain.mesh_type == "plane":

            if len(env_ids) >= 0:
                return torch.zeros(len(env_ids), num_height_points, device=self.device, requires_grad=False)

            else:
                raise ValueError("env_ids must be a list of indices or None")

        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        else:
            pass

        if frame_type == "local":  # in local frame

            if len(env_ids) > 0:
                points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, num_height_points),
                                        height_points[env_ids]) \
                         + (self.root_states[env_ids, 0:3]).unsqueeze(1)

            elif len(env_ids) == 0:
                return torch.zeros(0, num_height_points, device=self.device, requires_grad=False)

            else:
                raise ValueError("env_ids must be a list of indices or None")

        elif frame_type == "world":  # in world frame

            if len(env_ids) > 0:
                points = height_points \
                         + (self.root_states[env_ids, 0:3]).unsqueeze(1)

            elif len(env_ids) == 0:
                return torch.zeros(0, num_height_points, device=self.device, requires_grad=False)

            else:
                raise ValueError("env_ids must be a list of indices or None")

        else:
            raise ValueError("frame_type must be 'local' or 'world'")

        points += self.cfg.terrain.border_size
        points = (points / self.cfg.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]

        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        measured_heights = heights.view(len(env_ids), -1) * self.cfg.terrain.vertical_scale

        return measured_heights

    # ==========================================================================================================================
    # Reward functions

    def _reward_termination(self):
        # Terminal reward / penalty
        reward_termination = self.reset_buf * ~self.time_out_buf

        return reward_termination

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        flag_penalty = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1
        reward_collision = torch.sum(flag_penalty, dim=1)

        return reward_collision
