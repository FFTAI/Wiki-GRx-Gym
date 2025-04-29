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

import numpy

from .base_task_config import BaseConfig


class LeggedRobotCfg(BaseConfig):
    class sim:
        dt = 0.002
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class env:
        num_envs = 4096
        episode_length_s = 20  # episode length in seconds

        num_obs = 1
        # if not None a priviledge_obs_buf will be returned by step()
        # (critic obs for assymetric training).
        # None is returned otherwise.
        num_pri_obs = None
        num_actions = 1

        use_stack = False
        num_stack = 1

        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm

    class terrain:
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh

        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]

        curriculum = True
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = num_rows - 1  # maximum initial terrain level

        static_friction = 0.30  # 0.35  # 0.25
        dynamic_friction = 0.30  # 0.35  # 0.25
        restitution = 0.0  # 0.0: no bounce

        # rough terrain only: 1mx1m rectangle (without center line)
        measure_heights = True
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain

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
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
        ]

        # trimesh only:
        slope_threshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

        terrain_length = 8.0
        terrain_width = 8.0

    class asset:
        file = ""
        name = "legged_robot"  # actor name

        torso_name = "torso"
        chest_name = "chest"
        foot_name = "None"
        forehead_name = "None"
        payload_name = "None"

        terminate_contacts_on = [
            "base"
        ]
        terminate_project_gravity_less_than = 0.33  # ~70 degrees

        penalize_contacts_on = [
        ]
        penalize_project_gravity_less_than = 0.33  # ~70 degrees

        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)

        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation

        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class dof:
        friction = {
            "joint_a": 0.0,
            "joint_b": 0.0,
        }
        armature = {
            "joint_a": 0.0,
            "joint_b": 0.0,
        }
        mechanism = {
            "joint_a": 0.0,
            "joint_b": 0.0,
        }

    class init_state:
        pos = [0.0, 0.0, 1.0]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        default_joint_angles = {
            # target angles when action = 0.0
            "joint_a": 0.0,
            "joint_b": 0.0,
        }

        init_from_key_states = False
        init_from_select_key_states = False
        init_select_key_states_indices = [
        ]
        key_states = [
            numpy.array(
                pos + rot + lin_vel + ang_vel \
                + [
                    default_joint_angles["joint_a"],
                    default_joint_angles["joint_b"]
                ]
            )
        ]

    class commands:
        command_profile = "base_velocity"  # base_velocity: [lin_vel_x, lin_vel_y, ang_vel_yaw]

        curriculum = False
        curriculum_chg_lin_vel_x = 0.50  # additional linear velocity in x direction
        curriculum_chg_lin_vel_y = 0.50  # additional linear velocity in y direction
        curriculum_chg_ang_vel_yaw = 0.50  # additional angular velocity around z axis
        curriculum_min_lin_vel_x = -0.50  # maximum linear velocity in x direction
        curriculum_min_lin_vel_y = -0.50  # maximum linear velocity in y direction
        curriculum_min_ang_vel_yaw = -1.00  # maximum angular velocity around z axis
        curriculum_max_lin_vel_x = +0.50  # maximum linear velocity in x direction
        curriculum_max_lin_vel_y = +0.50  # maximum linear velocity in y direction
        curriculum_max_ang_vel_yaw = +1.00  # maximum angular velocity around z axis

        max_curriculum = 1.

        num_commands = 3  # default: [lin_vel_x, lin_vel_y, ang_vel_yaw]
        resample_command_interval_s = 10.  # -1: not resample, >0: time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        resample_ofc_dof_pos_interval_s = -1  # -1: not resample, >0: resample every x seconds

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-3.14, 3.14]  # min max [rad]

            pos_x = [-2.0, 2.0]  # min max [m]
            pos_y = [-2.0, 2.0]  # min max [m]
            pos_z = [-0.2, 0.2]  # min max [m]

    class control:
        # action sequence: match the dof sequence in the urdf
        action_names = [
            "joint_a",
            "joint_b",
        ]

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = {
            "joint_a": 1.0,
            "joint_b": 1.0,
        }

        # control_type = 'P'  # P: position, V: velocity, T: torques
        # control_type = (0 is None, 1 is Position, 2 is Velocity, 3 Torques)
        control_type = {
            "joint_a": 1,
            "joint_b": 1,
        }

        # PD Drive parameters:
        stiffness = {
            "joint_a": 10.0,
            "joint_b": 15.0,
        }  # [N*m/rad]
        damping = {
            "joint_a": 1.0,
            "joint_b": 1.5,
        }  # [N*m*s/rad]

        dof_pos_offset_scale = {
            "joint_a": 1.0,
            "joint_b": 1.0,
        }

        dof_vel_scale = {
            "joint_a": 1.0,
            "joint_b": 1.0,
        }

        dof_tor_scale = {
            "joint_a": 1.0,
            "joint_b": 1.0,
        }

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class domain_rand:
        # randomize reset episode length
        randomize_reset_episode_length = False

        # randomize friction and restitution
        """
        friction:
        静摩擦力 Coefficient of static friction.
        Value should be equal or greater than zero.

        restitution:
        Coefficient of restitution.
        It’s the ratio of the final to initial velocity after the rigid body collides.
        Range [0, 1]
        """
        randomize_friction = True
        friction_range = [0.30, 1.70]  # [0.35, 1.65]  # [0.25, 1.75]

        randomize_restitution = True
        restitution_range = [0.0, 0.7]  # 0.0: no bounce, 1.0: perfect bounce

        # randomize base mass
        randomize_base_mass = True
        multiply_base_mass_range = [0.9, 1.1]  # unit : kg

        # randomize base center of mass
        randomize_base_com = True
        add_base_com_range_x = [-0.1, 0.1]  # unit : m
        add_base_com_range_y = [-0.1, 0.1]  # unit : m
        add_base_com_range_z = [-0.1, 0.1]  # unit : m

        # randomize base inertia
        randomize_base_inertia = True
        multiply_base_inertia_range = [0.9, 1.1]

        """
        base is removed from the torso indexes 
        """
        # randomize torso mass
        randomize_torso_mass = True
        multiply_torso_mass_range = [0.9, 1.1]  # unit : kg

        # randomize torso center of mass
        randomize_torso_com = True
        add_torso_com_range_x = [-0.1, +0.1]
        add_torso_com_range_y = [-0.1, +0.1]
        add_torso_com_range_z = [-0.1, +0.1]

        # randomize torso inertia
        randomize_torso_inertia = True
        multiply_torso_inertia_range = [0.9, 1.1]

        # randomize payload mass
        randomize_payload_mass = True
        multiply_payload_mass_range = [0.1, 5.0]  # unit : kg

        # randomize payload center of mass
        randomize_payload_com = True
        add_payload_com_range_x = [-0.1, +0.1]
        add_payload_com_range_y = [-0.1, +0.1]
        add_payload_com_range_z = [-0.1, +0.1]

        # randomize payload inertia (I = 1/12 * m * a^2)
        randomize_payload_inertia = True
        multiply_payload_inertia_range = [0.1, 5.0]

        """
        base, torso, payload are removed from the links indexes 
        """
        # randomize link mass
        randomize_link_mass = True
        multiply_link_mass_range = [0.9, 1.1]  # unit : kg

        # randomize link center of mass
        randomize_link_com = True
        add_link_com_range_x_range = [-0.01, +0.01]  # unit : m
        add_link_com_range_y_range = [-0.01, +0.01]  # unit : m
        add_link_com_range_z_range = [-0.01, +0.01]  # unit : m

        # randomize link inertia
        randomize_link_inertia = True
        multiply_link_inertia_range = [0.9, 1.1]

        # randomize motor friction
        randomize_motor_friction = True
        multiply_motor_friction_range = [-1.0, 1.0]

        # randomize mechanism friction (传动机构摩擦)
        randomize_mechanism_friction = True
        multiply_mechanism_friction_range = [0.0, 1.0]

        # randomize motor armature
        randomize_motor_armature = True
        multiply_motor_armature_range = [0.9, 1.1]

        # randomize motor strength
        # 1. 执行器供电电压不同带来的输出力矩不同
        # 2. 执行器摩擦力不同带来的输出力矩不同 (Joint_output = PD_output - friction)
        randomize_motor_strength = True
        multiply_motor_strength_range = [0.9, 1.1]

        # randomize motor stiffness
        randomize_motor_stiffness = True
        multiply_motor_stiffness_range = [0.95, 1.05]

        # randomize motor damping
        randomize_motor_damping = True
        multiply_motor_damping_range = [0.95, 1.05]

        # randomize observations
        randomize_obs_lin_vel = False
        multiply_obs_lin_vel_range = [0.9, 1.1]

        # randomize push robot
        push_robots = False
        push_interval_s = 10.0  # unit: second
        max_push_vel_xy = 0.5  # unit: m/s

        # randomize drag robot
        drag_robots = True
        drag_interval_s = 10.0  # unit: second
        drag_keep_s = 2.50  # unit: second
        max_drag_force = 100.0  # unit: N

        # randomize kick robot
        kick_robots = False
        kick_interval_s = 16.6  # unit: second
        kick_keep_s = 0.50
        max_kick_force = 200.0  # unit: N

        # --------------------------------------------------------------------

        # randomize init base
        randomize_init_base_position_xy = True
        add_init_base_position_xy_range = [-1.0, +1.0]  # unit : m

        randomize_init_base_position_z = False
        multiply_init_base_position_z_range = [0.9, 1.1]

        randomize_init_base_orientation_roll = False
        randomize_init_base_orientation_yaw = True
        randomize_init_base_orientation_pitch = False

        randomize_init_base_linear_velocity = True
        randomize_init_base_angular_velocity = True

        # randomize init dof
        randomize_init_dof_pos_full_range = False

        randomize_init_dof_pos_near_default = True
        randomize_init_dof_pos_near_default_add = False
        add_init_dof_pos_near_default_range = [-0.2, +0.2]  # unit : rad
        randomize_init_dof_pos_near_default_multiply = True
        multiply_init_dof_pos_near_default_range = [0.5, 1.5]

        randomize_init_ofc_dof_pos = False

        randomize_init_dof_pos_near_coach = False
        randomize_init_dof_pos_near_coach_multiply = False
        multiply_init_dof_pos_near_coach_range = [0.8, 1.2]

        randomize_init_dof_pos = False
        multiply_init_dof_pos_range = [0.5, 1.5]

        randomize_init_dof_vel = False
        add_init_dof_vel_range = [-0.5, +0.5]  # unit : rad/s

        # randomize init action
        randomize_init_action = False
        multiply_init_action_range = [0.5, 1.5]

        # --------------------------------------------------------------------

        # randomize control delay
        randomize_control_delay = True
        control_delay_s_range = [0.0, 0.005]  # 系统控制循环延时问题 (状态 -> 算法 -> 指令)

        # randomize terrain level placement when meet "move_up"
        # upgrade the terrain level randomly may accelerate the learning process
        randomize_move_up_terrain_level = True
        move_up_terrain_level_prob = 0.1  # only select 10% of the actors to move up level randomly

    class rewards:
        robot_mass = 1.0  # kg

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)

        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_dof_tor_limit = 1.
        soft_dof_pwr_limit = 1.

        soft_sum_dof_pwr_limit = 1.
        sum_dof_pwr_limit = 1000.  # sum of all joint powers (W)

        base_height_target = 1.
        max_contact_force = 100.  # forces above this value are penalized

        close_distance = 0.5  # distance to target to consider it reached nearby
        too_high_velocity = 10.0  # velocity above this value is penalized
        too_low_velocity = 0.1  # velocity below this value is penalized

        class scales:
            termination = -0.0

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            """
            Jason 2024-11-17:
            全部为基本国际单位制

            base_pos: 位置 m
            base_ang: 角度 rad
            action: 动作 rad
            lin_vel: 线速度 m/s
            ang_vel: 角速度 rad/s
            gravity: 重力 m/s^2
            dof_pos: 关节位置 rad
            dof_vel: 关节速度 rad/s
            height_measurements: 高度 m
            """
            base_pos = 0.01
            base_ang = 0.01
            action = 0.00
            lin_vel = 0.10
            ang_vel = 0.20
            gravity = 0.05
            dof_pos = 0.01
            dof_vel = 1.50
            height_measurements = 0.02

    class normalization:
        class obs_scales:
            """
            Jason 2024-11-17:
            全部为基本国际单位制

            base_pos: 位置 m
            base_ang: 角度 rad
            action: 动作 rad
            lin_vel: 线速度 m/s
            ang_vel: 角速度 rad/s
            gravity: 重力 m/s^2
            dof_pos: 关节位置 rad
            dof_vel: 关节速度 rad/s
            height_measurements: 高度 m
            """
            base_pos = 1.0
            base_ang = 1.0
            action = 1.0
            lin_vel = 2.0
            ang_vel = 0.25
            gravity = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        clip_observations = 100.
        clip_actions = 100.

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [-3.0, -3.0, 3.0]  # [m]
        lookat = [2.0, 2.0, 1.0]  # [m]


class LeggedRobotCfgPPO(BaseConfig):
    seed = -1  # random seed
    runner_class_name = "OnPolicyRunner"

    class runner:
        num_steps_per_env = 24  # per iteration
        max_iterations = 1500  # number of policy updates
        init_at_random_ep_len = True  # if true, the first episode length is random

        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''

        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt

    class algorithm:
        class_name = "PPO"

        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class policy:
        class_name = 'ActorCriticMLP'

        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        init_weights = False

        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

        """
        Jason 2025-01-14:
        As different joint may have different control mode and Kp, Kd, 
        so we need to set their init_noise_std separately.
        """
        init_noise_std = [
            1.0,  # joint_a
            1.0,  # joint_b
        ]
