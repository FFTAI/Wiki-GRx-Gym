import numpy

from legged_gym.envs.fftai.legged_robot_fftai_config import (
    LeggedRobotFFTAICfg,
    LeggedRobotFFTAICfgPPO,
)


class GR1T1Cfg(LeggedRobotFFTAICfg):
    class sim(LeggedRobotFFTAICfg.sim):
        dt = 0.002  # simulation time step [s]

    class env(LeggedRobotFFTAICfg.env):
        num_envs = 8192  # NVIDIA 4090 has 16384 CUDA cores
        episode_length_s = 20  # episode length in seconds

        num_obs = 121
        num_actions = 32

    class terrain(LeggedRobotFFTAICfg.terrain):
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh

    class asset(LeggedRobotFFTAICfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/GR1T1/urdf/GR1T1.urdf'
        name = "GR1T1"

        # for both joint and link name
        torso_name = 'torso'  # humanoid pelvis part
        forehead_name = 'head_pitch'  # humanoid head part

        # imu
        imu_name = 'imu'

        # waist
        waist_name = 'waist'
        waist_yaw_name = 'waist_yaw'
        waist_roll_name = 'waist_roll'
        waist_pitch_name = 'waist_pitch'

        # head
        head_name = 'head'
        head_roll_name = 'head_roll'
        head_pitch_name = 'head_pitch'

        # for link name
        thigh_name = 'thigh'
        shank_name = 'shank'
        foot_name = 'foot_roll'
        sole_name = 'sole'
        upper_arm_name = 'upper_arm'
        lower_arm_name = 'lower_arm'
        hand_name = 'hand'

        # for joint name
        hip_name = 'hip'
        hip_roll_name = 'hip_roll'
        hip_yaw_name = 'hip_yaw'
        hip_pitch_name = 'hip_pitch'
        knee_name = 'knee'
        ankle_name = 'ankle'
        ankle_pitch_name = 'ankle_pitch'
        ankle_roll_name = 'ankle_roll'
        shoulder_name = 'shoulder'
        shoulder_pitch_name = 'shoulder_pitch'
        shoulder_roll_name = 'shoulder_roll'
        shoulder_yaw_name = 'shoulder_yaw'
        elbow_name = 'elbow'
        wrist_name = 'wrist'
        wrist_yaw_name = 'wrist_yaw'
        wrist_roll_name = 'wrist_roll'
        wrist_pitch_name = 'wrist_pitch'

        # for arm reaching
        arm_base_name = 'arm_base'
        arm_end_name = 'arm_end'

        penalize_contacts_on = []
        terminate_after_contacts_on = [
            imu_name,
            torso_name, forehead_name, waist_name,
            upper_arm_name, lower_arm_name, hand_name,

            # FIXME: add this link will cause continuous reset
            # thigh_name,
        ]

    class init_state(LeggedRobotFFTAICfg.init_state):
        pos = [0.0, 0.0, 0.95]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            # left leg
            'left_hip_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0,
            'left_hip_pitch_joint': -(numpy.deg2rad(15)),
            'left_knee_pitch_joint': (numpy.deg2rad(30)),
            'left_ankle_pitch_joint': -(numpy.deg2rad(15)),
            'left_ankle_roll_joint': 0.0,

            # right leg
            'right_hip_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'right_hip_pitch_joint': -(numpy.deg2rad(15)),
            'right_knee_pitch_joint': (numpy.deg2rad(30)),
            'right_ankle_pitch_joint': -(numpy.deg2rad(15)),
            'right_ankle_roll_joint': 0.0,

            # waist
            'waist_yaw_joint': 0.0,
            'waist_pitch_joint': 0.0,
            'waist_roll_joint': 0.0,

            # head
            'head_yaw_joint': 0.0,
            'head_pitch_joint': 0.0,
            'head_roll_joint': 0.0,

            # left arm
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.2,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_pitch_joint': -0.3,
            'left_wrist_yaw_joint': 0.0,
            'left_wrist_roll_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,

            # right arm
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': -0.2,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_pitch_joint': -0.3,
            'right_wrist_yaw_joint': 0.0,
            'right_wrist_roll_joint': 0.0,
            'right_wrist_pitch_joint': 0.0
        }

    class commands(LeggedRobotFFTAICfg.commands):
        curriculum = False
        curriculum_chg_lin_vel_x = 0.25  # additional linear velocity in x direction
        curriculum_chg_lin_vel_y = 0.25  # additional linear velocity in y direction
        curriculum_chg_ang_vel_yaw = 0.25  # additional angular velocity around z axis
        curriculum_max_lin_vel_x = 1.00  # maximum linear velocity in x direction
        curriculum_max_lin_vel_y = 0.50  # maximum linear velocity in y direction
        curriculum_max_ang_vel_yaw = 1.00  # maximum angular velocity around z axis

        num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_command_interval_s = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges(LeggedRobotFFTAICfg.commands.ranges):
            lin_vel_x = [-1.00, 1.00]  # min max [m/s]
            lin_vel_y = [-0.50, 0.50]  # min max [m/s]
            ang_vel_yaw = [-1.00, 1.00]  # min max [rad/s]

    class control(LeggedRobotFFTAICfg.control):
        # PD Drive parameters:
        stiffness = {
            'hip_roll': 251.625, 'hip_yaw': 362.5214, 'hip_pitch': 200,
            'knee_pitch': 200,
            'ankle_pitch': 10.9805, 'ankle_roll': 0.25,  # 'ankleRoll': 0.0,
            'waist_yaw': 362.5214, 'waist_pitch': 362.5214, 'waist_roll': 362.5214,
            'head_yaw': 10.0, 'head_pitch': 10.0, 'head_roll': 10.0,
            'shoulder_pitch': 92.85, 'shoulder_roll': 92.85, 'shoulder_yaw': 112.06,
            'elbow_pitch': 112.06,
            'wrist_yaw': 10.0, 'wrist_roll': 10.0, 'wrist_pitch': 10.0
        }  # [N*m/rad]
        damping = {
            'hip_roll': 14.72, 'hip_yaw': 10.0833, 'hip_pitch': 11,
            'knee_pitch': 11,
            'ankle_pitch': 0.5991, 'ankle_roll': 0.01,
            'waist_yaw': 10.0833, 'waist_pitch': 10.0833, 'waist_roll': 10.0833,
            'head_yaw': 1.0, 'head_pitch': 1.0, 'head_roll': 1.0,
            'shoulder_pitch': 2.575, 'shoulder_roll': 2.575, 'shoulder_yaw': 3.1,
            'elbow_pitch': 3.1,
            'wrist_yaw': 1.0, 'wrist_roll': 1.0, 'wrist_pitch': 1.0
        }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1.0

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class rewards(LeggedRobotFFTAICfg.rewards):
        only_positive_rewards = False

        base_height_target = 0.85  # 期望的机器人身体高度
        swing_feet_height_target = 0.10  # 期望的脚抬高度

        feet_stumble_ratio = 5.0  # ratio = fxy / fz

        # ---------------------------------------------------------------

        feet_air_time_target = 0.5  # 期望的脚空中时间
        feet_land_time_max = 1.0  # 最大的脚着地时间

        tracking_sigma = 1.0  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.95
        max_contact_force = 500.

        # ---------------------------------------------------------------

        sigma_collision = -1.0
        sigma_stand_still = -1.0

        sigma_cmd_diff_lin_vel_x = -3.0
        sigma_cmd_diff_lin_vel_y = -3.0
        sigma_cmd_diff_lin_vel_z = -3.0
        sigma_cmd_diff_ang_vel_roll = -3.0
        sigma_cmd_diff_ang_vel_pitch = -3.0
        sigma_cmd_diff_ang_vel_yaw = -3.0
        sigma_cmd_diff_base_height = -50.0

        sigma_cmd_diff_base_orient = -20.0
        sigma_cmd_diff_torso_orient = -20.0
        sigma_cmd_diff_forehead_orient = -20.0

        sigma_action_diff = -0.1
        sigma_action_diff_knee = -1.0

        sigma_dof_vel_new = -0.01
        sigma_dof_vel_new_knee = -0.05

        sigma_dof_acc_new = -0.00005
        sigma_dof_tor_new = -0.00005
        sigma_dof_tor_new_hip_roll = -0.002

        sigma_dof_tor_ankle_feet_lift_up = -1.0

        sigma_pose_offset = -0.1
        sigma_pose_offset_hip_yaw = -0.1

        sigma_limits_dof_pos = -1.0
        sigma_limits_dof_vel = -10.0
        sigma_limits_dof_tor = -0.1

        sigma_feet_speed_xy_close_to_ground = -10.0
        sigma_feet_speed_z_close_to_height_target = -10.0

        sigma_feet_air_time = -1.0
        sigma_feet_air_time_mid = -10.0
        sigma_feet_air_height = -200.0
        sigma_feet_air_force = -0.05
        sigma_feet_land_time = -1.0

        sigma_on_the_air = -1.0

        sigma_feet_stumble = -1.0

        # sigma ---------------------------------------------------------------

        class scales(LeggedRobotFFTAICfg.rewards.scales):
            termination = 0.0

    class noise(LeggedRobotFFTAICfg.noise):
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales(LeggedRobotFFTAICfg.noise.noise_scales):
            action = 0.00  # rad
            lin_vel = 0.10  # m/s
            ang_vel = 0.05  # rad/s
            gravity = 0.03  # m/s^2
            dof_pos = 0.04  # rad
            dof_vel = 0.20  # rad/s
            height_measurements = 0.05  # m

    class normalization(LeggedRobotFFTAICfg.normalization):
        class obs_scales(LeggedRobotFFTAICfg.normalization.obs_scales):
            action = 1.0
            lin_vel = 1.0
            ang_vel = 1.0
            gravity = 1.0
            dof_pos = 1.0
            dof_vel = 1.0
            height_measurements = 5.0

        actions_max = numpy.array([
            0.79, 0.7, 0.7, 1.92, 0.52, 0.44,  # left leg
            0.09, 0.7, 0.7, 1.92, 0.52, 0.44,  # right leg
            1.05, 1.22, 0.7,  # waist
            2.71, 0.35, 0.35,  # head
            1.92, 3.27, 2.97, 2.27, 2.97, 0.61, 0.61,  # left arm
            1.92, 0.57, 2.97, 2.27, 2.97, 0.61, 0.61,  # right arm
        ])
        actions_min = numpy.array([
            -0.09, -0.7, -1.75, -0.09, -1.05, -0.44,  # left leg
            -0.79, -0.7, -1.75, -0.09, -1.05, -0.44,  # right leg
            -1.05, -0.52, -0.7,  # waist
            -2.71, -0.35, -0.52,  # head
            -2.79, -0.57, -2.97, -2.27, -2.97, -0.61, -0.61,  # left arm
            -2.79, -3.27, -2.97, -2.27, -2.97, -0.61, -0.61,  # right arm
        ])

        clip_observations = 100.0
        clip_actions_max = \
            actions_max \
            + (numpy.abs(actions_max) + numpy.abs(actions_min)) * 0.01
        clip_actions_min = \
            actions_min \
            - (numpy.abs(actions_max) + numpy.abs(actions_min)) * 0.01


class GR1T1CfgPPO(LeggedRobotFFTAICfgPPO, GR1T1Cfg):
    runner_class_name = 'OnPolicyRunner'

    class runner(LeggedRobotFFTAICfgPPO.runner):
        algorithm_class_name = 'PPO'
        policy_class_name = 'ActorCriticMLP'

        experiment_name = 'GR1T1'
        num_steps_per_env = 64

        run_name = 'gr1t1'
        max_iterations = 2000
        save_interval = 100

    class algorithm(LeggedRobotFFTAICfgPPO.algorithm):
        # training params
        num_learning_epochs = 8
        num_mini_batches = 25  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-4
        learning_rate_min = 1.e-5
        learning_rate_max = 1.e-3
        schedule = 'adaptive'  # could be adaptive, fixed
        desired_kl = 0.01

        # storage class
        storage_class = "RolloutStorage"

    class policy(LeggedRobotFFTAICfgPPO.policy):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        actor_output_activation = None  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        critic_output_activation = None  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        fixed_std = False
        init_noise_std = 0.2
