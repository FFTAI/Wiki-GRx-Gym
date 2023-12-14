from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Gr1t1Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096  # 4096, number of instances trained in parallel
        num_actions = 32 # number of actuators on robot

        # IMPORTANT, num_observations should match the dimension of following items (subject to robot configurations) that included in observation instance
        # | `base_lin_vel`              | 3    |
		# | `base_ang_vel`              | 3    |
		# | `projected_gravity`         | 3    |
		# | `commands`                  | 3(4) |
		# | `dof_pos`                   | 32   |
		# | `dof_vel`                   | 32   |
        # | `action`                    | 32   |
        # | `num_measured_heights`      | 11*11|    meshgrid defined in `terrian` if `measure_heights` is `True` 
        num_observations = 108 + 11 * 11
         
        episode_length_s = 20 # episode length in seconds

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh

        measure_heights = True
        # 1mx1m rectangle (without center line)
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    class init_state(LeggedRobotCfg.init_state):
        # pos = [0.0 0.0 0.95]
        pos = [0.0, 0.0, 0.95] # x, y, z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            # left leg
            'l_hip_roll': 0.0,
            'l_hip_yaw': 0.,
            'l_hip_pitch': -0.5236,
            'l_knee_pitch': 1.0472,
            'l_ankle_pitch': -0.5236,
            'l_ankle_roll': 0.0,

            # right leg
            'r_hip_roll': -0.,
            'r_hip_yaw': 0.,
            'r_hip_pitch': -0.5236,
            'r_knee_pitch': 1.0472,
            'r_ankle_pitch': -0.5236,
            'r_ankle_roll': 0.0,

            # waist
            'waist_yaw': 0.0,
            'waist_pitch': 0.0,
            'waist_roll': 0.0,

            # head
            'head_yaw': 0.0,
            'head_pitch': 0.0,
            'head_roll': 0.0,

            # left arm
            'l_shoulder_pitch': 0.0,
            'l_shoulder_roll': 0.2,
            'l_shoulder_yaw': 0.0,
            'l_elbow_pitch': -0.3,
            'l_wrist_yaw': 0.0,
            'l_wrist_roll': 0.0,
            'l_wrist_pitch': 0.0,

            # right arm
            'r_shoulder_pitch': 0.0,
            'r_shoulder_roll': -0.2,
            'r_shoulder_yaw': 0.0,
            'r_elbow_pitch': -0.3,
            'r_wrist_yaw': 0.0,
            'r_wrist_roll': 0.0,
            'r_wrist_pitch': 0.0
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {
            'hip_roll': 251.625, 'hip_yaw': 453.15, 'hip_pitch': 285.8131,
            'knee_pitch': 285.8131,
            'ankle_pitch': 21.961, 'ankle_roll': 2.0761,  # 'ankleRoll': 0.0,
            'waist_yaw': 453.15, 'waist_pitch': 453.15, 'waist_roll': 453.15,
            'head_yaw': 10.0, 'head_pitch': 10.0, 'head_roll': 10.0,
            'shoulder_pitch': 92.85, 'shoulder_roll': 92.85, 'shoulder_yaw': 112.06,
            'elbow_pitch': 112.06,
            'wrist_yaw': 10.0, 'wrist_roll': 10.0, 'wrist_pitch': 10.0
        }  # [N*m/rad]
        damping = {
            'hip_roll': 14.72, 'hip_yaw': 50.4164, 'hip_pitch': 16.5792,
            'knee_pitch': 16.5792,
            'ankle_pitch': 1.195, 'ankle_roll': 0.1233,
            'waist_yaw': 50.4164, 'waist_pitch': 50.4164, 'waist_roll': 50.4164,
            'head_yaw': 1.0, 'head_pitch': 1.0, 'head_roll': 1.0,
            'shoulder_pitch': 2.575, 'shoulder_roll': 2.575, 'shoulder_yaw': 3.1,
            'elbow_pitch': 3.1,
            'wrist_yaw': 1.0, 'wrist_roll': 1.0, 'wrist_pitch': 1.0
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/gr1t1/urdf/GR1T1.urdf'
        name = 'gr1t1'

        # for both joint and link name
        torso_name = 'base'  # humanoid pelvis part
        chest_name = 'waist_roll'  # humanoid chest part
        forehead_name = 'head_pitch'  # humanoid head part

        waist_name = 'waist'
        waist_roll_name = 'waist_roll'
        waist_pitch_name = 'waist_pitch'
        head_name = 'head'
        head_roll_name = 'head_roll'
        head_pitch_name = 'head_pitch'

        # for link name
        thigh_name = 'thigh'
        shank_name = 'shank'
        foot_name = 'foot_roll'  # foot_pitch is not used
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
        shoulder_name = 'shoulder'
        shoulder_pitch_name = 'shoulder_pitch'
        shoulder_roll_name = 'shoulder_roll'
        shoulder_yaw_name = 'shoulder_yaw'
        elbow_name = 'elbow'
        wrist_name = 'wrist'
        wrist_yaw_name = 'wrist_yaw'
        wrist_roll_name = 'wrist_roll'
        wrist_pitch_name = 'wrist_pitch'

        terminate_after_contacts_on = ['waist', 'thigh', 'shoulder', 'elbow', 'hand']
        terminate_after_base_projected_gravity_greater_than = 1.0  # [m/s^2]
        terminate_after_base_lin_vel_greater_than = None  # [m/s]
        terminate_after_base_ang_vel_greater_than = None  # [rad/s]

        flip_visual_attachments = False
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class commands(LeggedRobotCfg.commands):
        curriculum = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.3, 1.0]
        restitution_range = [0.0, 1.0]

        # randomize mass
        randomize_base_mass = False
        randomize_thigh_mass = False
        randomize_shank_mass = False
        randomize_torso_mass = False
        randomize_upper_arm_mass = False
        randomize_lower_arm_mass = False
        added_mass_range = [-0.05, 0.05]

        # randomize inertias
        randomize_base_com = False
        added_com_range_x = [-0.05, 0.05]
        added_com_range_y = [-0.02, 0.02]
        added_com_range_z = [-0.05, 0.05]

        # randomize motor strength
        randomize_motor_strength = False
        motor_strength = [0.7, 1.4]

        # randomize external forces
        push_robots = True
        push_interval_s = 3.5
        max_push_vel_xy = 0.5

        apply_forces = False
        continue_time_s = 0.5
        max_ex_forces = [-200.0, 200.0]
        max_ex_torques = [-0.0, 0.0]

        # randomize observations
        randomize_obs_lin_vel = False
        obs_lin_vel = [0.8, 1.2]

    class rewards(LeggedRobotCfg.rewards):

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.8 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.8

        feet_height_target = 0.2
        swing_height_offset_target = 0.05
        swing_feet_distance_offset_target = [0.00, 0.25]
        max_contact_force = 500. # forces above this value are penalized

        class scales(LeggedRobotCfg.rewards.scales):
            termination = 0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001  # -30.e-6  # -25.e-6
            dof_vel = -0.    # -0.01  # -0.001
            dof_acc = -2.5e-7  # -1.e-6  # -1.e-7
            base_height = -0. 
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.


    # class normalization(LeggedRobotCfg.normalization):
    #     class obs_scales(LeggedRobotCfg.normalization.obs_scales):
    #         lin_vel = 2.0

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 0.1  # scales other values

        # class noise_scales:
        #     dof_pos = 0.01
        #     dof_vel = 1.5
        #     lin_vel = 0.1
        #     ang_vel = 0.2
        #     gravity = 0.05
        #     height_measurements = 0.1

    class sim(LeggedRobotCfg.sim):
        dt = 0.001


class Gr1t1CfgPPO(LeggedRobotCfgPPO):

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'gr1t1-vanilla'
        experiment_name = 'gr1t1'

        # export log files to desired category to keep this repo clean, configurate path in `task_registry`
        log_root = 'synology' 

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # training params
        num_learning_epochs = 8
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches

    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
