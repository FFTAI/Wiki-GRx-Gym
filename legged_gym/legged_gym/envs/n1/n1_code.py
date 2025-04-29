from isaacgym.torch_utils import *

from legged_gym.envs.fftai.legged_robot_fftai_bipedal_code import LeggedRobotFFTAIBipedal
from legged_gym.envs.n1.n1_config import N1Cfg


class N1(LeggedRobotFFTAIBipedal):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg: N1Cfg = cfg

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

    def compute_observation_profile(self):

        obs_buf = torch.cat(
            (
                # command
                self.commands[:, 0:3] * self.commands_scale,

                # base related
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.base_projected_gravity * self.obs_scales.gravity,

                # dof related
                self.dof_pos_offset * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,

                # action related
                self.actions * self.obs_scales.action,
            ), dim=-1)

        pri_obs_buf = torch.cat(
            (
                obs_buf,

                # base related
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_heights_offset * self.obs_scales.height_measurements,

                # foot related
                self.feet_contact,
                self.feet_height * self.obs_scales.height_measurements,
                self.avg_feet_speed_xyz[:, 0, 0:1] * self.obs_scales.lin_vel,
                self.avg_feet_speed_xyz[:, 1, 0:1] * self.obs_scales.lin_vel,
                self.avg_feet_speed_xyz[:, 0, 1:2] * self.obs_scales.lin_vel,
                self.avg_feet_speed_xyz[:, 1, 1:2] * self.obs_scales.lin_vel,

                # terrain related
                self.surround_heights_offset * self.obs_scales.height_measurements,
            ), dim=-1)

        self.obs_buf = obs_buf
        self.pri_obs_buf = pri_obs_buf

    def compute_obs_noise_scale_vec_profile(self):
        """
        Returns the noise scale vector for the observation vector.

        The noise scale vector is used to scale the noise vector to the same scale as the observation vector.

        Output:
        - noise_scale_vec: torch.Tensor
        """

        noise_vec = torch.zeros_like(self.obs_buf[0])

        # command
        start_index_of_commands = 0
        index_offset_of_commands = self.cfg.commands.num_commands
        noise_vec[start_index_of_commands + 0:
                  start_index_of_commands + self.cfg.commands.num_commands] = 0.  # commands x, y, yaw

        # base related
        start_index_of_base_related = start_index_of_commands + index_offset_of_commands
        index_offset_of_base_related = 6
        noise_vec[start_index_of_base_related + 0:
                  start_index_of_base_related + 3] = \
            self.noise_scales.ang_vel \
            * self.noise_level \
            * self.obs_scales.ang_vel  # base ang vel
        noise_vec[start_index_of_base_related + 3:
                  start_index_of_base_related + 6] = \
            self.noise_scales.gravity \
            * self.noise_level \
            * self.obs_scales.gravity  # base projected gravity

        # dof related
        start_index_of_dof_related = start_index_of_base_related + index_offset_of_base_related
        index_offset_of_dof_related = 2 * self.num_dofs
        noise_vec[start_index_of_dof_related + 0 * self.num_dofs:
                  start_index_of_dof_related + 1 * self.num_dofs] = \
            self.noise_scales.dof_pos \
            * self.noise_level \
            * self.obs_scales.dof_pos  # dof_pos_offset
        noise_vec[start_index_of_dof_related + 1 * self.num_dofs:
                  start_index_of_dof_related + 2 * self.num_dofs] = \
            self.noise_scales.dof_vel \
            * self.noise_level \
            * self.obs_scales.dof_vel  # dof_vel

        # action related
        start_index_of_action_related = start_index_of_dof_related + index_offset_of_dof_related
        index_offset_of_action_related = 1 * self.num_actions
        noise_vec[start_index_of_action_related + 0 * self.num_actions:
                  start_index_of_action_related + 1 * self.num_actions] = \
            self.noise_scales.action \
            * self.noise_level \
            * self.obs_scales.action  # actions

        return noise_vec

    # ----------------------------------------------

    def _resample_commands(self, env_ids=None, command_profile=None):
        super()._resample_commands(env_ids, command_profile)

        self.update_gait_generator_pattern()

    def set_commands(self, env_ids, commands):
        """
        Sets the commands for the specified environments.

        NOTE: should not be called in the training process!!!
        """
        self.commands[env_ids] = commands

        self._command_refinement(env_ids)
        self.update_gait_generator_pattern()

    def update_flags_of_stand_command(self):
        # situation 1: the norm of the command x, y lin_vel is less than 0.10
        flags_of_stand_command_s1 = torch.norm(self.commands[:, 0:2], dim=1) <= 0.10

        # situation 2: the value of the command yaw ang_vel is less than 0.10
        flags_of_stand_command_s2 = torch.abs(self.commands[:, 2]) <= 0.10

        # situation 1 and situation 2
        flags_of_stand_command = flags_of_stand_command_s1 * flags_of_stand_command_s2
        flags_of_stand_command = flags_of_stand_command.bool()

        return flags_of_stand_command

    def update_flags_of_walk_command(self):
        # situation 1: the norm of the command x, y lin_vel is greater than 0.10
        flags_of_walk_command_s1 = torch.norm(self.commands[:, 0:2], dim=1) > 0.10

        # situation 2: the value of the command yaw ang_vel is greater than 0.10
        flags_of_walk_command_s2 = torch.abs(self.commands[:, 2]) > 0.10

        # situation 1 and situation 2
        flags_of_walk_command = flags_of_walk_command_s1 + flags_of_walk_command_s2
        flags_of_walk_command = flags_of_walk_command.bool()

        return flags_of_walk_command

    def update_gait_generator_pattern(self):
        env_ids_of_off_command = torch.Tensor([]).int().to(self.device)
        env_ids_of_stand_command = torch.Tensor([]).int().to(self.device)
        env_ids_of_walk_command = torch.Tensor([]).int().to(self.device)

        if "stand" in self.cfg.commands.gait_patterns:
            env_ids_of_stand_command = torch.where(self.update_flags_of_stand_command())[0]

        if "walk" in self.cfg.commands.gait_patterns:
            env_ids_of_walk_command = torch.where(self.update_flags_of_walk_command())[0]

        # update the env_ids of different gait patterns
        self.env_ids_of_off_command = env_ids_of_off_command
        self.env_ids_of_stand_command = env_ids_of_stand_command
        self.env_ids_of_walk_command = env_ids_of_walk_command

    # ==========================================================================================================================
    # Reward functions
