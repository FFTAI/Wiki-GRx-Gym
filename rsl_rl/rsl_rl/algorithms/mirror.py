from rsl_rl.env import *


class Mirror:
    def __init__(self, env: VecEnv):
        self.env = env

    def get_mirror_obs_batch(
            self,
            obs_batch
    ):
        mirror_obs_batch = self.env.get_mirror_observations(obs_batch)
        return mirror_obs_batch

    def get_mirror_action_batch(
            self,
            action_batch
    ):
        mirror_action_batch = self.env.get_mirror_actions(action_batch)
        return mirror_action_batch
