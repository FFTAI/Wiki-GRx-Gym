import torch

from .base_storage import BaseStorage


class RolloutStorage(BaseStorage):
    """Class stores data of all transitions per iteration.
    """

    class Transition(BaseStorage.Transition):
        """Subclass of RolloutStorage, stores quantities of each transition.
        """

        def __init__(self):
            """Initialize Transition members with None.
            """
            super().__init__()

            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

    def __init__(self,
                 num_envs,
                 num_transitions_per_env,
                 actor_obs_shape,
                 critic_obs_shape,
                 actions_shape,
                 device,
                 **kwargs):
        """Initialize RolloutStorage members.

        Args:
            num_envs (int): number of envs
            num_transitions_per_env (int): transitions per env per iteration.
            actor_obs_shape (:obj: `list` of :obj:`int`): tensor shape of observations.
            privileged_obs_shape (:obj:`list` of :obj:`int`): tensor shape of privileged observations.
            actions_shape (:obj:`list` of :obj:`int`): tensor shape of actions.
            device (str, optional): device runs sim and policy. Defaults to 'cpu'.
        """
        if kwargs:
            print("RolloutStorage.__init__ got unexpected arguments, which will be ignored: "
                  + str([key for key in kwargs.keys()]))

        super().__init__(num_envs,
                         num_transitions_per_env,
                         actor_obs_shape,
                         critic_obs_shape,
                         actions_shape,
                         device,
                         **kwargs)

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

    def _add_transition(self, transition: Transition):
        super()._add_transition(transition)

        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """Generator to generate mini batch data from batch data in RolloutStorage.

        Args:
            num_mini_batches (int): number of mini batches
            num_epochs (int, optional): number of epochs. Defaults to 8.

        Yields:
            torch.Tensor: mini batch data of each entry.
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)

        if self.pri_observations is not None:
            critic_observations = self.pri_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                advantages_batch = advantages[batch_idx]
                returns_batch = returns[batch_idx]

                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                yield (obs_batch, critic_observations_batch, actions_batch,
                       target_values_batch, advantages_batch, returns_batch,
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None)
