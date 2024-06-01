import torch


class BaseStorage:
    """Class stores data of all transitions per iteration.
    """

    class Transition:
        """Subclass of RolloutStorage, stores quantities of each transition.
        """

        def __init__(self):
            """Initialize Transition members with None.
            """
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None  # 用于存储 reward
            self.dones = None
            self.values = None  # 用于存储 critic 的输出

        def clear(self):
            """Resets Transition members to None.
            """
            self.__init__()

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
            obs_shape (:obj: `list` of :obj:`int`): tensor shape of observations.
            privileged_obs_shape (:obj:`list` of :obj:`int`): tensor shape of privileged observations.
            actions_shape (:obj:`list` of :obj:`int`): tensor shape of actions.
            device (str, optional): device runs sim and policy. Defaults to 'cpu'.
        """
        if kwargs:
            print("BaseStorage.__init__ got unexpected arguments, which will be ignored: "
                  + str([key for key in kwargs.keys()]))

        self.device = device

        self.obs_shape = actor_obs_shape
        self.pri_obs_shape = critic_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *actor_obs_shape, device=self.device)

        if critic_obs_shape[0] is not None:
            self.pri_observations = torch.zeros(
                num_transitions_per_env, num_envs, *critic_obs_shape, device=self.device)
        else:
            self.pri_observations = None

        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)  # 用于计算 reward->value 的关联关系
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def _add_transition(self, transition: Transition):
        """Adds transition quantities to RolloutStorage.

        Args:
            transition (Transition): Transition class of transition quantities members.

        Raises:
            AssertionError: transition step should not exceed num_transitions_per_env.
        """
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(transition.observations)

        if self.pri_observations is not None:
            self.pri_observations[self.step].copy_(transition.critic_observations)

        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)

    def add_transitions(self, transition: Transition):
        """Adds transition quantities to RolloutStorage.

        Args:
            transition (Transition): Transition class of transition quantities members.

        Raises:
            AssertionError: transition step should not exceed num_transitions_per_env.
        """
        self._add_transition(transition)

        self.step += 1

    def clear(self):
        """Reset transition step to 0.
        """
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        """Computes returns of trejactories of envs.

        Args:
            last_values (torch.Tensor): last value, output from critic network.
            gamma (float): PPO parameter.
            lam (float): PPO parameter.
        """
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        """Compute statistics of trajectories.

        Returns:
            torch.Tensor: mean of trajectory lengths and rewards.
        """
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

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
        returns = self.returns.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)

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

                yield (obs_batch, critic_observations_batch, actions_batch,
                       target_values_batch, advantages_batch, returns_batch,
                       None, None, None, (None, None), None)
