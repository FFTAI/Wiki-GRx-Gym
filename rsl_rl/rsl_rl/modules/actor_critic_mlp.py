import numpy
import torch
import torch.nn as nn
from torch.distributions import Normal

from .mlp import MLP


class ActorCriticMLP(nn.Module):

    def __init__(self,
                 actor_num_input,
                 critic_num_input,
                 actor_num_output,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_weights=False,
                 fixed_std=False,
                 init_noise_std=0.2,
                 decay_std=False,
                 decay_ratio=0.999,
                 decay_std_min=0.05,
                 **kwargs):
        """Default ActorCriticMLP network

        Args:
            actor_num_input:          input dim to actor network.
            critic_num_input:         input dim to critic network.
            actor_num_output:            output dim of actor network.
            actor_hidden_dims:      dims of hidden layers in actor network.
            critic_hidden_dims:     dims of hidden layers in critic network.
            activation:             activation function name.
            output_activation:      name of output layers' activation function
            init_noise_std:         initial value of ActorCriticMLP.std.
            **kwargs:               Arbitrary keyword arguments.

        Returns:
            nn.Module:              policy of MLPs
        """

        print("----------------------------------")
        print("ActorCriticMLP")

        if kwargs:
            print("ActorCriticMLP.__init__ got unexpected arguments, which will be ignored: "
                  + str([key for key in kwargs.keys()]))

        super(ActorCriticMLP, self).__init__()

        # dimensions
        self.num_actor_input = actor_num_input
        self.num_actor_output = actor_num_output
        self.num_critic_input = critic_num_input
        self.num_critic_output = 1

        # Policy
        self.actor = MLP(input_size=self.num_actor_input,
                         output_size=self.num_actor_output,
                         hidden_dims=actor_hidden_dims,
                         activation=activation,
                         init_weights=init_weights, )

        print(f"\033[94mActor MLP: {self.actor}\033[0m")

        # Value function
        self.critic = MLP(input_size=self.num_critic_input,
                          output_size=self.num_critic_output,
                          hidden_dims=critic_hidden_dims,
                          activation=activation,
                          init_weights=init_weights, )

        print(f"\033[94mCritic MLP: {self.critic}\033[0m")

        # Action noise
        self.fixed_std = fixed_std
        self.init_noise_std = numpy.array(init_noise_std)  # numpy.array

        # Jason 2023-12-27:
        # every action has the different noise std
        std = torch.from_numpy(self.init_noise_std)  # torch.Tensor

        self.std = nn.Parameter(std)
        self.distribution = None

        print(f"ActorCriticMLP: init_noise_std = {self.init_noise_std})")
        print(f"ActorCriticMLP: std = {self.std})")

        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def load_state_dict(self, state_dict, strict=True):
        self.std.data = state_dict["std"]
        print(f"set state_dict[std] = {state_dict['std']}")

        super(ActorCriticMLP, self).load_state_dict(state_dict, strict)

    def reset(self, dones=None):
        """Reset hidden states if env resets.

        Args:
            dones (bool):   reset flags of envs.

        Returns:
            None.
        """
        pass

    def forward(self, observations=None, **kwargs):
        raise NotImplementedError

    @property
    def action_mean(self):
        """torch.Tensor: mean of action distributions"""
        return self.distribution.mean

    @property
    def action_std(self):
        """torch.Tensor: std of action distributions."""
        return self.distribution.stddev

    @property
    def entropy(self):
        """torch.Tensor:   entropy of action distributions."""
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        """Update action distributions loc and scale parameters.

        Args:
            observations    (torch.Tensor): input tensor of actor network.

        Returns:
            torch.distributions.Normal: distributions of actions.
        """
        mean = self.actor(observations)
        std = self.std.to(mean.device)

        self.distribution = Normal(mean, mean * 0. + std)

    def act(self, observations, **kwargs):
        """Generate actions from current observations.

        Args:
            observations    (torch.Tensor): input tensor of actor network.
            **kwargs:   Arbitrary keyword arguments.

        Returns:
            torch.Tensor:   actions sampled from action distributions.
        """
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        """Compute log probabilities of actions.

        Args:
            actions         (torch.Tensor): output of actor network.

        Returns:
            torch.Tensor:   log probability of envs.
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        """Freeze mode of actor network.

        Args:
            observations    (torch.Tensor): input tensor of actor network.

        Returns:
            torch.Tensor:   output of actor network with parameters freezed.
        """
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations=None, **kwargs):
        """Compute value from critic network.

        Args:
            critic_observations (torch.Tensor): input tensor of critic network.
            **kwargs:   Arbitrary keyword arguments.

        Returns:
            torch.Tensor:   [num_envs, 1], output of critic network.

        """
        value = self.critic(critic_observations)
        return value
