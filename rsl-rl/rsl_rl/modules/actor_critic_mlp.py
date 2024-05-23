import numpy as np

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
                 fixed_std=False,
                 init_noise_std=1.0,
                 set_std=True,
                 set_noise_std=1.0,
                 actor_output_activation=None,
                 critic_output_activation=None,
                 **kwargs):
        """Default ActorCritic network

        Args:
            actor_num_input:          input dim to actor network.
            critic_num_input:         input dim to critic network.
            actor_num_output:            output dim of actor network.
            actor_hidden_dims:      dims of hidden layers in actor network.
            critic_hidden_dims:     dims of hidden layers in critic network.
            activation:             activation function name.
            output_activation:      name of output layers' activation function
            init_noise_std:         initial value of ActorCritic.std.
            **kwargs:               Arbitrary keyword arguments.

        Returns:
            nn.Module:              policy of MLPs
        """

        print("----------------------------------")
        print("ActorCriticMLP")

        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))

        super(ActorCriticMLP, self).__init__()

        # dimensions
        self.num_actor_input = actor_num_input
        self.num_actor_output = actor_num_output
        self.num_critic_input = critic_num_input
        self.num_critic_output = 1

        # Policy
        self.actor = MLP(self.num_actor_input,
                         self.num_actor_output,
                         actor_hidden_dims,
                         activation,
                         norm="none")

        print(f"Actor MLP: {self.actor}")

        # Value function
        self.critic = MLP(self.num_critic_input,
                          1,
                          critic_hidden_dims,
                          activation,
                          norm="none")

        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.fixed_std = fixed_std
        self.init_noise_std = init_noise_std

        # Jason 2023-12-27:
        # every action has the different noise std
        std = init_noise_std * torch.ones(actor_num_output)
        self.std = nn.Parameter(std)
        self.distribution = None

        print(f"ActorCritic: fixed_std = {fixed_std})")
        print(f"ActorCritic: init_noise_std = {init_noise_std})")
        print(f"ActorCritic: std = {std})")

        # set std when load state_dict
        self.set_std = set_std
        self.set_noise_std = set_noise_std

        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        """Initialize network weights

        Args:
            sequential (nn.Module): sequential model of network
            scale   (list): initial weights of model

        Returns:
            None
        """
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def load_state_dict(self, state_dict, strict=True):
        if self.set_std:
            print("Warning: Not loading std from state_dict, use init value")

            print(f"original state_dict[std] = {state_dict['std']}")

            # change Mapping class state_dict value
            state_dict["std"] = torch.ones_like(state_dict["std"]) * self.set_noise_std

            print(f"set state_dict[std] = {state_dict['std']}")
        else:
            self.std.data = state_dict["std"]

        # set std to fixed
        if self.fixed_std:
            self.std.data = self.init_noise_std * torch.ones_like(self.std.data)
            self.std.requires_grad = False

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

        if self.fixed_std:
            std = self.init_noise_std
        else:
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
