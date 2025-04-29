import torch.nn as nn

from rsl_rl.modules import *
from rsl_rl.storage import *
from rsl_rl.algorithms import *

from .ppo import PPO


class PPOMirror(PPO):
    """
    PPO algorithm class.
    """

    def __init__(
            self,
            actor_critic=None,
            num_learning_epochs=1,
            num_mini_batches=1,
            clip_param=0.2,
            gamma=0.998,
            lam=0.95,
            value_loss_coef=1.0,
            entropy_coef=0.0,
            learning_rate=1e-3,
            learning_rate_min=1e-5,
            learning_rate_max=1e-2,
            weight_decay=0.0,
            max_grad_norm=1.0,
            use_clipped_value_loss=True,
            schedule="fixed",
            desired_kl=0.01,
            device='cpu',
            storage_class="RolloutStorage",
            # Mirror
            mirror=None,
            mirror_coef=1.0,
            **kwargs
    ):
        """Initialize PPO algorithm.

        Args:
            actor_critic (ActorCritic): policy network.
            num_learning_epochs (int, optional): number of learning epochs. Defaults to 1.
            num_mini_batches (int, optional): number of mini batches. Defaults to 1.
            clip_param (float, optional): clipping parameter. Defaults to 0.2.
            gamma (float, optional): discount factor. Defaults to 0.998.
            lam (float, optional): GAE parameter. Defaults to 0.95.
            value_loss_coef (float, optional): value loss coefficient. Defaults to 1.0.
            entropy_coef (float, optional): entropy coefficient. Defaults to 0.0.
            learning_rate (float, optional): learning rate. Defaults to 1e-3.
            learning_rate_min (float, optional): minimum learning rate. Defaults to 1e-5.
            learning_rate_max (float, optional): maximum learning rate. Defaults to 1e-2.
            weight_decay (float, optional): weight decay. Defaults to 0.0.
            max_grad_norm (float, optional): maximum gradient norm. Defaults to 1.0.
            use_clipped_value_loss (bool, optional): use clipped value loss. Defaults to True.
            schedule (str, optional): learning rate schedule. Defaults to "fixed".
            desired_kl (float, optional): desired KL divergence. Defaults to 0.01.
            device (str, optional): device. Defaults to 'cpu'.
        """
        if kwargs:
            print("PPOMirror.__init__ got unexpected arguments, which will be ignored: "
                  + str([key for key in kwargs.keys()]))

        PPO.__init__(
            self,
            actor_critic=actor_critic,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            learning_rate_min=learning_rate_min,
            learning_rate_max=learning_rate_max,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
            storage_class=storage_class,
            **kwargs
        )

        self._init_mirror(
            mirror=mirror,
            mirror_coef=mirror_coef,
            **kwargs
        )

    def _init_mirror(
            self,
            mirror=None,
            mirror_coef=1.0,
            **kwargs
    ):
        # Mirror parameters
        self.mirror: Mirror = mirror
        self.mirror_coef = mirror_coef
        print("Mirror coef: ", self.mirror_coef)

        self.mirror_loss = 0.0
        self.mean_mirror_loss = 0.0

    def calculate_other_loss(self, obs_batch, critic_obs_batch, actions_batch):
        # Mirror loss 镜像损失
        if self.mirror_coef > 0:
            # Origin observation and action
            origin_action_batch = self.actor_critic.act_inference(obs_batch)

            # get mirror observation and action
            mirror_obs_batch, target_action_batch = self._update_mirror_observation_and_action(obs_batch)

            # get mirror loss
            self.mirror_loss = self._update_mirror_loss(origin_action_batch, target_action_batch)
        else:
            self.mirror_loss = 0

        loss = self.mirror_loss

        return loss

    def log_other_loss(self):
        if self.mirror_coef > 0:
            self.mean_mirror_loss += self.mirror_loss.item()
        else:
            self.mean_mirror_loss = 0

    def mean_other_loss(self):
        self.mean_mirror_loss /= self.num_updates

    def update_mirror(self):
        return self.mean_mirror_loss

    def _update_mirror_observation_and_action(self, obs_batch):
        # obs ->(mirror)-> mirror_obs -> mirror_action ->(mirror)-> action -> compare with origin action
        mirror_obs_batch = self.mirror.get_mirror_obs_batch(obs_batch=obs_batch)
        mirror_action_batch = self.actor_critic.act_inference(mirror_obs_batch)
        target_action_batch = self.mirror.get_mirror_action_batch(mirror_action_batch)
        return mirror_obs_batch, target_action_batch

    def _update_mirror_loss(self, origin_action_batch, mirror_action_batch):
        mse_loss = nn.MSELoss()
        if self.mirror_coef > 0:
            mirror_loss = mse_loss(origin_action_batch, mirror_action_batch) * self.mirror_coef
        else:
            mirror_loss = 0
        return mirror_loss
