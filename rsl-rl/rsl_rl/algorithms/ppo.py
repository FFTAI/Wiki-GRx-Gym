import json
import torch
import torch.optim as optim
import torch.nn as nn

from rsl_rl.modules import *
from rsl_rl.storage import *


class PPO:
    """PPO algorithm class.
    """

    actor_critic: ActorCriticMLP

    def __init__(self,
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
                 **kwargs):
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
            print("PPO.__init__ got unexpected arguments, which will be ignored: "
                  + str([key for key in kwargs.keys()]))

        self.device = device

        self.desired_kl = desired_kl
        self.mean_kl = 0.0
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.learning_rate_min = learning_rate_min
        self.learning_rate_max = learning_rate_max
        self.weight_decay = weight_decay

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)

        # Storage
        self.storage_class = storage_class
        self.transition = None  #: Transition: transition data of current control step.
        self.storage = None  #: RolloutStorage: storage for batch data.

        # 将 actor_critic 的参数传入优化器，整体进行优化
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.num_updates = 0
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # mirror
        self.symmetry_coef = 0

    def _init_storage_cfg(self):
        actor_obs_shape = [self.actor_critic.num_actor_input]
        critic_obs_shape = [self.actor_critic.num_critic_input]
        actions_shape = [self.actor_critic.num_actor_output]

        storage_cfg = {
            "actor_obs_shape": actor_obs_shape,
            "critic_obs_shape": critic_obs_shape,
            "actions_shape": actions_shape,
            "device": self.device,
        }

        return storage_cfg

    def init_storage(self,
                     num_envs,
                     num_transitions_per_env,
                     **kwargs):
        """Initialize storage for batch data.

        Args:
            num_envs (int): number envs.
            num_transitions_per_env (int): number of transitions per env per iteration.
        """
        storage_cfg = self._init_storage_cfg()

        print("storage_class: \n",
              self.storage_class)
        print("storage_cfg: \n",
              json.dumps(storage_cfg, indent=4, sort_keys=True))

        storage_class = eval(self.storage_class)

        self.transition = storage_class.Transition()
        self.storage = storage_class(num_envs,
                                     num_transitions_per_env,
                                     **storage_cfg)

    def test_mode(self):
        """set ActorCritic to test mode."""
        self.actor_critic.test()

    def train_mode(self):
        """set ActorCritic to train mode."""
        self.actor_critic.train()

    def act(self, actor_observations, critic_observations):
        """Process observations and produce actions.

        Args:
            actor_observations (torch.Tensor): [num_envs, num_obs], actor observation tensor.
            critic_observations (torch.Tensor): [num_envs, critic_num_input], critic observation tensor.

        Returns:
            torch.Tensor: [num_envs, actor_num_output], action tensor.
        """

        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(actor_observations).detach()
        self.transition.values = self.actor_critic.evaluate(critic_observations).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()

        # need to record obs and critic_obs before env.step()
        self.transition.observations = actor_observations
        self.transition.critic_observations = critic_observations

        # Jason 2023-11-15:
        # should not call detach() here, otherwise the actor_output will not be updated
        # actor_output = self.actor_critic.act(actor_observations).detach()
        # return actor_output

        return self.transition.actions

    def act_inference(self, obs):
        actions = self.actor_critic.act_inference(obs)
        return actions

    def process_env_step(self, rewards, dones, infos):
        """Process quantities after environment step-in with new actions.

        Args:
            rewards (torch.Tensor): rewards
            dones (torch.Tensor): done flag
            infos (_type_): other infos
        """

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        """Compute returns of trejactory

        Args:
            last_critic_obs (torch.Tensor): last observation tensor input to critic per env.
        """
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update_learning_rate(self, kl_mean):
        if kl_mean > self.desired_kl * 2.0:
            self.learning_rate = max(self.learning_rate_min, self.learning_rate / 1.5)
        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            self.learning_rate = min(self.learning_rate_max, self.learning_rate * 1.5)
        else:
            self.learning_rate = self.learning_rate

    def update(self):
        """Update policy with batch data collected during current learning iteration.

        Returns:
            float: value loss and surrogate loss of PPO.
        """
        mean_value_loss = 0
        mean_surrogate_loss = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, critic_obs_batch, actions_batch, \
                target_values_batch, advantages_batch, returns_batch, \
                old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, \
                _, _ in generator:

            # ================================================================================================================
            # mean_imitation_loss = 0
            # mean_symmetry_loss = 0
            # if self.actor_critic.is_recurrent:
            #     generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            # elif self.symmetry_coef > 0:
            #     generator = self.storage.symmetric_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            # else:
            #     generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            # for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
            # old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
            # ================================================================================================================

            self.actor_critic.act(obs_batch)

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)

            value_batch = self.actor_critic.evaluate(critic_obs_batch)

            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL (Kullback-Leibler divergence)
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5)
                        + (torch.square(old_sigma_batch)
                           + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    self.mean_kl = kl_mean.item()
                    self.update_learning_rate(kl_mean)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(torch.squeeze(actions_log_prob_batch) - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) \
                                * torch.clamp(ratio,
                                              1.0 - self.clip_param,
                                              1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch \
                                + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # calculate other loss
            other_loss = self.calculate_other_loss(obs_batch, critic_obs_batch, actions_batch)

            loss = surrogate_loss \
                   + self.value_loss_coef * value_loss \
                   - self.entropy_coef * entropy_batch.mean() \
                   + other_loss

            if torch.isnan(loss):
                # print("loss is nan")
                continue

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # log
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

            # log other loss
            self.log_other_loss()

        self.num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= self.num_updates
        mean_surrogate_loss /= self.num_updates

        # mean other loss
        self.mean_other_loss()

        return mean_value_loss, mean_surrogate_loss

    def calculate_other_loss(self, obs_batch, critic_obs_batch, actions_batch):
        return 0.0

    def log_other_loss(self):
        pass

    def mean_other_loss(self):
        pass

    def clear_storage(self):
        self.storage.clear()
