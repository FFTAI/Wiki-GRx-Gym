import time
import os
import json
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.env import *
from rsl_rl.modules import *
from rsl_rl.algorithms import *
from rsl_rl.storage import *

from .on_policy_runner import OnPolicyRunner


class OnPolicyRunnerMirror(OnPolicyRunner):
    """A class prepares policy and alogrithm, and methods to train them and logging.

    Attributes:
        env (VecEnv): environment the robots live in and interact with.
        cfg (dict): configuration of OnPolicyRunner.
        algorithm_cfg (dict): configuration of PPO.
        alg (PPO): policy gradient algorithm.
        policy_cfg (dict): configuration of ActorCritic.
        num_steps_per_env (int): number of transitions per env per iteration.
        save_interval (int): interation interval before saving.
        log_dir (str): logging path.
        writer (SummaryWriter): tensorboard logging handle.
        tot_timesteps (int): total time step policy learnt through.
        tot_time (float): total sim time policy learnt.
        current_learning_iteration (int): current learning iteration number.
    """

    def __init__(
            self,
            env: VecEnv,
            train_cfg,
            log_dir=None,
            device="cpu",
    ):
        """Init method of OnPolicyRunner.

        Args:
            env (VecEnv): environment the robots live in and interact with.
            train_cfg (dict): training configuration.
            log_dir (str, optional): directory to put logs. Defaults to None.
            device (str, optional): device where simulation and policy runninng on. Defaults to 'cpu'.
        """

        super().__init__(env, train_cfg, log_dir, device)

    def init(self, env, train_cfg, device):

        print("----------------------------------")
        print("OnPolicyRunnerMirror")

        self.cfg = train_cfg["runner"]
        self.algorithm_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]

        print("self.cfg: \n", json.dumps(self.cfg, indent=4, sort_keys=True))
        print("self.algorithm_cfg: \n", json.dumps(self.algorithm_cfg, indent=4, sort_keys=True))
        print("self.policy_cfg: \n", json.dumps(self.policy_cfg, indent=4, sort_keys=True))

        self.device = device
        self.env = env

        print("self.device: \n", self.device)
        print("self.env: \n", self.env)

        # Actor-Critic
        actor_num_input = self._init_actor_num_input()
        critic_num_input = self._init_critic_num_input()
        actor_num_output = self._init_actor_num_output()

        print("actor_num_input: \n", actor_num_input)
        print("critic_num_input: \n", critic_num_input)
        print("actor_num_output: \n", actor_num_output)

        actor_critic_class = eval(self.policy_cfg["class_name"])

        actor_critic: ActorCriticMLP = actor_critic_class(actor_num_input,
                                                          critic_num_input,
                                                          actor_num_output,
                                                          **self.policy_cfg).to(self.device)

        # Mirror
        mirror = Mirror(self.env)

        # PPO
        algorithm_class = eval(self.algorithm_cfg["class_name"])

        self.algorithm = algorithm_class(actor_critic=actor_critic,
                                         device=self.device,
                                         # Mirror
                                         mirror=mirror,
                                         **self.algorithm_cfg)

        # init storage and model
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.algorithm.init_storage(self.env.num_envs,
                                    self.num_steps_per_env)

        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """Defines learning process of policy.

        Args:
            num_learning_iterations (int): total iterations model will learn.
            init_at_random_ep_len (bool, optional): flag of random episode length. Defaults to False.
        """
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # actor_critic obs
        obs = self.env.get_observations()
        pri_obs = self.env.get_privileged_observations()
        critic_obs = pri_obs if pri_obs is not None else obs

        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        # switch to train mode (for dropout for example)
        self.algorithm.actor_critic.train()
        self.algorithm.actor_critic.actor.train()

        ep_infos = []
        rew_buffer = deque(maxlen=100)
        len_buffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()

            # Rollout
            with torch.inference_mode():  # 关闭 Actor 梯度计算，开启训练数据采集过程

                for i in range(self.num_steps_per_env):

                    # rl -> env: calculate ppo act
                    actions = self.algorithm.act(obs, critic_obs)

                    # env -> rl: env step
                    obs, pri_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = pri_obs if pri_obs is not None else obs

                    # get obs
                    obs, critic_obs, rewards, dones = \
                        obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # process env step
                    self.algorithm.process_env_step(rewards, dones, infos)

                    # logging
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rew_buffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        len_buffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                # compute time elapsed
                stop = time.time()
                collection_time = stop - start
                start = stop

                # Learning step
                self.algorithm.compute_returns(critic_obs)

            (
                mean_value_loss,
                mean_surrogate_loss,
            ) = self.algorithm.update()

            mean_mirror_loss = \
                self.algorithm.update_mirror()

            # will clear storage here!
            self.algorithm.clear_storage()

            stop = time.time()
            learn_time = stop - start

            # update current learning iteration
            self.current_learning_iteration = it

            if self.log_dir is not None:
                self.log(locals())

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _addition_log_string(self, width=80, pad=35) -> str:
        """Additional logging string.

        Returns:
            str: additional logging string.
        """
        addition_log_string = (
            f"""{'Mirror loss:':>{pad}} {self.algorithm.mean_mirror_loss:.6f}\n"""
        )

        return addition_log_string

    def log(self, locs, width=80, pad=35):
        """logging method.

        Args:
            locs (locals): local variables.
            width (int): width of output string.
            pad (int): padding length of output string.
        """
        super().log(locs, width, pad)

        self.writer.add_scalar('Loss/mirror', locs['mean_mirror_loss'], locs['it'])
