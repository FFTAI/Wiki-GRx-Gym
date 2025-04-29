import time
import os
import json
from collections import deque
import statistics

import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.env import *
from rsl_rl.modules import *
from rsl_rl.algorithms import *
from rsl_rl.storage import *


class OnPolicyRunner:
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
            device (str, optional): device where simulation and policy runninng on. Defaults to "cpu".
        """

        self.init(env, train_cfg, device)
        self.init_log(log_dir)

    def init(self, env, train_cfg, device):

        print("----------------------------------")
        print("OnPolicyRunner")

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

        # PPO
        algorithm_class = eval(self.algorithm_cfg["class_name"])

        self.algorithm = algorithm_class(actor_critic=actor_critic,
                                         device=self.device,
                                         **self.algorithm_cfg, )

        # init storage and model
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.algorithm.init_storage(self.env.num_envs,
                                    self.num_steps_per_env, )

        self.env.reset()

    def _init_actor_num_input(self):
        actor_num_input = self.env.num_obs

        if self.env.actor_obs_use_stack:
            actor_num_input *= self.env.num_stack

        return actor_num_input

    def _init_critic_num_input(self):
        critic_num_input = self.env.num_obs

        if self.env.num_pri_obs is not None:
            critic_num_input = self.env.num_pri_obs

        return critic_num_input

    def _init_actor_num_output(self):
        actor_num_output = self.env.num_actions

        return actor_num_output

    def init_log(self, log_dir):
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
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
        self.train_mode()

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
            pass

            with torch.inference_mode():  # 关闭 Actor 梯度计算，开启训练数据采集过程

                for i in range(self.num_steps_per_env):

                    # rl -> env: calculate ppo act
                    """
                    Jason 2024-11-30:
                    在此处完成 obs, critic_obs, actions 的记录
                    """
                    actions = self.algorithm.act(obs, critic_obs)

                    # env -> rl: env step
                    obs, pri_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = pri_obs if pri_obs is not None else obs

                    # get obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )

                    # process env step
                    """
                    Jason 2024-11-30:
                    在此处完成 rewards, dones, infos 的记录
                    """
                    self.algorithm.process_env_step(rewards, dones, infos)

                    # logging
                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])

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
        return f""" """

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """logging method.

        Args:
            locs (locals): local variables.
            width (int): width of output string.
            pad (int): padding length of output string.
        """
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # ------------------------------------------------
        # Tensorboard log

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)

                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.algorithm.learning_rate, locs["it"])
        self.writer.add_scalar("Loss/kl", self.algorithm.mean_kl, locs["it"])

        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["rew_buffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rew_buffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["len_buffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rew_buffer"]), self.tot_time)
            self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["len_buffer"]), self.tot_time)

        # log stds
        stds = self.algorithm.actor_critic.std
        mean_std = self.algorithm.actor_critic.std.mean()

        for i, std in enumerate(stds):
            self.writer.add_scalar(f"Policy/noise_std_{i}", std.item(), locs["it"])

        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # ------------------------------------------------
        # Print log

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rew_buffer"]) > 0:
            log_string = (f"""{"#" * width}\n"""
                          f"""{str.center(width, " ")}\n\n"""
                          f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs[
                              "collection_time"]:.3f}s, learning {locs["learn_time"]:.3f}s)\n"""
                          f"""{"Value function loss:":>{pad}} {locs["mean_value_loss"]:.4f}\n"""
                          f"""{"Surrogate loss:":>{pad}} {locs["mean_surrogate_loss"]:.4f}\n"""
                          f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
                          f"""{"Mean reward:":>{pad}} {statistics.mean(locs["rew_buffer"]):.2f}\n"""
                          f"""{"Mean episode length:":>{pad}} {statistics.mean(locs["len_buffer"]):.2f}\n""")
            #   f"""{"Mean reward/step:":>{pad}} {locs["mean_reward"]:.2f}\n"""
            #   f"""{"Mean episode length/episode:":>{pad}} {locs["mean_trajectory_length"]:.2f}\n""")
        else:
            log_string = (f"""{"#" * width}\n"""
                          f"""{str.center(width, " ")}\n\n"""
                          f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs[
                              "collection_time"]:.3f}s, learning {locs["learn_time"]:.3f}s)\n"""
                          f"""{"Value function loss:":>{pad}} {locs["mean_value_loss"]:.4f}\n"""
                          f"""{"Surrogate loss:":>{pad}} {locs["mean_surrogate_loss"]:.4f}\n"""
                          f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{"Mean reward/step:":>{pad}} {locs["mean_reward"]:.2f}\n"""
            #   f"""{"Mean episode length/episode:":>{pad}} {locs["mean_trajectory_length"]:.2f}\n""")

        log_string += ep_string

        # make the eta in H:M:S
        eta_seconds = self.tot_time / (locs["it"] + 1) * (locs["num_learning_iterations"] - locs["it"])

        # Convert seconds to H:M:S
        eta_h, rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(rem, 60)

        log_string += (f"""{"-" * width}\n"""
                       f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
                       f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
                       f"""{"Total time:":>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{"ETA:":>{pad}} {int(eta_h)}h {int(eta_m)}m {int(eta_s)}s\n""")

        log_string += self._addition_log_string(width=width, pad=pad)

        print(log_string)

    def save(self, path, infos=None):
        """Save trained model to path.

        Args:
            path (str): path to save model
            infos (dict, optional): addtional information of model. Defaults to None.
        """
        saved_dict = {
            "model_state_dict": self.algorithm.actor_critic.state_dict(),
            "optimizer_state_dict": self.algorithm.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }

        torch.save(saved_dict, path)

    def load(self, path, load_optimizer=True):
        """Load saved model from path.

        Args:
            path (str): path to load model.
            load_optimizer (bool, optional): flag if load optimizer with model. Defaults to True.

        Returns:
            nn.Module: torch model load from path.
        """
        loaded_dict = torch.load(path)
        self.algorithm.actor_critic.load_state_dict(loaded_dict["model_state_dict"])

        if load_optimizer:
            self.algorithm.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])

        self.current_learning_iteration = loaded_dict["iter"]

        infos = loaded_dict["infos"]
        return infos

    def get_inference_policy(self, device=None):
        """Switch model to evaluation mode

        Args:
            device (str, optional): device model goes to. Defaults to None.

        Returns:
            method: inference method of model
        """
        self.eval_mode()  # switch to evaluation mode (dropout for example)

        if device is not None:
            self.algorithm.actor_critic.to(device)

        return self.algorithm.actor_critic.act_inference

    def train_mode(self):
        self.algorithm.actor_critic.train()
        self.algorithm.actor_critic.actor.train()

    def eval_mode(self):
        self.algorithm.actor_critic.eval()
        self.algorithm.actor_critic.actor.eval()
