from abc import ABC, abstractmethod
import torch
from typing import Tuple, Union


# minimal interface of the environment
class VecEnv(ABC):
    num_envs: int
    num_obs: int
    num_pri_obs: int
    num_stack: int
    use_stack: bool
    actor_num_input: int
    critic_num_input: int
    actor_num_output: int

    max_episode_length: int
    pri_obs_buf: torch.Tensor
    obs_buf: torch.Tensor
    rew_buf: torch.Tensor
    reset_buf: torch.Tensor
    episode_length_buf: torch.Tensor  # current episode duration
    extras: dict
    device: torch.device

    @abstractmethod
    def step(self, actions: torch.Tensor) \
            -> Tuple[torch.Tensor,
            Union[torch.Tensor, None],
            torch.Tensor,
            torch.Tensor,
            dict]:
        pass

    @abstractmethod
    def reset(self, env_ids: Union[list, torch.Tensor]):
        pass

    @abstractmethod
    def get_observations(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        pass
