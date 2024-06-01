from typing import Tuple, List

import numpy
import torch
import torch.nn as nn

_EPS = numpy.finfo(float).eps * 4.0


def split_and_pad_trajectories(tensor, dones):
    """ Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length of the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories

    Example:

        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]

    Assumes that the input has the following dimension order: [time, number of envs, aditional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
    # TODO: pad trajectories to fixed length of steps_per_env.
    # assert trajectory_lengths.max() == tensor.shape[0], f'Longest traj is {trajectory_lengths.max()}, we need at least one can hold {tensor.shape[0]} steps without falling!'
    # add at least one full length trajectory
    trajectories = trajectories + (torch.zeros(tensor.shape[0], tensor.shape[-1], device=tensor.device),)
    # pad the trajectories to the length of the longest trajectory
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)
    # remove the added tensor
    padded_trajectories = padded_trajectories[:, :-1]

    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
    """ Does the inverse operation of  split_and_pad_trajectories()
    """
    # Need to transpose before and after the masking to have proper reshaping
    return trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, trajectories.shape[0], trajectories.shape[-1]).transpose(1, 0)


# Shiwen @Jan. 2024: add this function to build MLP
def build_mlp(input_size, output_size=1, hidden_dims=[128, 128], activation_fn=nn.ELU(), output_activation_fn=None) -> nn.Module:
    """
        Builds a feedforward neural network

        Args:
            hidden_dim: dimension of hidden layers
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        Returns:
            nn.Module: a MLP network
            [list[float]]: initial weight scale of each layer
    """
    layers = []
    layers.append(nn.Linear(input_size, hidden_dims[0]))
    layers.append(activation_fn)
    scale = [numpy.sqrt(2)]
    for l in range(len(hidden_dims)):
        if l == len(hidden_dims) - 1:
            layers.append(nn.Linear(hidden_dims[l], output_size))
            scale.append(numpy.sqrt(2))
        else:
            layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
            layers.append(activation_fn)
            scale.append(numpy.sqrt(2))
    if output_activation_fn is not None:
        layers.append(output_activation_fn)

        # mlp = nn.Sequential(*layers)
    return layers, scale


class RunningMeanStd(object):
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = numpy.zeros(shape, numpy.float64)
        self.var = numpy.ones(shape, numpy.float64)
        self.count = epsilon

    def update(self, arr: numpy.ndarray) -> None:
        batch_mean = numpy.mean(arr, axis=0)
        batch_var = numpy.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: numpy.ndarray, batch_var: numpy.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + numpy.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class Normalizer(RunningMeanStd):
    def __init__(self, input_dim, epsilon=1e-4, clip_obs=10.0):
        super().__init__(shape=input_dim)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize(self, input):
        return numpy.clip(
            (input - self.mean) / numpy.sqrt(self.var + self.epsilon),
            -self.clip_obs, self.clip_obs)

    def normalize_torch(self, input, device):
        mean_torch = torch.tensor(
            self.mean, device=device, dtype=torch.float32)
        std_torch = torch.sqrt(torch.tensor(
            self.var + self.epsilon, device=device, dtype=torch.float32))
        return torch.clamp(
            (input - mean_torch) / std_torch, -self.clip_obs, self.clip_obs)

    def update_normalizer(self, rollouts, expert_loader):
        policy_data_generator = rollouts.feed_forward_generator_amp(
            None, mini_batch_size=expert_loader.batch_size)
        expert_data_generator = expert_loader.dataset.feed_forward_generator_amp(
            expert_loader.batch_size)

        for expert_batch, policy_batch in zip(expert_data_generator, policy_data_generator):
            self.update(
                torch.vstack(tuple(policy_batch) + tuple(expert_batch)).cpu().numpy())


class Normalize(torch.nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
        self.normalize = torch.nn.functional.normalize

    def forward(self, x):
        x = self.normalize(x, dim=-1)
        return x


def quaternion_slerp(q0, q1, fraction, spin=0, shortestpath=True):
    """Batch quaternion spherical linear interpolation."""

    out = torch.zeros_like(q0)

    zero_mask = torch.isclose(fraction, torch.zeros_like(fraction)).squeeze()
    ones_mask = torch.isclose(fraction, torch.ones_like(fraction)).squeeze()
    out[zero_mask] = q0[zero_mask]
    out[ones_mask] = q1[ones_mask]

    d = torch.sum(q0 * q1, dim=-1, keepdim=True)
    dist_mask = (torch.abs(torch.abs(d) - 1.0) < _EPS).squeeze()
    out[dist_mask] = q0[dist_mask]

    if shortestpath:
        d_old = torch.clone(d)
        d = torch.where(d_old < 0, -d, d)
        q1 = torch.where(d_old < 0, -q1, q1)

    angle = torch.acos(d) + spin * torch.pi
    angle_mask = (torch.abs(angle) < _EPS).squeeze()
    out[angle_mask] = q0[angle_mask]

    final_mask = torch.logical_or(zero_mask, ones_mask)
    final_mask = torch.logical_or(final_mask, dist_mask)
    final_mask = torch.logical_or(final_mask, angle_mask)
    final_mask = torch.logical_not(final_mask)

    isin = 1.0 / angle
    q0 *= torch.sin((1.0 - fraction) * angle) * isin
    q1 *= torch.sin(fraction * angle) * isin
    q0 += q1
    out[final_mask] = q0[final_mask]
    return out


def swap_lr(value: torch.Tensor, left_idx: List[int], right_idx: List[int]) -> torch.Tensor:
    """Swaps elements of the tensor `value` at indices provided by `left_idx` and `right_idx`.

    Args:
        value (torch.Tensor): Tensor with at least as many elements as the largest index in `left_idx` or `right_idx`.
        left_idx (list[int]): List of indices corresponding to the "left" elements in the tensor.
        right_idx (list[int]): List of indices corresponding to the "right" elements in the tensor.

    Returns:
        torch.Tensor: A new tensor with "left" and "right" elements swapped.
    """

    # Ensure that the lists are of equal length to swap elements pairwise
    assert len(left_idx) == len(right_idx), "Index lists must be of the same length."
    # Clone the input tensor to avoid modifying it in-place
    swapped = value.clone()
    # Use indexing to swap 'left' and 'right' elements
    # The following assumes that left_idx and right_idx are 1D lists of indices
    for l_idx, r_idx in zip(left_idx, right_idx):
        swapped[..., l_idx], swapped[..., r_idx] = value[..., r_idx], value[..., l_idx]

    return swapped


def get_activation(act_name):
    """Get activation function from name.

    Args:
        act_name (str): activation function name.

    Returns:
        nn.functional: pytorch activation function.
    """
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


def get_norm(norm_name):
    if norm_name == "batch":
        return nn.BatchNorm1d()
    elif norm_name == "layer":
        return nn.LayerNorm()
    elif norm_name == "instance":
        return nn.InstanceNorm1d()
    elif norm_name == "softmax":
        return nn.Softmax(dim=-1)
    elif norm_name == "none":
        return None
    else:
        print("MLP: invalid normalization function!")
        return None
