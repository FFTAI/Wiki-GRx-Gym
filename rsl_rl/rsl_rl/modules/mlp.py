import torch
import torch.nn as nn

from rsl_rl.utils.utils import get_activation, get_norm


class MLP(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_dims=[256, 256, 256],
                 activation="relu",
                 norm="none",
                 requires_grad=True,
                 **kwargs):
        super(MLP, self).__init__()

        # local vriables
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dims = hidden_dims

        layers = []
        for l in range(len(hidden_dims)):
            if l == 0:
                layers.append(nn.Linear(input_size, hidden_dims[l]))
            else:
                layers.append(nn.Linear(hidden_dims[l - 1], hidden_dims[l]))
            layers.append(get_activation(activation))

        layers.append(nn.Linear(hidden_dims[-1], output_size))

        if norm is not None and norm != "none":
            layers.append(get_norm(norm))

        self.model = nn.Sequential(*layers)

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        return self.model(x)
