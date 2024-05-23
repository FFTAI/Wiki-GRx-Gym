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

        for l in range(len(hidden_dims)):
            if l == 0:
                self.add_module(
                    "fc{}".format(l), nn.Linear(input_size, hidden_dims[l]),
                )
            else:
                self.add_module(
                    "fc{}".format(l), nn.Linear(hidden_dims[l - 1], hidden_dims[l]),
                )
            self.add_module(
                "act{}".format(l), get_activation(activation),
            )
        self.add_module(
            "fc{}".format(len(hidden_dims)), nn.Linear(hidden_dims[-1], output_size),
        )

        # add normalization layer if there is any
        if norm is not None and norm != "none":
            self.add_module(
                "norm", get_norm(norm),
            )

        # local vriables
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dims = hidden_dims

        # set requires_grad
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        out = x

        # get number of layers named "fc"
        number_of_fc_layers = len([name for name in self._modules if "fc" in name])
        number_of_norm_layers = len([name for name in self._modules if "norm" in name])

        for l in range(number_of_fc_layers - 1):
            out = self._modules["fc{}".format(l)](out)
            out = self._modules["act{}".format(l)](out)

        out = self._modules["fc{}".format(number_of_fc_layers - 1)](out)

        # apply normalization if there is any
        if number_of_norm_layers > 0:
            out = self._modules["norm"](out)

        return out
