import torch
import torch.nn as nn
import numpy as np


class HyperNetwork(nn.Module):
    def __init__(self, hyper: nn.Module, base: nn.Module):
        super().__init__()
        self._hyper = hyper
        params_lookup = [[None, None, 0, 0]]
        for name, param in base.named_parameters():
            s = params_lookup[-1][-1]
            params_lookup.append([name, param.shape, s, s + np.prod(param.shape)])
        self._params_lookup = params_lookup[1:]
        buffers = {}
        self._call = lambda params, data: torch.func.functional_call(
            base, (params, buffers), (data,)
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        params = self._hyper(x)
        params_dict = {}
        for i in self._params_lookup:
            params_dict[i[0]] = params[:, i[2] : i[3]].reshape(-1, *i[1])
        return torch.vmap(self._call, in_dims=(0, 0))(params_dict, z)


class SetEmbeddingNetwork(nn.Module):
    def __init__(
        self, input_dim, output_dim, num_heads=2, num_layers=2, hidden_dim=128
    ):
        super(SetEmbeddingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.attention_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    batch_first=True,
                    dropout=0,
                    activation="relu",
                )
                for _ in range(num_layers)
            ]
        )
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, N, D)
        x = self.fc1(x)
        for layer in self.attention_layers:
            x = layer(x)
        # Mean pooling over the set dimension (N)
        x = x.mean(dim=1)
        # Final linear layer to get the output dimension (B, M)
        x = self.fc2(x)
        return x
