import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            middle_dim: int,
            latent_dims: int):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(feature_dim, middle_dim)
        self.linear2 = nn.Linear(middle_dim, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)
