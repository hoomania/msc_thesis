import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
            self,
            latent_dims: int,
            middle_dim: int,
            output_dim: int):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.linear1 = nn.Linear(latent_dims, middle_dim)
        self.linear2 = nn.Linear(middle_dim, output_dim)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, self.output_dim))
