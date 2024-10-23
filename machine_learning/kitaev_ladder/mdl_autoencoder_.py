import torch.nn as nn
import mdl_encoder as enc
import mdl_decoder as dec


class Autoencoder(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            middle_dim: int,
            latent_dim: int):
        super(Autoencoder, self).__init__()
        self.encoder = enc.Encoder(
            feature_dim=feature_dim,
            middle_dim=middle_dim,
            latent_dims=latent_dim,
        )
        self.decoder = dec.Decoder(
            latent_dims=latent_dim,
            middle_dim=middle_dim,
            output_dim=feature_dim,
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
