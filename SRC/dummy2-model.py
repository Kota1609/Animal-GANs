import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, Z_dim, Stru_dim, Time_dim, Dose_dim, Measurement_dim):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.LayerNorm(out_feat))  # LayerNorm for small datasets
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            *block(Z_dim + Stru_dim + Time_dim + Dose_dim, 2048, normalize=True),
            *block(2048, 1024),
            *block(1024, 512),
            *block(512, 128),
            nn.Linear(128, Measurement_dim),
            nn.Tanh()
        )

    def forward(self, noise, Stru, Time, Dose):
        gen_input = torch.cat([noise, Stru, Time, Dose], -1)
        Measurement = self.model(gen_input)
        return Measurement


class Discriminator(nn.Module):
    def __init__(self, Stru_dim, Time_dim, Dose_dim, Measurement_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(Stru_dim + Time_dim + Dose_dim + Measurement_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 128),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, Measurement, Stru, Time, Dose):
        d_in = torch.cat((Measurement, Stru, Time, Dose), -1)
        validity = self.model(d_in)
        return validity
