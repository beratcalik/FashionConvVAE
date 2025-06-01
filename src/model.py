import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        # Encoder aynen…
        self.enc1 = nn.Conv2d(1, 32, 4, 2, 1)   # 28→14
        self.enc2 = nn.Conv2d(32, 64, 4, 2, 1)  # 14→7
        self.enc3 = nn.Conv2d(64, 128, 4, 2, 1) # 7→3

        # Latent katmanlar…
        self.fc_mu  = nn.Linear(128*3*3, latent_dim)
        self.fc_log = nn.Linear(128*3*3, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 128*3*3)
        # Aşağıdaki dec1’e output_padding=1 ekliyoruz, böylece 3→7 boyutu sağlanacak
        self.dec1 = nn.ConvTranspose2d(128, 64, 4, 2, 1, output_padding=1)  # 3→7
        # Geriye kalanlar:
        self.dec2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)                      # 7→14
        self.dec3 = nn.ConvTranspose2d(32, 1,  4, 2, 1)                      # 14→28

    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_log(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc_dec(z))
        h = h.view(-1, 128, 3, 3)
        h = F.relu(self.dec1(h))
        h = F.relu(self.dec2(h))
        return torch.sigmoid(self.dec3(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
