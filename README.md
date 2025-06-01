# Convolutional Variational Autoencoder (ConvVAE) on Fashion-MNIST

## Introduction  
Variational Autoencoder (VAE) modelleri, hem sıkıştırma (compression) hem de yeni veri üretme (generation) yeteneğine sahip generatif modellerdir.  
Bu projede, 28×28 gri-ton Fashion-MNIST veri seti üzerinde bir **Convolutional VAE (ConvVAE)** eğitildi. Amaçlar:  
1. Görüntülerin yüksek boyutlu piksellerini, düşük boyutlu latent uzayda temsil edebilmek  
2. Bu temsillerden rekonstrüksiyon ve yeni örnekler üretebilmek  
3. Encoder’ın çıkardığı latent kodları kullanarak downstream bir sınıflandırma görevi yapmak

## Methods

### Veri Hazırlama  
- **Veri Kümesi**: Fashion-MNIST (60 000 eğitim, 10 000 test örneği).  
- **Öznitelik Boyutu**: Her örnek 28×28 piksel, gri-ton.  
- **Normalize Etme**: Piksel değerleri [0, 1] aralığına çekildi (PyTorch `ToTensor()` ile).  
- **Batchleme**: Eğitim için `batch_size = 128`, test için `batch_size = 128`.  
- **DataLoader Kullanımı**:  
  ```python
  from torch.utils.data import DataLoader
  from torchvision import datasets, transforms

  transform = transforms.Compose([
      transforms.ToTensor()
  ])
  train_ds = datasets.FashionMNIST("data/fashion-mnist",
                                   train=True,
                                   download=True,
                                   transform=transform)
  test_ds = datasets.FashionMNIST("data/fashion-mnist",
                                  train=False,
                                  download=True,
                                  transform=transform)
  train_loader = DataLoader(train_ds,
                            batch_size=128,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True)
  test_loader = DataLoader(test_ds,
                           batch_size=128,
                           shuffle=False,
                           num_workers=4,
                           pin_memory=True)

### Model Mimarisi (ConvVAE)

#### Encoder
1. **Katman 1**: `Conv2d(1, 32, kernel_size=4, stride=2, padding=1)`  
   - Girdi: 28×28×1  
   - Çıktı: 14×14×32  
2. **Katman 2**: `Conv2d(32, 64, kernel_size=4, stride=2, padding=1)`  
   - Girdi: 14×14×32  
   - Çıktı: 7×7×64  
3. **Katman 3**: `Conv2d(64, 128, kernel_size=4, stride=2, padding=1)`  
   - Girdi: 7×7×64  
   - Çıktı: 3×3×128  
4. **Flatten ve FC Katmanları**  
   - Çıktı boyutu: 128 × 3 × 3 = 1152  
   - **μ(x)**: `Linear(1152, latent_dim)`  
   - **logσ²(x)**: `Linear(1152, latent_dim)`

#### Reparameterization Trick
\[
z = \mu(x) + \exp\Bigl(\tfrac{1}{2}\log\sigma^2(x)\Bigr)\,\odot\,\epsilon,\quad
\epsilon \sim \mathcal{N}(0, I)
\]

#### Decoder
1. **FC ve Yeniden Şekillendirme**  
   - `Linear(latent_dim, 128*3*3)`  
   - Yeniden şekillendirme: 128×3×3  
2. **Katman 1**:  
   `ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1)`  
   - Girdi: 3×3×128  
   - Çıktı: 7×7×64  
3. **Katman 2**:  
   `ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)`  
   - Girdi: 7×7×64  
   - Çıktı: 14×14×32  
4. **Katman 3**:  
   `ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)`  
   - Girdi: 14×14×32  
   - Çıktı: 28×28×1  
   - Aktivasyon: `Sigmoid` (çıktıyı [0,1] aralığına dönüştürür)

#### Tam Kod (PyTorch)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(1, 32, 4, 2, 1)    # 28→14
        self.enc2 = nn.Conv2d(32, 64, 4, 2, 1)   # 14→7
        self.enc3 = nn.Conv2d(64, 128, 4, 2, 1)  # 7→3
        self.fc_mu  = nn.Linear(128*3*3, latent_dim)
        self.fc_log = nn.Linear(128*3*3, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 128*3*3)
        self.dec1   = nn.ConvTranspose2d(128, 64, 4, 2, 1, output_padding=1)  # 3→7
        self.dec2   = nn.ConvTranspose2d(64, 32, 4, 2, 1)                       # 7→14
        self.dec3   = nn.ConvTranspose2d(32, 1, 4, 2, 1)                        # 14→28

    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))
        h = h.view(h.size(0), -1)  # [batch, 128*3*3]
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
        return self.decode(z), mu, logvar
