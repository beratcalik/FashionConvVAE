import os
# OpenMP çakışma hatası çözümü
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ConvVAE
from utils import plot_losses, visualize_reconstruction, plot_latent_space, sample_images

# 1. Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Hiperparametreler
batch_size    = 128
epochs        = 100
learning_rate = 5e-4
log_interval  = 200
latent_dim    = 32

# 3. Veri yükleyiciler
transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.FashionMNIST("data/fashion-mnist", train=True,  download=True, transform=transform)
test_ds  = datasets.FashionMNIST("data/fashion-mnist", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# 4. Model ve optimizer
model     = ConvVAE(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. Kayıp fonksiyonu: Recon + KLD
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

# 6. Eğitim fonksiyonu
def train(epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss, _, _ = loss_function(recon, data, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % log_interval == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss/pixel: {loss.item()/len(data):.4f}")

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"====> Epoch {epoch} Average loss: {avg_loss:.4f}")
    return avg_loss

# 7. Test fonksiyonu
def test(epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon, mu, logvar = model(data)
            loss, _, _ = loss_function(recon, data, mu, logvar)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader.dataset)
    print(f"====> Test set loss: {avg_loss:.4f}")
    return avg_loss

# 8. Ana döngü
if __name__ == "__main__":
    train_losses, test_losses = [], []

    for epoch in range(1, epochs+1):
        tr = train(epoch)
        ts = test(epoch)
        train_losses.append(tr)
        test_losses.append(ts)

    # Kayıp eğrileri
    plot_losses(train_losses, test_losses, save_path="loss_curve.png")

    # Görselleştirmeler
    visualize_reconstruction(model, train_loader, device, num_images=8)
    # Latent uzayı 2D'de görmek için latent_dim==2 olmalı
    if latent_dim == 2:
        plot_latent_space(model, test_loader, device)
    sample_images(model, device, num_samples=16, latent_dim=latent_dim)

    # Modeli kaydet
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/convvae_fashionmnist.pth")
    print("Model kaydedildi → checkpoints/convvae_fashionmnist.pth")
