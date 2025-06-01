import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def plot_losses(train_losses, test_losses, save_path=None):
    """
    Epoch başına biriken eğitim ve test kayıplarını çizdirir.
    """
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def visualize_reconstruction(model, data_loader, device, num_images=8):
    """
    İlk batch'ten alınan veriyi model ile yeniden inşa eder ve orijinal-rekonstrüksiyon karşılaştırması yapar.
    """
    model.eval()
    data, _ = next(iter(data_loader))
    data = data.to(device)
    with torch.no_grad():
        recon, _, _ = model(data)
    recon = recon.view(-1, 1, 28, 28)

    # Orijinal ve rekonstrüksiyonları art arda birleştir
    comparison = torch.cat([data[:num_images], recon[:num_images]])
    grid = make_grid(comparison.cpu(), nrow=num_images)

    plt.figure(figsize=(num_images, 2))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()


def plot_latent_space(model, data_loader, device, num_batches=1):
    """
    Latent uzayı 2 boyuta indirgediysek, her verinin mu'sunu scatter plot olarak çizer.
    """
    import numpy as np
    model.eval()
    latents = []
    labels = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            data = data.to(device)
            _, mu, _ = model(data)
            latents.append(mu.cpu().numpy())
            labels.append(target.numpy())

    latents = np.concatenate(latents)
    labels = np.concatenate(labels)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', alpha=0.5)
    plt.colorbar(scatter, ticks=range(10))
    plt.xlabel('Latent Dim 1')
    plt.ylabel('Latent Dim 2')
    plt.show()


def sample_images(model, device, num_samples=16, latent_dim=16):
    """
    Önceden tanımlı prior'dan z örnekleyip decoder ile yeni görüntüler üretir.
    """
    import math
    z = torch.randn(num_samples, latent_dim).to(device)
    with torch.no_grad():
        samples = model.decode(z).cpu()
    samples = samples.view(-1, 1, 28, 28)

    grid = make_grid(samples, nrow=int(math.sqrt(num_samples)), pad_value=1)

    plt.figure(figsize=(int(math.sqrt(num_samples)) * 2, int(math.sqrt(num_samples)) * 2))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()
