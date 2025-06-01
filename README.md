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

Model Mimarisi (ConvVAE)
Encoder

Conv2d(1, 32, kernel_size=4, stride=2, padding=1) → 28×28 → 14×14

Conv2d(32, 64, kernel_size=4, stride=2, padding=1) → 14×14 → 7×7

Conv2d(64, 128, kernel_size=4, stride=2, padding=1) → 7×7 → 3×3

Flatten edilmiş çıktı (128 × 3 × 3 = 1152 boyut) iki ayrı tam bağlı katmana (FC) beslenir:

μ(x): Linear(128*3*3, latent_dim)

logσ²(x): Linear(128*3*3, latent_dim)

Reparameterization Trick

𝑧
=
𝜇
(
𝑥
)
+
exp
⁡
(
1
2
log
⁡
𝜎
2
(
𝑥
)
)
⊙
𝜖
,
𝜖
∼
𝑁
(
0
,
𝐼
)
z=μ(x)+exp( 
2
1
​
 logσ 
2
 (x))⊙ϵ,ϵ∼N(0,I)
Decoder

Linear(latent_dim, 128*3*3) → yeniden şekillendirme: 128 × 3 × 3

ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1) → 3×3 → 7×7

ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) → 7×7 → 14×14

ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1) → 14×14 → 28×28

Son katmanda Sigmoid aktivasyonu kullanılarak çıktı [0, 1] aralığında elde edilir.
