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

## Kayıp Fonksiyonu

### Reconstruction Loss (Binary Cross-Entropy)
\[
\mathrm{BCE} = -\sum_{i=1}^{N} \bigl[x_i \,\log \hat{x}_i + (1 - x_i)\,\log\bigl(1 - \hat{x}_i\bigr)\bigr], \quad (\text{reduction='sum'})
\]

```python
import torch.nn.functional as F

# recon_x: model.decode(z) çıktısı (28×28 boyutunda, sigmoid ile [0,1] aralığında)
# x: orijinal giriş görüntüsü
BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")

### Kullback–Leibler Divergence (KLD)

\[
\mathrm{KLD} = -\frac{1}{2} \sum_{j=1}^{d} \bigl(1 + \log(\sigma_j^2)\;-\;\mu_j^2\;-\;\sigma_j^2 \bigr)
\]
Burada:
- \(d = \text{latent\_dim}\)  
- \(\mu_j\) ve \(\sigma_j^2 = \exp(\log\sigma_j^2)\) sırasıyla encoder’dan çıkan ortalama ve log varyans parametreleridir.

```python
# mu: [batch, latent_dim] boyutunda ortalama vektörü
# logvar: [batch, latent_dim] boyutunda log varyans vektörü
KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

### Toplam Loss

\[
\mathcal{L}(x) = \underbrace{\mathrm{BCE}(x, \hat{x})}_{\text{Reconstruction}} 
\;+\; 
\underbrace{\mathrm{KLD}\bigl(q(z \mid x)\,\|\,\mathcal{N}(0, I)\bigr)}_{\text{Regularization}}
\]

```python
import torch.nn.functional as F

def loss_function(recon_x, x, mu, logvar):
    # Reconstruction (Binary Cross-Entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    # Kullback–Leibler Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

### Eğitim Prosedürü

#### Parametreler
- `batch_size = 128`
- `epochs = 100`
- `learning_rate = 5e-4`
- Optimizer: `Adam(model.parameters(), lr=5e-4)`

#### Eğitim Döngüsü
```python
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvVAE(latent_dim=32).to(device)
optimizer = Adam(model.parameters(), lr=5e-4)

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss = loss_function(recon, data, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader.dataset)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon, mu, logvar = model(data)
            test_loss += loss_function(recon, data, mu, logvar).item()
    avg_test_loss = test_loss / len(test_loader.dataset)

    print(f"Epoch {epoch}: Train loss {avg_train_loss:.2f}, Test loss {avg_test_loss:.2f}")
Downstream Görev: Latent Özelliklerle Sınıflandırma
Eğitilmiş ConvVAE modelinin encoder’ından çıkarılan 
𝜇
(
𝑥
)
∈
𝑅
32
μ(x)∈R 
32
  boyutlu latent vektörler,
özellik (feature) olarak RandomForestClassifier içine beslenir.

Adımlar:

Eğitilmiş modeli yükleme:
```python
model = ConvVAE(latent_dim=32).to(device)
model.load_state_dict(torch.load("src/checkpoints/convvae_fashionmnist.pth", map_location=device))
model.eval()
'''
Latent Çıkarımı:
```python
import numpy as np
from torch.utils.data import DataLoader

def extract_latent(loader):
    feats, labels = [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            feats.append(mu.cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(feats), np.concatenate(labels)

X_train, y_train = extract_latent(train_loader)
X_test,  y_test  = extract_latent(test_loader)
'''
Random Forest Eğitimi ve Değerlendirme:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Classification accuracy: {acc*100:.2f}%")
cm = confusion_matrix(y_test, preds)
'''
Performans:

Latent boyut = 16 ile denemede %84.01 doğruluk elde edildi.

Latent boyut = 32’de benzer veya daha yüksek doğruluk beklenir.

Results
Loss Curves

Mavi: Eğitim kaybı, Turuncu: Test kaybı

epoch’ta ≈ 308; sonraki epoch’larda ∼ 236 civarına indi

100 epoch sonunda plateau, eğitim ve test kayıpları birbirine yakın (overfitting az)

Rekonstrüksiyon Örnekleri

Üst satır: Test setten alınan orijinal resimler

Alt satır: ConvVAE’nin ürettiği rekonstrüksiyonlar

Kenar detayları belirginleşmiş; bulanıklık büyük ölçüde azalmış

Örnek Üretim (Samples)

Latent uzaydan rastgele 
𝑧
∼
𝑁
(
0
,
𝐼
)
z∼N(0,I) çekilip decoder’dan geçirilerek üretildi

“Sneaker”, “Pullover” vb. sınıf hatları seçilebiliyor, ancak ince detaylar hâlâ biraz net değil

Latent Embedding Sınıflandırma

Doğruluk: %84.01 (latent_dim=16)

Sınıflandırıcı: Random Forest (n_estimators=100)

Karmaşıklık Matrisi
Aşağıdaki tablo, test set üzerindeki sınıflandırma sonuçlarının karşılaştırmalı karmaşıklık matrisini göstermektedir (satırlar gerçek sınıflar, sütunlar tahmin edilen sınıflar):
| True\Pred | Pred_0 | Pred_1 | Pred_2 | Pred_3 | Pred_4 | Pred_5 | Pred_6 | Pred_7 | Pred_8 | Pred_9 | Total |
|-----------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-------|
| **True_0**    | 841    | 1      | 1      | 2      | 2      | 4      | 1      | 2      | 1      | 3      | 858   |
| **True_1**    | 1      | 927    | 0      | 5      | 0      | 1      | 3      | 2      | 5      | 2      | 946   |
| **True_2**    | 5      | 1      | 794    | 7      | 6      | 4      | 8      | 8      | 19     | 3      | 855   |
| **True_3**    | 5      | 10     | 9      | 843    | 6      | 16     | 5      | 4      | 18     | 6      | 922   |
| **True_4**    | 2      | 3      | 1      | 3      | 838    | 4      | 4      | 5      | 18     | 4      | 882   |
| **True_5**    | 8      | 1      | 1      | 16     | 2      | 684    | 11     | 33     | 10     | 22     | 788   |
| **True_6**    | 7      | 3      | 7      | 1      | 7      | 4      | 850    | 1      | 5      | 2      | 893   |
| **True_7**    | 0      | 3      | 4      | 2      | 4      | 19     | 0      | 893    | 2      | 2      | 929   |
| **True_8**    | 1      | 13     | 6      | 11     | 17     | 11     | 5      | 5      | 812    | 20     | 901   |
| **True_9**    | 4      | 5      | 0      | 3      | 0      | 19     | 1      | 3      | 9      | 913    | 957   |
| **Total** | 874    | 967    | 813    | 892    | 882    | 750    | 888    | 961    | 880    | 997    | 9123  |

Discussion
Başarılar

Convolutional mimari, tam bağlantılı VAE’ye kıyasla çok daha net rekonstrüksiyonlar sundu.

Latent kodlar (%84 doğruluk) downstream görevlerde anlamlı temsil oluşturdu.

Kısıtlar ve İyileştirme Alanları

Daha Derin Mimari

Ek Conv katmanları (ör. 256 filtre), BatchNorm/LeakyReLU eklenebilir.

Eğitim Teknikleri

β-VAE (“β>1”) ile KLD’ye ağırlık vererek latent uzayın düzenini iyileştirme.

Öğrenme Oranı Planlayıcıları: ReduceLROnPlateau veya CosineAnnealing eklenebilir.

Epoch sayısını 150–200’e çıkarmak uzun vadeli iyileşmeler sunabilir.

Görsel Kayıp Terimleri

Perceptual Loss (VGG tabanlı) veya Feature Matching ekleyerek daha net çıktı sağlama.

Latent Boyut

32→64/128 deneyerek latent uzayın kapasitesini artırma; 2D görselleştirme için ayrı bir modelle t-SNE/PCA analizi yapılabilir.

References
D. P. Kingma & M. Welling, “Auto-Encoding Variational Bayes,” ICLR 2014.

Fashion-MNIST dataset, https://github.com/zalandoresearch/fashion-mnist

PyTorch dokümantasyonu: https://pytorch.org/docs/stable/


