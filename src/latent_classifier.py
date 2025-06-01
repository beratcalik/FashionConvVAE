# src/latent_classifier.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import VAE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os

# 1. Cihaz ve veri yükleyiciler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.FashionMNIST("data/fashion-mnist", train=True, download=True, transform=transform)
test_ds  = datasets.FashionMNIST("data/fashion-mnist", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False)

# 2. Modeli yükleme (encoder kısmını kullanacağız)
latent_dim = 16
model = VAE(latent_dim=latent_dim).to(device)
checkpoint_path = "checkpoints/vae_fashionmnist.pth"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Model checkpoint yüklendi.")
model.eval()

# 3. Latent özellikleri çıkaran fonksiyon
def extract_latent(loader):
    feats, labs = [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            mu, _ = model.encode(data.view(-1, 28*28))
            feats.append(mu.cpu().numpy())
            labs.append(target.numpy())
    return np.concatenate(feats), np.concatenate(labs)

X_train, y_train = extract_latent(train_loader)
X_test,  y_test  = extract_latent(test_loader)

# 4. Random Forest sınıflandırıcısını eğit ve değerlendir
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

acc = accuracy_score(y_test, preds)
print(f"Latent temsille sınıflandırma doğruluğu: {acc*100:.2f}%")
