import torch.profiler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np

# 📌 Usa GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# Modelo
class AutoencoderClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        logits = self.classifier(z)
        return x_recon, logits

# Dados
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

# Inicializar modelo e enviar para GPU
model = AutoencoderClassifier().to(device)
optimizer = Adam(model.parameters(), lr=0.001)

# Pesos das perdas
lambda_recon = 0.5
lambda_class = 1.0

# Treinamento com logs
num_epochs = 5
train_losses = []

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log_mnist"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs, logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            prof.step()  # ← chama o profiler a cada batch

        avg_loss = running_loss / len(train_loader)
        print(f"→ Epoch {epoch+1} finalizada | Loss médio: {avg_loss:.4f}")

# Avaliação
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        _, logits = model(images)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
print(f"\n✅ Acurácia no conjunto de teste: {acc:.4f}")

# Reconstrução visual
model.eval()
with torch.no_grad():
    sample_images, _ = next(iter(test_loader))
    sample_images = sample_images.to(device)
    recon, _ = model(sample_images[:8])
    recon = recon.view(-1, 1, 28, 28).cpu()
    sample_images = sample_images[:8].cpu()

fig, axs = plt.subplots(2, 8, figsize=(12, 3))
for i in range(8):
    axs[0, i].imshow(sample_images[i][0], cmap='gray')
    axs[0, i].axis('off')
    axs[1, i].imshow(recon[i][0], cmap='gray')
    axs[1, i].axis('off')
axs[0, 0].set_title('Original')
axs[1, 0].set_title('Reconstrução')
plt.suptitle("Reconstrução com Autoencoder")
plt.show()
