# Salva um arquvio de profilling, na pasta /logdir/full_run
# Possível acessá-lo via chrome://tracing/
# Também é possível fazer o profiling lançando 
# >> tensorboard --logdir=log_mnist
# e acessando via http://localhost:6006/

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
from torch.profiler import profile, record_function, ProfilerActivity

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

# Inicializar modelo e otimizador
model = AutoencoderClassifier().to(device)
optimizer = Adam(model.parameters(), lr=0.001)

# Pesos das perdas
lambda_recon = 0.5
lambda_class = 1.0

# Profiling do código inteiro
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logdir/full_run'),
    record_shapes=True,
    with_stack=True
) as prof:

    # 🔁 Treinamento
    train_losses = []
    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        last_log_time = time.time()
        total_images = 0

        print(f"\n🔁 Epoch {epoch+1}/{num_epochs}")

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)

            with record_function("forward_backward"):
                x_recon, logits = model(images)
                loss_recon = F.mse_loss(x_recon, images.view(-1, 28*28))
                loss_class = F.cross_entropy(logits, labels)
                loss = lambda_recon * loss_recon + lambda_class * loss_class

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            total_images += images.size(0)

            now = time.time()
            if now - last_log_time >= 5:
                print(f"[LOG] {total_images}/{len(train_dataset)} imagens processadas...")
                last_log_time = now

            prof.step()  # avança o profiler em cada batch

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"→ Epoch {epoch+1} finalizada | Loss médio: {avg_loss:.4f}")

    # 🧪 Avaliação
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
            prof.step()

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ Acurácia no conjunto de teste: {acc:.4f}")

    # 🎯 Visualização da Reconstrução
    model.eval()
    with torch.no_grad():
        sample_images, _ = next(iter(test_loader))
        sample_images = sample_images.to(device)
        recon, _ = model(sample_images[:8])
        recon = recon.view(-1, 1, 28, 28).cpu()
        sample_images = sample_images[:8].cpu()
        prof.step()

# 🔍 Visualização da imagem reconstruída
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
