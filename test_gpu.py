import torch

print("PyTorch version:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Número de GPUs disponíveis:", torch.cuda.device_count())
    print("Nome da GPU:", torch.cuda.get_device_name(0))
    print("Memória disponível (MB):", round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 2))
else:
    print("CUDA não está disponível. O Python está rodando apenas na CPU.")
