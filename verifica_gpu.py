import torch

# Verificar se a GPU está disponível
num_gpus = torch.cuda.device_count()
print("Num GPUs Available: ", num_gpus)

if num_gpus > 0:
    try:
        # Configurar o PyTorch para usar a GPU
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        device = torch.device('cuda')
        print("PyTorch configurado para usar a GPU")
    except RuntimeError as e:
        print(e)
else:
    device = torch.device('cpu')
    print("No GPU available, using CPU")