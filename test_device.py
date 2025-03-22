import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

print(f"Using device: {device}")
print(f"Model on device: {next(model.parameters()).device}")

if torch.cuda.is_available():
    print("CUDA Version:", torch.version.cuda)
    print("Current GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
