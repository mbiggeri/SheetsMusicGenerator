import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA is available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(f"CUDA Version (PyTorch sees): {torch.version.cuda}")