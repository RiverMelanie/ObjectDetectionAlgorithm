import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print("CPU设备:", torch.device('cpu'))
if torch.cuda.is_available():
    print("GPU设备:", torch.device('cuda'))