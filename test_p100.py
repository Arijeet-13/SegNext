import torch
import torch.nn as nn

print("Starting P100 isolation test...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device('cuda')

# Create a tiny standard convolution that has NOTHING to do with SegNext
print("Creating a 1-layer convolution matrix...")
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3).to(device)

print("Creating native PyTorch random tensor...")
dummy_tensor = torch.randn(1, 3, 256, 256).to(device)

print("Attempting to run standard PyTorch calculation on your GPU...")
try:
    output = conv(dummy_tensor)
    print("SUCCESS! PyTorch successfully executed the cuda kernel.")
except Exception as e:
    print(f"\nCRASH DETECTED. PyTorch itself cannot communicate with the P100.")
    print(f"Exact error:\n{e}")
