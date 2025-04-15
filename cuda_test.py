import torch

print("🔍 CUDA available:", torch.cuda.is_available())
print("💻 Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("🧠 CUDA version (PyTorch):", torch.version.cuda)
print("🐍 Torch compiled with CUDA:", torch.backends.cuda.is_built())
