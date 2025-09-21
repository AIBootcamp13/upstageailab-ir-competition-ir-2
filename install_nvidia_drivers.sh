#!/bin/bash

echo "🚀 Installing NVIDIA drivers and CUDA toolkit for WSL2..."

# Update package list
echo "📦 Updating package list..."
sudo apt update

# Install prerequisites
echo "🔧 Installing prerequisites..."
sudo apt install -y wget gnupg2 software-properties-common

# Add NVIDIA repository
echo "📥 Adding NVIDIA repository..."
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Update package list again
sudo apt-get update

# Install CUDA toolkit
echo "⚡ Installing CUDA toolkit..."
sudo apt-get install -y cuda-toolkit-12-4

# Install NVIDIA driver packages for WSL2
echo "🎮 Installing NVIDIA driver packages..."
sudo apt-get install -y nvidia-driver-550

# Add CUDA to PATH
echo "🔗 Configuring CUDA environment..."
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Source the updated bashrc
source ~/.bashrc

echo "✅ Installation complete!"
echo ""
echo "🔍 To verify installation, run:"
echo "   nvidia-smi"
echo "   nvcc --version"
echo ""
echo "📝 Note: Make sure NVIDIA drivers are installed on your Windows host first!"
