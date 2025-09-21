#!/bin/bash

echo "🧪 Testing GPU access in container..."

echo "1. Checking NVIDIA devices:"
ls -la /dev/nvidia* 2>/dev/null || echo "   No NVIDIA devices found"

echo ""
echo "2. Testing nvidia-smi:"
nvidia-smi || echo "   nvidia-smi not available"

echo ""
echo "3. Testing PyTorch CUDA:"
python -c "
import torch
print('   PyTorch version:', torch.__version__)
print('   CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('   CUDA device count:', torch.cuda.device_count())
    print('   CUDA device name:', torch.cuda.get_device_name(0))
    print('   CUDA version:', torch.version.cuda)
else:
    print('   CUDA not available - check GPU configuration')
"

echo ""
echo "4. Testing CUDA runtime:"
python -c "
try:
    import torch
    if torch.cuda.is_available():
        # Test basic CUDA operations
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print('   CUDA tensor operations: SUCCESS')
        print('   GPU memory allocated:', torch.cuda.memory_allocated()/1024/1024, 'MB')
    else:
        print('   CUDA not available')
except Exception as e:
    print('   CUDA test failed:', str(e))
"