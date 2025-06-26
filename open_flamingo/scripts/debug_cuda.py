#!/usr/bin/env python3
"""
CUDA and NCCL diagnostic script to help troubleshoot distributed training issues.
"""

import torch
import torch.distributed as dist
import os
import sys
import subprocess
import time

def check_cuda_setup():
    """Check basic CUDA setup"""
    print("=== CUDA Setup Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA not available!")
        return False
    
    return True

def check_nccl():
    """Check NCCL availability"""
    print("\n=== NCCL Check ===")
    try:
        # Try to import NCCL
        import torch.distributed as dist
        print("torch.distributed imported successfully")
        
        # Check if NCCL backend is available
        if dist.is_nccl_available():
            print("NCCL backend is available")
        else:
            print("NCCL backend is NOT available")
            return False
            
    except Exception as e:
        print(f"Error importing torch.distributed: {e}")
        return False
    
    return True

def test_gpu_memory():
    """Test GPU memory allocation"""
    print("\n=== GPU Memory Test ===")
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return False
    
    try:
        for i in range(torch.cuda.device_count()):
            device = f"cuda:{i}"
            print(f"Testing GPU {i} ({device})")
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Test small allocation
            x = torch.randn(1000, 1000, device=device)
            print(f"  Small allocation: OK")
            
            # Test larger allocation
            y = torch.randn(5000, 5000, device=device)
            print(f"  Large allocation: OK")
            
            # Test computation
            z = torch.matmul(x[:1000, :1000], y[:1000, :1000])
            print(f"  Computation: OK")
            
            # Memory info
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            cached = torch.cuda.memory_reserved(device) / 1024**3
            print(f"  Memory allocated: {allocated:.2f} GB")
            print(f"  Memory cached: {cached:.2f} GB")
            
            # Clean up
            del x, y, z
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"GPU memory test failed: {e}")
        return False
    
    return True

def test_simple_distributed():
    """Test simple distributed setup"""
    print("\n=== Simple Distributed Test ===")
    
    # Set environment variables for single-node multi-GPU
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    
    try:
        # Initialize process group
        dist.init_process_group(backend='nccl', timeout=60)
        print("Process group initialized successfully")
        
        # Clean up
        dist.destroy_process_group()
        print("Process group destroyed successfully")
        
    except Exception as e:
        print(f"Distributed test failed: {e}")
        return False
    
    return True

def check_environment():
    """Check relevant environment variables"""
    print("\n=== Environment Variables ===")
    
    relevant_vars = [
        'CUDA_VISIBLE_DEVICES',
        'NCCL_DEBUG',
        'NCCL_IB_DISABLE',
        'NCCL_P2P_DISABLE',
        'NCCL_SHM_DISABLE',
        'PYTORCH_CUDA_ALLOC_CONF',
        'WORLD_SIZE',
        'RANK',
        'LOCAL_RANK',
        'MASTER_ADDR',
        'MASTER_PORT'
    ]
    
    for var in relevant_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")

def run_nvidia_smi():
    """Run nvidia-smi to check GPU status"""
    print("\n=== nvidia-smi output ===")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except Exception as e:
        print(f"Failed to run nvidia-smi: {e}")

def main():
    print("CUDA/NCCL Diagnostic Tool")
    print("=" * 50)
    
    # Run all checks
    checks = [
        ("CUDA Setup", check_cuda_setup),
        ("NCCL", check_nccl),
        ("GPU Memory", test_gpu_memory),
        ("Simple Distributed", test_simple_distributed),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"Error in {name} check: {e}")
            results[name] = False
    
    # Show environment and nvidia-smi
    check_environment()
    run_nvidia_smi()
    
    # Summary
    print("\n=== Summary ===")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status}")
    
    # Recommendations
    print("\n=== Recommendations ===")
    if not results.get("CUDA Setup", False):
        print("- CUDA is not properly set up. Check CUDA installation and drivers.")
    
    if not results.get("NCCL", False):
        print("- NCCL is not available. Reinstall PyTorch with CUDA support.")
    
    if not results.get("GPU Memory", False):
        print("- GPU memory issues detected. Try:")
        print("  * Reduce batch size")
        print("  * Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128")
        print("  * Check for other processes using GPU memory")
    
    if not results.get("Simple Distributed", False):
        print("- Distributed setup failed. Try:")
        print("  * Set NCCL_IB_DISABLE=1")
        print("  * Set NCCL_P2P_DISABLE=1")
        print("  * Use fewer GPUs initially")
        print("  * Check firewall settings")

if __name__ == "__main__":
    main()
