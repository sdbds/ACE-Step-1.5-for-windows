#!/usr/bin/env python3
"""
GPU Detection Diagnostic Tool for ACE-Step

This script helps diagnose GPU detection issues by checking:
- PyTorch installation and build type (CUDA/ROCm/CPU)
- GPU availability and properties
- Environment variables
- Common configuration issues

Usage:
    python scripts/check_gpu.py
"""

import os
import sys
import subprocess

# Constants
HEADER_WIDTH = 80
PYTORCH_CUDA_INSTALL_URL = "https://download.pytorch.org/whl/cu121"
PYTORCH_ROCM_INSTALL_URL = "https://download.pytorch.org/whl/rocm6.0"


def print_section(title):
    """Print a section header."""
    print(f"\n{'=' * HEADER_WIDTH}")
    print(f"  {title}")
    print('=' * HEADER_WIDTH)


def check_pytorch():
    """Check PyTorch installation and build type."""
    print_section("PyTorch Installation")
    
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
        
        # Check build type
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        is_cuda = hasattr(torch.version, 'cuda') and torch.version.cuda is not None
        
        if is_rocm:
            print(f"✓ Build type: ROCm (HIP {torch.version.hip})")
        elif is_cuda:
            print(f"✓ Build type: CUDA {torch.version.cuda}")
        else:
            print("⚠️ Build type: CPU-only")
            print("\n❌ You have installed a CPU-only version of PyTorch!")
            print("\nTo enable GPU support:")
            print("  For NVIDIA GPUs:")
            print(f"    pip install torch --index-url {PYTORCH_CUDA_INSTALL_URL}")
            print("\n  For AMD GPUs with ROCm:")
            print("    Windows: See requirements-rocm.txt")
            print(f"    Linux: pip install torch --index-url {PYTORCH_ROCM_INSTALL_URL}")
            return False
        
        return True
    except ImportError:
        print("❌ PyTorch not installed")
        print("\nPlease install PyTorch first:")
        print("  pip install torch")
        return False


def check_cuda_availability():
    """Check CUDA/ROCm availability."""
    print_section("GPU Availability Check")
    
    try:
        import torch
        
        is_available = torch.cuda.is_available()
        print(f"torch.cuda.is_available(): {is_available}")
        
        if is_available:
            print(f"✓ GPU detected!")
            device_count = torch.cuda.device_count()
            print(f"  Number of GPUs: {device_count}")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"\n  GPU {i}: {device_name}")
                print(f"    Total memory: {memory_gb:.2f} GB")
                print(f"    Compute capability: {props.major}.{props.minor}")
            
            return True
        else:
            print("❌ No GPU detected by PyTorch")
            return False
            
    except Exception as e:
        print(f"❌ Error checking GPU availability: {e}")
        return False


def check_rocm_setup():
    """Check ROCm-specific setup for AMD GPUs."""
    print_section("ROCm Configuration (AMD GPUs)")
    
    try:
        import torch
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        
        if not is_rocm:
            print("Skipping - not a ROCm build")
            return
        
        print("Checking ROCm environment variables:")
        
        # Check HSA_OVERRIDE_GFX_VERSION
        hsa_override = os.environ.get('HSA_OVERRIDE_GFX_VERSION')
        if hsa_override:
            print(f"  ✓ HSA_OVERRIDE_GFX_VERSION = {hsa_override}")
        else:
            print("  ⚠️ HSA_OVERRIDE_GFX_VERSION not set")
            print("\n  This variable is required for many AMD GPUs!")
            print("  Set it according to your GPU:")
            print("    RX 7900 XT/XTX, RX 9070 XT: HSA_OVERRIDE_GFX_VERSION=11.0.0")
            print("    RX 7800 XT, RX 7700 XT: HSA_OVERRIDE_GFX_VERSION=11.0.1")
            print("    RX 7600: HSA_OVERRIDE_GFX_VERSION=11.0.2")
            print("    RX 6000 series: HSA_OVERRIDE_GFX_VERSION=10.3.0")
        
        # Check MIOPEN_FIND_MODE
        miopen_mode = os.environ.get('MIOPEN_FIND_MODE')
        if miopen_mode:
            print(f"  ✓ MIOPEN_FIND_MODE = {miopen_mode}")
        else:
            print("  ℹ️ MIOPEN_FIND_MODE not set (recommended: FAST)")
        
        # Try to run rocm-smi
        print("\nChecking ROCm system management interface:")
        try:
            result = subprocess.run(['rocm-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("  ✓ rocm-smi found and working")
                print("\n  Output (first 10 lines):")
                lines = result.stdout.split('\n')[:10]
                for line in lines:
                    if line.strip():
                        print(f"    {line}")
            else:
                print("  ⚠️ rocm-smi found but returned error")
        except FileNotFoundError:
            print("  ❌ rocm-smi not found in PATH")
            print("     This suggests ROCm is not properly installed")
        except Exception as e:
            print(f"  ⚠️ Error running rocm-smi: {e}")
            
    except ImportError:
        print("❌ PyTorch not installed")


def check_nvidia_setup():
    """Check NVIDIA CUDA setup."""
    print_section("NVIDIA CUDA Configuration")
    
    try:
        import torch
        is_cuda = hasattr(torch.version, 'cuda') and torch.version.cuda is not None
        
        if not is_cuda:
            print("Skipping - not a CUDA build")
            return
        
        print(f"CUDA version in PyTorch: {torch.version.cuda}")
        
        # Try to run nvidia-smi
        print("\nChecking NVIDIA System Management Interface:")
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("  ✓ nvidia-smi found and working")
                print("\n  Output (first 15 lines):")
                lines = result.stdout.split('\n')[:15]
                for line in lines:
                    if line.strip():
                        print(f"    {line}")
            else:
                print("  ⚠️ nvidia-smi found but returned error")
        except FileNotFoundError:
            print("  ❌ nvidia-smi not found in PATH")
            print("     This suggests NVIDIA drivers are not properly installed")
        except Exception as e:
            print(f"  ⚠️ Error running nvidia-smi: {e}")
            
    except ImportError:
        print("❌ PyTorch not installed")


def check_ace_step_env():
    """Check ACE-Step specific environment variables."""
    print_section("ACE-Step Environment Variables")
    
    relevant_vars = [
        'MAX_CUDA_VRAM',
        'HSA_OVERRIDE_GFX_VERSION',
        'MIOPEN_FIND_MODE',
        'TORCH_COMPILE_BACKEND',
        'ACESTEP_LM_BACKEND',
    ]
    
    found_any = False
    for var in relevant_vars:
        value = os.environ.get(var)
        if value:
            print(f"  {var} = {value}")
            found_any = True
    
    if not found_any:
        print("  No ACE-Step specific environment variables set")


def print_recommendations():
    """Print recommendations based on detected issues."""
    print_section("Recommendations")
    
    try:
        import torch
        
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        is_cuda = hasattr(torch.version, 'cuda') and torch.version.cuda is not None
        is_available = torch.cuda.is_available()
        
        if is_available:
            print("✓ Your GPU setup appears to be working correctly!")
            print("\nYou can now run ACE-Step with GPU acceleration.")
        elif is_rocm:
            print("❌ ROCm build detected but GPU not available")
            print("\nTroubleshooting steps for AMD GPUs:")
            print("  1. Set HSA_OVERRIDE_GFX_VERSION for your GPU model (see above)")
            print("  2. Verify ROCm installation with: rocm-smi")
            print("  3. Check that your GPU is supported by your ROCm version")
            print("  4. On Windows: Use start_gradio_ui_rocm.bat which sets all required variables")
            print("  5. On Linux: See docs/en/ACE-Step1.5-Rocm-Manual-Linux.md")
            print("\nFor RX 9070 XT specifically:")
            print("  export HSA_OVERRIDE_GFX_VERSION=11.0.0")
            print("  or on Windows: set HSA_OVERRIDE_GFX_VERSION=11.0.0")
        elif is_cuda:
            print("❌ CUDA build detected but GPU not available")
            print("\nTroubleshooting steps for NVIDIA GPUs:")
            print("  1. Install NVIDIA drivers from https://www.nvidia.com/download/index.aspx")
            print("  2. Verify installation with: nvidia-smi")
            print("  3. Ensure CUDA version compatibility between driver and PyTorch")
        else:
            print("❌ CPU-only PyTorch build detected")
            print("\nYou need to reinstall PyTorch with GPU support:")
            print("\nFor NVIDIA GPUs:")
            print("  pip uninstall torch torchvision torchaudio")
            print(f"  pip install torch torchvision torchaudio --index-url {PYTORCH_CUDA_INSTALL_URL}")
            print("\nFor AMD GPUs:")
            print("  Windows: Follow instructions in requirements-rocm.txt")
            print(f"  Linux: pip install torch --index-url {PYTORCH_ROCM_INSTALL_URL}")
            
    except ImportError:
        print("❌ PyTorch not installed")
        print("\nPlease install PyTorch first. See README.md for instructions.")


def main():
    """Main diagnostic routine."""
    print("=" * HEADER_WIDTH)
    print("  ACE-Step GPU Detection Diagnostic Tool")
    print("=" * HEADER_WIDTH)
    print("\nThis tool will help diagnose GPU detection issues.")
    print("Please share the output with support when reporting issues.")
    
    # Run all checks
    pytorch_ok = check_pytorch()
    
    if pytorch_ok:
        gpu_ok = check_cuda_availability()
        check_rocm_setup()
        check_nvidia_setup()
        check_ace_step_env()
    
    print_recommendations()
    
    print("\n" + "=" * HEADER_WIDTH)
    print("  Diagnostic Complete")
    print("=" * HEADER_WIDTH)


if __name__ == "__main__":
    main()
