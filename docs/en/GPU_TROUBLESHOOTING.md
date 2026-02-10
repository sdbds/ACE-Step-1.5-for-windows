# GPU Detection Troubleshooting Guide

This guide helps resolve "No GPU detected, running on CPU" errors.

## Quick Diagnostic

Run the diagnostic tool to identify your issue:

```bash
python scripts/check_gpu.py
```

This will check your PyTorch installation, GPU availability, and environment configuration.

## Common Issues and Solutions

### Issue 1: AMD GPU Not Detected (ROCm)

**Symptoms:**
- You have an AMD GPU (RX 6000/7000/9000 series)
- ROCm is installed
- Still getting "No GPU detected"

**Solution:**

#### For RDNA3 GPUs (RX 7000/9000 series):

The `HSA_OVERRIDE_GFX_VERSION` environment variable is required:

**Linux/macOS:**
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # For RX 7900 XT/XTX, RX 9070 XT
export HSA_OVERRIDE_GFX_VERSION=11.0.1  # For RX 7800 XT, RX 7700 XT
export HSA_OVERRIDE_GFX_VERSION=11.0.2  # For RX 7600
```

**Windows:**
```cmd
set HSA_OVERRIDE_GFX_VERSION=11.0.0
```

Or use the provided launcher scripts which set this automatically:
```cmd
start_gradio_ui_rocm.bat
start_api_server_rocm.bat
```

#### For RDNA2 GPUs (RX 6000 series):

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # Linux/macOS
set HSA_OVERRIDE_GFX_VERSION=10.3.0     # Windows
```

#### Verify ROCm Installation:

```bash
# Check if ROCm can see your GPU
rocm-smi

# Check PyTorch ROCm build
python -c "import torch; print(f'ROCm: {torch.version.hip}')"
```

### Issue 2: CPU-Only PyTorch Installed

**Symptoms:**
- Diagnostic shows "Build type: CPU-only"

**Solution:**

You need to reinstall PyTorch with GPU support.

#### For AMD GPUs:

**Windows (ROCm 7.2):**
Follow the detailed instructions in `requirements-rocm.txt`:

```cmd
# 1. Install ROCm SDK components (see requirements-rocm.txt for full URLs)
pip install --no-cache-dir [ROCm SDK wheels...]

# 2. Install PyTorch for ROCm
pip install --no-cache-dir [PyTorch ROCm wheel...]

# 3. Install dependencies
pip install -r requirements-rocm.txt
```

**Linux (ROCm 6.0+):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
pip install -r requirements-rocm-linux.txt
```

#### For NVIDIA GPUs:

```bash
# For CUDA 12.1 (check PyTorch website for latest version)
pip install torch --index-url https://download.pytorch.org/whl/cu121
# Or for CUDA 12.4+:
# pip install torch --index-url https://download.pytorch.org/whl/cu124
```

> **Note:** Check https://pytorch.org/get-started/locally/ for the latest CUDA version supported by PyTorch.

### Issue 3: NVIDIA GPU Not Detected (CUDA)

**Symptoms:**
- You have an NVIDIA GPU
- CUDA is installed
- Still getting "No GPU detected"

**Solution:**

1. **Check NVIDIA drivers:**
   ```bash
   nvidia-smi
   ```
   
   If this fails, install/update NVIDIA drivers from: https://www.nvidia.com/download/index.aspx

2. **Check CUDA version compatibility:**
   
   The CUDA version in your PyTorch build must be compatible with your driver.
   
   Check PyTorch CUDA version:
   ```bash
   python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
   ```
   
   Check driver CUDA version:
   ```bash
   nvidia-smi  # Look for "CUDA Version: X.X"
   ```

3. **Reinstall PyTorch if needed:**
   ```bash
   pip uninstall torch torchvision torchaudio
   # Check https://pytorch.org/get-started/locally/ for the latest CUDA version
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### Issue 4: WSL2 GPU Access Issues

**Symptoms:**
- Running in WSL2 (Windows Subsystem for Linux)
- GPU not detected

**Solution:**

For NVIDIA GPUs in WSL2, you need CUDA on WSL2:
1. Install NVIDIA drivers on Windows (not in WSL2)
2. Install CUDA toolkit in WSL2
3. Follow: https://docs.nvidia.com/cuda/wsl-user-guide/index.html

For AMD GPUs, ROCm support in WSL2 is limited. Consider:
- Running on native Linux
- Using Windows with `start_gradio_ui_rocm.bat` / `start_api_server_rocm.bat`

## GPU-Specific Configuration

### RX 9070 XT (RDNA3)

```bash
# Linux/macOS
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export MIOPEN_FIND_MODE=FAST

# Windows (or use start_gradio_ui_rocm.bat / start_api_server_rocm.bat)
set HSA_OVERRIDE_GFX_VERSION=11.0.0
set MIOPEN_FIND_MODE=FAST
```

### RX 7900 XT/XTX (RDNA3)

Same as RX 9070 XT above.

### RX 6900 XT (RDNA2)

```bash
# Linux/macOS
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Windows
set HSA_OVERRIDE_GFX_VERSION=10.3.0
```

## Additional Resources

- **ROCm Linux Setup:** See `docs/en/ACE-Step1.5-Rocm-Manual-Linux.md`
- **ROCm Windows Setup:** See `requirements-rocm.txt`
- **GPU Tiers:** See `docs/en/GPU_COMPATIBILITY.md`
- **General Installation:** See `README.md`

## Still Having Issues?

If none of the above solutions work:

1. Run the diagnostic tool and save the output:
   ```bash
   python scripts/check_gpu.py > gpu_diagnostic.txt
   ```

2. Open an issue on GitHub with:
   - The diagnostic output
   - Your GPU model
   - Your OS (Windows/Linux/macOS)
   - ROCm/CUDA version installed

## Environment Variables Reference

### ROCm (AMD GPUs)

| Variable | Purpose | Example |
|----------|---------|---------|
| `HSA_OVERRIDE_GFX_VERSION` | Override GPU architecture | `11.0.0` (RDNA3), `10.3.0` (RDNA2) |
| `MIOPEN_FIND_MODE` | MIOpen kernel selection mode | `FAST` (recommended) |
| `TORCH_COMPILE_BACKEND` | PyTorch compilation backend | `eager` (ROCm Windows) |
| `ACESTEP_LM_BACKEND` | Language model backend | `pt` (recommended for ROCm) |

### CUDA (NVIDIA GPUs)

| Variable | Purpose | Example |
|----------|---------|---------|
| `CUDA_VISIBLE_DEVICES` | Select which GPU to use | `0` (first GPU) |

### ACE-Step Specific

| Variable | Purpose | Example |
|----------|---------|---------|
| `MAX_CUDA_VRAM` | Override detected VRAM (testing) | `8` (simulate 8GB GPU) |
