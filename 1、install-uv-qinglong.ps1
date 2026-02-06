Set-Location $PSScriptRoot

$Env:HF_HOME = "huggingface"
$Env:HF_ENDPOINT="https://hf-mirror.com"
$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_NO_CACHE_DIR = 1
#$Env:PIP_INDEX_URL="https://pypi.mirrors.ustc.edu.cn/simple"
$Env:UV_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple/"
$Env:UV_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu130"
$Env:UV_CACHE_DIR = "${env:LOCALAPPDATA}/uv/cache"
$Env:UV_NO_BUILD_ISOLATION = "1"
$Env:UV_NO_CACHE = "0"
$Env:UV_LINK_MODE = "symlink"
$Env:GIT_LFS_SKIP_SMUDGE = 1
$Env:CUDA_HOME = "${env:CUDA_PATH}"

function InstallFail {
    Write-Output "Install failed|安装失败。"
    Read-Host | Out-Null ;
    Exit
}

function Check {
    param (
        $ErrorInfo
    )
    if (!($?)) {
        Write-Output $ErrorInfo
        InstallFail
    }
}

try {
    ~/.local/bin/uv --version
    Write-Output "uv installed|UV模块已安装."
}
catch {
    Write-Output "Installing uv|安装uv模块中..."
    if ($Env:OS -ilike "*windows*") {
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        Check "uv install failed|安装uv模块失败。"
    }
    else {
        curl -LsSf https://astral.sh/uv/install.sh | sh
        Check "uv install failed|安装uv模块失败。"
    }
}

if ($env:OS -ilike "*windows*") {
    chcp 65001
    # First check if UV cache directory already exists
    if (Test-Path -Path "${env:LOCALAPPDATA}/uv/cache") {
        Write-Host "UV cache directory already exists, skipping disk space check"
    }
    else {
        # Check C drive free space with error handling
        try {
            $CDrive = Get-WmiObject Win32_LogicalDisk -Filter "DeviceID='C:'" -ErrorAction Stop
            if ($CDrive) {
                $FreeSpaceGB = [math]::Round($CDrive.FreeSpace / 1GB, 2)
                Write-Host "C: drive free space: ${FreeSpaceGB}GB"
                
                # $Env:UV cache directory based on available space
                if ($FreeSpaceGB -lt 10) {
                    Write-Host "Low disk space detected. Using local .cache directory"
                    $Env:UV_CACHE_DIR = ".cache"
                } 
            }
            else {
                Write-Warning "C: drive not found. Using local .cache directory"
                $Env:UV_CACHE_DIR = ".cache"
            }
        }
        catch {
            Write-Warning "Failed to check disk space: $_. Using local .cache directory"
            $Env:UV_CACHE_DIR = ".cache"
        }
    }
    if (Test-Path "./venv/Scripts/activate") {
        Write-Output "Windows venv"
        . ./venv/Scripts/activate
    }
    elseif (Test-Path "./.venv/Scripts/activate") {
        Write-Output "Windows .venv"
        . ./.venv/Scripts/activate
    }
    else {
        Write-Output "Create .venv"
        ~/.local/bin/uv venv -p 3.11 --seed
        . ./.venv/Scripts/activate
    }
}
elseif (Test-Path "./venv/bin/activate") {
    Write-Output "Linux venv"
    . ./venv/bin/Activate.ps1
}
elseif (Test-Path "./.venv/bin/activate") {
    Write-Output "Linux .venv"
    . ./.venv/bin/activate.ps1
}
else {
    Write-Output "Create .venv"
    ~/.local/bin/uv venv -p 3.11 --seed
    . ./.venv/bin/activate.ps1
}

Write-Output "Installing main requirements"

~/.local/bin/uv pip install --upgrade wheel_stub torch==2.10.0 psutil hatchling editables

if ($env:OS -ilike "*windows*") {
    ~/.local/bin/uv pip install -r requirements.txt --index-strategy unsafe-best-match
    Check "Install main requirements failed"
}
else {
    ~/.local/bin/uv pip install -r requirements.txt --index-strategy unsafe-best-match
    Check "Install main requirements failed"
}

# ============= Step 1: Select DiT Model | 第一步：选择 DiT 模型 =====================
Write-Output "`n=== DiT 模型下载 / DiT Model Download ==="
$dit_choice = Read-Host "请选择要下载的 DiT 模型 [1/2/3/4/5/6/a/n] (默认为 n)
1: acestep-v15-turbo
2: acestep-v15-base
3: acestep-v15-sft
4: acestep-v15-turbo-shift1
5: acestep-v15-turbo-shift3
6: acestep-v15-turbo-continuous
a: 下载全部 DiT 模型 / Download all DiT models
n: 跳过 DiT 下载 / Skip DiT download
Please select DiT model [1/2/3/4/5/6/a/n] (default is n)"

$dit_models = @()
if ($dit_choice -eq "1") {
    $dit_models += "acestep-v15-turbo"
}
elseif ($dit_choice -eq "2") {
    $dit_models += "acestep-v15-base"
}
elseif ($dit_choice -eq "3") {
    $dit_models += "acestep-v15-sft"
}
elseif ($dit_choice -eq "4") {
    $dit_models += "acestep-v15-turbo-shift1"
}
elseif ($dit_choice -eq "5") {
    $dit_models += "acestep-v15-turbo-shift3"
}
elseif ($dit_choice -eq "6") {
    $dit_models += "acestep-v15-turbo-continuous"
}
elseif ($dit_choice -eq "a") {
    Write-Output "将下载全部 DiT 模型 / Will download all DiT models"
    $dit_models += "acestep-v15-base"
    $dit_models += "acestep-v15-sft"
    $dit_models += "acestep-v15-turbo-shift1"
    $dit_models += "acestep-v15-turbo-shift3"
    $dit_models += "acestep-v15-turbo-continuous"
}
elseif ($dit_choice -eq "" -or $dit_choice -eq "n") {
    Write-Output "跳过 DiT 模型下载 / Skipping DiT model download"
}

# ============= Step 2: Select LM Model | 第二步：选择 LM 模型 =====================
Write-Output "`n=== LM 模型下载 / LM Model Download ==="
$lm_choice = Read-Host "请选择要下载的 LM 模型 [1/2/3/a/n] (默认为 n)
1: acestep-5Hz-lm-1.7B
2: acestep-5Hz-lm-0.6B (低显存 / Low VRAM)
3: acestep-5Hz-lm-4B (高质量 / High Quality)
a: 下载全部 LM 模型 / Download all LM models
n: 跳过 LM 下载 / Skip LM download
Please select LM model [1/2/3/a/n] (default is n)"

$lm_models = @()
if ($lm_choice -eq "1") {
    Write-Output "选择了 acestep-5Hz-lm-1.7B"
    $lm_models += "acestep-5Hz-lm-1.7B"
    # 1.7B is included in main model, will be downloaded automatically
}
elseif ($lm_choice -eq "2") {
    $lm_models += "acestep-5Hz-lm-0.6B"
}
elseif ($lm_choice -eq "3") {
    $lm_models += "acestep-5Hz-lm-4B"
}
elseif ($lm_choice -eq "a") {
    Write-Output "将下载全部 LM 模型 / Will download all LM models"
    $lm_models += "acestep-5Hz-lm-0.6B"
    $lm_models += "acestep-5Hz-lm-1.7B"
    $lm_models += "acestep-5Hz-lm-4B"
}
elseif ($lm_choice -eq "" -or $lm_choice -eq "n") {
    Write-Output "跳过 LM 模型下载 / Skipping LM model download"
}

# ============= Download Models | 下载模型 =====================
# 只有两个都明确选择 n 才完全跳过，否则至少下载基础模型
$skip_dit = ($dit_choice -eq "" -or $dit_choice -eq "n")
$skip_lm = ($lm_choice -eq "" -or $lm_choice -eq "n")

if ($skip_dit -and $skip_lm) {
    Write-Output "`n跳过所有模型下载 / Skipping all model downloads"
}
else {
    Write-Output "`n=== 开始下载模型 / Starting model download ==="

    # Download additional DiT models (only if not skipped)
    if (-not $skip_dit) {
        foreach ($model in $dit_models) {
            Write-Output "`n正在下载 DiT 模型: $model / Downloading DiT model: $model"
            uv run acestep-download --model $model
        }
    }

    # Download additional LM models (only if not skipped)
    if (-not $skip_lm) {
        foreach ($model in $lm_models) {
            Write-Output "`n正在下载 LM 模型: $model / Downloading LM model: $model"
            uv run acestep-download --model $model
        }
    }
  
    Write-Output "`n模型下载完成 / Model download completed"
}

Set-Location ace-step-ui

# Run setup script (installs all dependencies)
# 运行安装脚本（安装所有依赖）
if (Test-Path "setup.bat") {
    Write-Output "Running setup.bat..."
    & .\setup.bat
}
else {
    Write-Warning "Setup script not found"
}

Write-Output "Install finished"
Read-Host | Out-Null ;
