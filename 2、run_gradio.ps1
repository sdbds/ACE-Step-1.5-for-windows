
# ============= Command Line Parameters | 命令行参数 =====================
param(
  [int]$Port = 7860,                           # Server port
  [string]$ServerName = "127.0.0.1",          # Server address (use 0.0.0.0 for network access)
  [switch]$Share,                              # Create public Gradio link
  [ValidateSet("en", "zh", "ja")]
  [string]$Language = "zh",                    # UI language: en, zh, ja
  [switch]$InitService,                        # Auto-initialize models on startup
  [string]$ConfigPath = "auto",                # DiT model (e.g., acestep-v15-turbo, acestep-v15-turbo-shift3)
  [string]$LmModelPath = "auto",               # LM model (e.g., acestep-5Hz-lm-0.6B, acestep-5Hz-lm-1.7B)
  [string]$OffloadToCpu = "auto",              # CPU offload (auto-enabled if VRAM < 16GB)
  [switch]$EnableApi,                          # Enable REST API endpoints alongside Gradio UI
  [string]$ApiKey = "none",                    # API key for API endpoints authentication
  [string]$AuthUsername = "none",              # Username for Gradio authentication
  [string]$AuthPassword = "none"               # Password for Gradio authentication
)

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
# Activate python venv
Set-Location $PSScriptRoot
$env:PYTHONPATH = "$PSScriptRoot$([System.IO.Path]::PathSeparator)$($env:PYTHONPATH)"
$VenvPaths = @(
  "./venv/Scripts/activate",
  "./.venv/Scripts/activate",
  "./venv/bin/Activate.ps1",
  "./.venv/bin/activate.ps1"
)

foreach ($Path in $VenvPaths) {
  if (Test-Path $Path) {
    Write-Output "Activating venv: $Path"
    & $Path
    break
  }
}

$Env:HF_HOME = "huggingface"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
#$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:UV_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu130"
$Env:UV_CACHE_DIR = "${env:LOCALAPPDATA}/uv/cache"
$Env:UV_NO_BUILD_ISOLATION = "1"
$Env:UV_NO_CACHE = "0"
$Env:UV_LINK_MODE = "symlink"
$Env:UV_INDEX_STRATEGY = "unsafe-best-match"
#$Env:CUDA_VISIBLE_DEVICES = "1"  # 设置GPU id，0表示使用第一个GPU，-1表示不使用GPU

#$Env:HTTP_PROXY = "http://127.0.0.1:7890"
#$Env:HTTPS_PROXY = "http://127.0.0.1:7890"

$ext_args = [System.Collections.ArrayList]::new()
$uv_args = [System.Collections.ArrayList]::new()

# ============= Build Arguments | 构建参数 =====================
# Server configuration
[void]$ext_args.Add("--port")
[void]$ext_args.Add($Port)
[void]$ext_args.Add("--server-name")
[void]$ext_args.Add($ServerName)

# Share link
if ($Share) {
  [void]$ext_args.Add("--share")
}

# Language
[void]$ext_args.Add("--language")
[void]$ext_args.Add($Language)

# Initialize service
if ($InitService) {
  [void]$ext_args.Add("--init_service")
  [void]$ext_args.Add("false")
}

# Model paths
if ($ConfigPath -ne "auto") {
  [void]$ext_args.Add("--config_path")
  [void]$ext_args.Add($ConfigPath)
}

if ($LmModelPath -ne "auto") {
  [void]$ext_args.Add("--lm_model_path")
  [void]$ext_args.Add($LmModelPath)
}

# CPU offload
if ($OffloadToCpu -ne "auto") {
  [void]$ext_args.Add("--offload_to_cpu")
  [void]$ext_args.Add($OffloadToCpu)
}

# API configuration
if ($EnableApi) {
  [void]$ext_args.Add("--enable-api")
}

if ($ApiKey -ne "none") {
  [void]$ext_args.Add("--api-key")
  [void]$ext_args.Add($ApiKey)
}

# Authentication
if ($AuthUsername -ne "none") {
  [void]$ext_args.Add("--auth-username")
  [void]$ext_args.Add($AuthUsername)
}

if ($AuthPassword -ne "none") {
  [void]$ext_args.Add("--auth-password")
  [void]$ext_args.Add($AuthPassword)
}

# run train
uv run $uv_args acestep $ext_args

Write-Output "Start finished"
Read-Host | Out-Null ;
