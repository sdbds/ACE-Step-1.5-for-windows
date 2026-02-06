# Navigate to UI directory and run setup
# 进入 UI 目录并运行安装脚本
Set-Location ace-step-ui

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

# Run setup script (installs all dependencies)
# 运行安装脚本（安装所有依赖）
if (Test-Path "start.bat") {
    Write-Output "Running start.bat..."
    & .\start.bat
}
else {
    Write-Warning "Setup script not found"
}

Write-Output "Start finished"
Read-Host | Out-Null ;
