# Ace-Step-1.5-for-windows

original codebase from ACE-Step-1.5

https://github.com/ace-step/ACE-Step-1.5

## 🔧 Setting up the Environment for Windows

  Give unrestricted script access to powershell so venv can work:

- Open an administrator powershell window
- Type `Set-ExecutionPolicy Unrestricted` and answer A
- Close admin powershell window

## Installation

Clone the repo with `--recurse-submodules`:

```
git clone --recurse-submodules https://github.com/ace-step/ACE-Step-1.5.git
```

# MUST USE --recurse-submodules

### Windows
Run the following PowerShell script:
```powershell
./1、install-uv-qinglong.ps1
```

#### VS Studio 2022 for torch compile
Download from Microsoft offical link:
https://aka.ms/vs/17/release/vs_community.exe

Install C++ desktop and language package with English(especially for asian computer)

### Linux
1. First install PowerShell:
```bash
./0、install pwsh.sh
```
2. Then run the installation script using PowerShell:
```powershell
sudo pwsh ./1、install-uv-qinglong.ps1
```
use sudo pwsh if you in Linux without root user.

## Usage

Run

```powershell
3、run_server.ps1
```

for API_backend

```powershell
4、run_npmgui.ps1
```

for npm_frontend
