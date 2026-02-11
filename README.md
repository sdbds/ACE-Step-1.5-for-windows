# Ace-Step-1.5-for-windows

original backend codebase from ACE-Step-1.5

https://github.com/ace-step/ACE-Step-1.5

original frontend codebase from ace-step-ui

https://github.com/fspecii/ace-step-ui

<img width="2062" height="952" alt="image" src="https://github.com/user-attachments/assets/6e682194-99f2-4267-b412-1b5198720b87" />

## Feather

### 1. Complete style search and random selection, with a total of 936 styles, synchronized from Suno's explorer.

<img width="412" height="752" alt="image" src="https://github.com/user-attachments/assets/a17ad92c-9cdd-4927-9e0e-5e9848747438" />

### 2.Generate a complete record of song parameters, which can be reused at any time.

<img width="399" height="517" alt="image" src="https://github.com/user-attachments/assets/42804479-a252-46cd-9032-963f71123536" />

### 3.All pages are localized into four languages ​​(English, Chinese, Japanese, and Korean).

<img width="824" height="718" alt="image" src="https://github.com/user-attachments/assets/2b3a81fb-88fe-42a6-81d0-998d52da50c3" />

### 4.LoRA training support and memory offloading optimization.
There were still some issues with the original backend repository's GPU memory unloading, so we made significant reconstructions and optimizations. Now we train with the lowest GPU memory and the fastest speed.

<img width="1934" height="832" alt="image" src="https://github.com/user-attachments/assets/88b61873-a6dd-454c-ad97-4ceee5f9a22f" />

### 5.LoKR training support and reduce training time and improve sound quality

<img width="1913" height="813" alt="image" src="https://github.com/user-attachments/assets/68fe0074-c0cb-427a-92e3-ee7b7fe7acd5" />

### 6.We've also added the ability to load Lora/LoKR. Note that Lora reads folders, while LoKR reads safetensors files.

<img width="410" height="408" alt="image" src="https://github.com/user-attachments/assets/9c2bca58-4133-4884-aecf-fabe42bcf5c9" />


## 🔧 Setting up the Environment for Windows

  Give unrestricted script access to powershell so venv can work:

- Open an administrator powershell window
- Type `Set-ExecutionPolicy Unrestricted` and answer A
- Close admin powershell window

## Installation

Clone the repo with `--recurse-submodules`:

```
git clone --recurse-submodules https://github.com/sdbds/ACE-Step-1.5-for-windows.git -b qinglong
```

# MUST USE --recurse-submodules

### Windows
Run the following PowerShell script:
```powershell
./1、install-uv-qinglong.ps1
```
### (Optional)
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
