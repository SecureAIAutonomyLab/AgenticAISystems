# Code Previews

#### 1. LLM Basics: [here](https://static.marimo.app/static/llm-basics-lxym)
#### 2. Lang Basics: [here](https://static.marimo.app/static/lang-basics-ztxh)
#### 3. Prompt Engineering: [here](https://static.marimo.app/static/prompt-engineering-cxpu)
#### 4. Tool Agent: [here](https://static.marimo.app/static/tool-agent-m32m)
#### 5. Output Parser: [here](https://static.marimo.app/static/output-parser-6ste)
#### 6. Lang Memory: [here](https://static.marimo.app/static/lang-memory-6mnq)

# Get Started

## 1. Install Miniconda (Optional)

#### Install Miniconda
To get started quickly, ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed. Use the following command for your operating system to install Miniconda in one step:

##### For Linux or macOS:
```bash
curl -o Miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda.sh -b -p $HOME/miniconda && rm Miniconda.sh && export PATH="$HOME/miniconda/bin:$PATH"
```

##### For Windows (PowerShell):
```powershell
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
Start-Process -FilePath ".\miniconda.exe" -ArgumentList "/S" -Wait
del miniconda.exe
```

##### More information [here](https://docs.anaconda.com/miniconda/install/#quick-command-line-install).

##### Verify Installation:
Once installed, verify that `conda` is installed:
```bash
conda --version
```

##### Initialize Conda:
**For Linux/macOS**:
```bash
~/miniconda/bin/conda init
```

Restart your terminal to apply changes.

**For Windows**:
Search for Anaconda Prompt in Windows search and open it.

## 2. Set Up the Python Environment
**Using Conda (Recommended)**:
```bash
conda create -n marimo_env python=3.12 -y
conda activate marimo_env
```
**Using Virtualenv**:
```bash
python -m venv marimo_env
source marimo_env/bin/activate  # For Linux/macOS
harmful_detection_env\Scripts\activate     # For Windows
```

## 3. Install Dependencies

**Install Git**: Use the following guide to install Git: [Git](https://git-scm.com/downloads)

**Install Marimo**:
```bash
pip install marimo
```

## 4. Start Marimo

**Clone repository**:
```bash
git clone https://github.com/SecureAIAutonomyLab/AgenticAISystems.git
```

**Start marimo server**:
```bash
marimo edit --headless --no-token
```

**Note**: I highly recommend reading this [guide](https://docs.marimo.io/guides/reactivity/) about how to run marimo notebooks because they work a bit differently than regular notebooks.
