## 1) Install Python dependencies

    Open **PowerShell** in the project folder:

    ```powershell
    # 1) Create & activate a virtual environment
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

    # 2) Upgrade pip (recommended)
    python -m pip install --upgrade pip

    # 3) Install packages
    pip install -r requirements.txt

    If PowerShell blocks script execution, run:
    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned (then retry activation).

## 2) Please makes sure you have 
      llama3
      llama3.1:8b
      gpt-oss:20b
      zephyr

## 3) Install pytorch that runs on GPU
```powershell
pip install "torch==2.3.1+cu121" --index-url https://download.pytorch.org/whl/cu121

## 6) Run the app

With your virtual environment activated:

    python -m  streamlit run app.py
