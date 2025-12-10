### 1) Install Python dependencies

Open **PowerShell** in the project folder:

    # 1) Create & activate a virtual environment
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

    # 2) Upgrade pip (recommended)
    python -m pip install --upgrade pip

    # 3) Install packages
    pip install -r requirements.txt

    If PowerShell blocks script execution, run:
    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned (then retry activation).

### 2) Please makes sure you have 
      llama3
      llama3.1:8b
      gpt-oss:20b
      zephyr

      Or Either change the models you in python so that you can change to the one's you have.

### 3) Install pytorch that runs on GPU

       pip install "torch==2.3.1+cu121" --index-url https://download.pytorch.org/whl/cu121```

### 4) Run the app

With your virtual environment activated:

    python -m  run app.py
