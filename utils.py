import os
from pathlib import Path

def save_uploaded_files(files, save_dir="data"):
    os.makedirs(save_dir, exist_ok=True)
    for file in files:
        with open(os.path.join(save_dir, file.name), "wb") as f:
            f.write(file.getbuffer())