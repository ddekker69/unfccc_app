# pipeline_bootstrap.py

import os
import sys
import importlib

# Ensure stdout can display emoji on Windows CMD
if sys.stdout.encoding is None or "UTF-8" not in sys.stdout.encoding.upper():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

REQUIRED_FOLDERS = ["data", "indexes", "embeddings"]
REQUIRED_MODULES = ["pandas", "fitz", "faiss", "pickle", "sentence_transformers"]
REQUIRED_COLUMNS = ['document_id', 'title', 'text', 'country', 'status']

def check_folders():
    print("🔍 Checking required folders...")
    for folder in REQUIRED_FOLDERS:
        if not os.path.exists(folder):
            print(f"⚠ Folder '{folder}' is missing. Creating it...")
            os.makedirs(folder)
        else:
            print(f"✅ Folder '{folder}' exists.")

def check_dependencies():
    print("\n🔍 Checking required modules...")
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
            print(f"✅ Module '{module}' is available.")
        except ImportError:
            print(f"❌ Module '{module}' is missing! Please install it with: pip install {module}")
            sys.exit(1)

def check_dataframe(df):
    print("\n🔍 Checking dataframe columns...")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        print(f"❌ Missing columns in dataframe: {missing}")
        sys.exit(1)
    print(f"✅ All required columns found: {REQUIRED_COLUMNS}")

if __name__ == "__main__":
    check_folders()
    check_dependencies()
    print("\n✅ Bootstrap completed successfully.")
