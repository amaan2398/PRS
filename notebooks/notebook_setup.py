import sys
import os
from pathlib import Path

# Get the absolute path of the project's root directory (one level up from 'notebooks')
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / 'data'
ROW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Change the current working directory to the project root
os.chdir(ROOT_DIR)

# Verify the change (optional)
print(f"Current working directory changed to the project root directory to: {os.getcwd()}")

print("Directory Structure:")
print("Root :", ROOT_DIR)
print("Data :", DATA_DIR)
print("Row Data :", ROW_DATA_DIR)
print("Processed Data :", PROCESSED_DATA_DIR)