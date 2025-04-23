"""Download PaySim CSV via Kaggle API (or local copy).
Run: python -m src.data_ingest
Outputs: data/raw/PS_20174392719_1491204439457_log.csv
NOTE: Kaggle creds must be set via environment (KAGGLE_USERNAME / KAGGLE_KEY).
"""
from pathlib import Path
import subprocess
import sys

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
DATASET = "ealaxi/paysim1"
FILENAME = "ps_raw.csv"

def main():
    target = RAW_DIR / FILENAME
    if target.exists():
        print("✓ dataset already present")
        return

    cmd = [
        "kaggle", "datasets", "download", "-d", DATASET, "-f", FILENAME, 
        "-p", str(RAW_DIR), "--force"
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit(
            "Error: kaggle CLI not found. Install with pip install kaggle.\n"
            "Or drop the CSV manually into data/raw/."
        )

if __name__ == "__main__":
    main()