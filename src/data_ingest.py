"""
Download the PaySim transaction log via the Kaggle API.

Run:
    python -m src.data_ingest

Produces:
    data/raw/ps_raw.csv         (≈500 MB)

Prereqs:
    pip install kaggle
    Kaggle creds in ~/.kaggle/kaggle.json  OR  env vars (KAGGLE_USERNAME / KAGGLE_KEY)
"""
from pathlib import Path
import subprocess
import sys
import zipfile
import shutil
import tempfile

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASET          = "ealaxi/paysim1"
ORIG_CSV_NAME    = "PS_20174392719_1491204439457_log.csv"
TARGET_CSV_NAME  = "ps_raw.csv"


def _clean_leftovers():
    """Remove any directory/file left behind by a previous failed extract."""
    bad_path = RAW_DIR / ORIG_CSV_NAME
    if bad_path.is_dir():
        shutil.rmtree(bad_path, ignore_errors=True)
    elif bad_path.exists():
        bad_path.unlink(missing_ok=True)


def download_zip() -> Path:
    """Download the dataset ZIP and return its path."""
    zip_path = RAW_DIR / f"{DATASET.split('/')[-1]}.zip"
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(RAW_DIR), "--force"],
        check=True,
    )
    return zip_path


def extract_and_rename(zip_path: Path):
    """Unpack in a temp dir, rename CSV, then delete the ZIP."""
    with tempfile.TemporaryDirectory(dir=RAW_DIR) as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)

        src = Path(tmp) / ORIG_CSV_NAME
        dst = RAW_DIR / TARGET_CSV_NAME
        if dst.exists():
            dst.unlink()     # overwrite any stale copy
        shutil.move(src, dst)

    zip_path.unlink(missing_ok=True)


def main():
    target = RAW_DIR / TARGET_CSV_NAME
    if target.exists():
        print("✓ dataset already present")
        return

    _clean_leftovers()

    try:
        zip_path = download_zip()
    except FileNotFoundError:
        sys.exit(
            "kaggle CLI not found.  Install it with `pip install kaggle`, "
            "or manually place ps_raw.csv in data/raw/."
        )
    except subprocess.CalledProcessError as err:
        sys.exit(f"Error downloading dataset: {err}")

    extract_and_rename(zip_path)
    print(f"✓ Downloaded and unpacked ➜ {target}")


if __name__ == "__main__":
    main()
