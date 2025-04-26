"""
Package init
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
