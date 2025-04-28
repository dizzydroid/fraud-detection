"""
Package init
"""
from pathlib import Path
from src.models import MODEL_CLASSES

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"


MODEL_NAMES = {
    'knn': 'K-Nearest Neighbors',
    'lda': 'Linear Discriminant Analysis',
    'lr': 'Linear Regression'
}
