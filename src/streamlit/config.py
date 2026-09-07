from pathlib import Path

# PROJECT_ROOT is the 2nd parent of current file
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
