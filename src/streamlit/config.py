from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PSPNET_MODEL_PATH = PROJECT_ROOT / 'models' / 'pspnet_resnet34_binary.pth'
TRAIN_LABELS_PATH = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
