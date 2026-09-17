from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Everything the app reads lives in `data/presentation/` - also when the full
# dataset is sitting right next to it in `data/raw/`. `data/raw/` is ~1.3 GB
# and not part of the repository, so it is simply not there after a
# deployment; reading it locally would mean developing against data the
# deployed app never sees. `src.data.build_presentation_data` fills this
# directory from `data/raw/`.
PRESENTATION_DIR = PROJECT_ROOT / 'data' / 'presentation'

# All 22184 label rows, minus the RLE strings. Carries every number the app
# states about the dataset as a whole.
LABELS_SLIM_PATH = PRESENTATION_DIR / 'labels_slim.csv.gz'

# The first rows of `train.csv` exactly as they are, RLE strings included
LABELS_RAW_HEAD_PATH = PRESENTATION_DIR / 'labels_raw_head.csv'

# The example training images and their full rows, from which the masks are
# decoded. Examples - no number on any page is counted from these.
TRAIN_IMAGES_DIR = PRESENTATION_DIR / 'train_images'
TRAIN_LABELS_PATH = PRESENTATION_DIR / 'train_labels.csv'

# Unlabeled images the PSPNet has never seen, for the prediction page
TEST_IMAGES_DIR = PRESENTATION_DIR / 'test_images'

PSPNET_MODEL_PATH = PROJECT_ROOT / 'models' / 'pspnet_resnet34_binary.pth'

# Plots and diagrams that were exported for the report. Pages that only show
# results rather than computing them take their images from here.
FIGURES_DIR = PROJECT_ROOT / 'reports' / 'figures'
