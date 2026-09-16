from pathlib import Path

import pandas as pd
import torch
from config import (
    PSPNET_MODEL_PATH,
    TEST_IMAGES_DIR,
    TEST_IMAGES_FALLBACK_DIR,
    TRAIN_LABELS_PATH,
)

import streamlit as st
from src.data.labels import load_processed_labels
from src.models.predict_model import load_pspnet


@st.cache_data
def load_labels() -> pd.DataFrame:
    return load_processed_labels(TRAIN_LABELS_PATH)


@st.cache_data
def load_raw_labels() -> pd.DataFrame:
    """Loads the labels CSV file as it is - without any processing."""
    return pd.read_csv(TRAIN_LABELS_PATH)


@st.cache_resource
def load_model_pspnet(device: torch.device) -> torch.nn.Module:
    """Rebuilds the PSPNet and loads the trained weights into it."""
    return load_pspnet(PSPNET_MODEL_PATH, device)


@st.cache_data
def load_test_images_dir() -> Path | None:
    """The directory the test images are taken from: the full test set when it
    is there, otherwise the few images that are committed for the presentation
    - `data/` is not part of the repository, so after a deployment only those
    are left. Returns `None` when neither directory holds an image.
    """
    dir_paths = [TEST_IMAGES_DIR, TEST_IMAGES_FALLBACK_DIR]
    for dir_path in dir_paths:
        # `glob()` on a directory that isn't there yields nothing, so this also
        # covers the missing one - and unlike `is_dir()` an existing but empty
        # directory doesn't shadow the fallback either
        if any(dir_path.glob('*.jpg')):
            return dir_path
    return None


@st.cache_data
def load_test_image_names(dir_path: Path) -> list[str]:
    """Names of the test images in `dir_path`."""
    return sorted(path.name for path in dir_path.glob('*.jpg'))
