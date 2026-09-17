import pandas as pd
import torch
from config import (
    LABELS_RAW_HEAD_PATH,
    LABELS_SLIM_PATH,
    PSPNET_MODEL_PATH,
    TEST_IMAGES_DIR,
    TRAIN_LABELS_PATH,
)

import streamlit as st
from src.data.labels import load_processed_labels
from src.models.predict_model import load_pspnet


@st.cache_data
def load_labels_slim() -> pd.DataFrame:
    """The complete labels table: all 22184 rows, with `Image`, `Label`,
    `HasMask` and `MaskArea`, but without the RLE strings.

    This is what every statement the app makes *about the dataset* is counted
    from - never the files that happen to lie in `data/presentation/`. See
    `src.data.build_presentation_data`.
    """
    return pd.read_csv(LABELS_SLIM_PATH)


@st.cache_data
def load_labels_raw_head() -> pd.DataFrame:
    """The first rows of `train.csv` untouched, to show what the raw labels
    look like before they are cleaned up."""
    return pd.read_csv(LABELS_RAW_HEAD_PATH)


@st.cache_data
def load_labels_head_processed() -> pd.DataFrame:
    """The same rows as `load_labels_raw_head()`, but with `Image_Label` split
    into `Image` and `Label` - so both pages show the cleanup on identical
    rows."""
    return load_processed_labels(LABELS_RAW_HEAD_PATH)


@st.cache_data
def load_train_labels() -> pd.DataFrame:
    """Labels of the example training images, RLE strings included, so their
    masks can be decoded.

    Already carries the separate `Image`/`Label` columns - the build script
    writes it in the cleaned-up shape - so unlike the raw head it does not go
    through `load_processed_labels()`.

    These are a handful of examples. Nothing is counted from them.
    """
    return pd.read_csv(TRAIN_LABELS_PATH)


@st.cache_data
def load_train_image_names() -> list[str]:
    """Names of the example training images, in the order the build script
    selected them."""
    return load_train_labels()['Image'].drop_duplicates().tolist()


def first_row_id(labels_df: pd.DataFrame, image_name: str) -> int:
    """Index of the first row belonging to `image_name`.

    `CloudSegmentationImage` addresses an image by a row of the labels table,
    so a chosen name has to be translated back into one.
    """
    return int(labels_df.index[labels_df['Image'] == image_name][0])


@st.cache_resource
def load_model_pspnet(device: torch.device) -> torch.nn.Module:
    """Rebuilds the PSPNet and loads the trained weights into it."""
    return load_pspnet(PSPNET_MODEL_PATH, device)


@st.cache_data
def load_test_image_names() -> list[str]:
    """Names of the unlabeled test images shipped for the prediction page."""
    return sorted(path.name for path in TEST_IMAGES_DIR.glob('*.jpg'))
