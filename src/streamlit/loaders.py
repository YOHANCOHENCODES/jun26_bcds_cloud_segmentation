import pandas as pd
import segmentation_models_pytorch as smp
import torch
from config import PSPNET_MODEL_PATH, TRAIN_LABELS_PATH

import streamlit as st
from src.data.labels import load_processed_labels


@st.cache_data
def load_labels() -> pd.DataFrame:
    return load_processed_labels(TRAIN_LABELS_PATH)


@st.cache_resource
def load_model_pspnet(device: torch.device) -> smp.PSPNet:
    model = smp.PSPNet(encoder_name='resnet34', encoder_weights=None, classes=4)
    model.load_state_dict(torch.load(PSPNET_MODEL_PATH, map_location=device))
    return model.to(device).eval()
