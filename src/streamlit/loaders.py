from pathlib import Path

import pandas as pd
import segmentation_models_pytorch as smp
import torch

import streamlit as st

# from data.labels import load_processed_labels
from src.data.labels import load_processed_labels


@st.cache_data
def load_labels(data_dir: Path) -> pd.DataFrame:
    data_dir = data_dir / 'train.csv'
    return load_processed_labels(data_dir)


@st.cache_resource
def load_model_pspnet(models_dir: Path, device: torch.device) -> smp.PSPNet:
    models_dir = models_dir / 'pspnet_resnet34_binary.pth'
    model = smp.PSPNet(encoder_name='resnet34', encoder_weights=None, classes=4)
    model.load_state_dict(torch.load(models_dir, map_location=device))
    return model.to(device).eval()
