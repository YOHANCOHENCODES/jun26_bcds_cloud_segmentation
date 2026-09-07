import torch
from config import MODEL_DIR
from loaders import load_model_pspnet

import streamlit as st

st.write('### PSPNet model results')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model_pspnet(MODEL_DIR, device)
