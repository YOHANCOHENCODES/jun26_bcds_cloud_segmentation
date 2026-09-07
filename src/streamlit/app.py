from pathlib import Path

import segmentation_models_pytorch as smp
import torch

import streamlit as st


# NOTE: st.cache_resource prevents the model from being reloaded from disk every
# time it is rerun
@st.cache_resource
def load_model(path: Path, device: torch.device) -> smp.PSPNet:
    model = smp.PSPNet(encoder_name='resnet34', encoder_weights=None, classes=4)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


# PROJECT_ROOT is the 2nd parent of current file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = 'models/pspnet_resnet34_binary.pth'
MODEL_PATH = PROJECT_ROOT / MODEL_PATH

if not MODEL_PATH.exists():
    st.error(f'Model not found: {MODEL_PATH}')
    st.caption(f'Working directory: {Path.cwd()}')
    st.stop()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(MODEL_PATH, device)

model = smp.PSPNet(encoder_name='resnet34', encoder_weights=None, classes=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

st.title('Cloud Segmentation')
st.sidebar.title('Contents')
pages = ['Input data', 'Data Augmentation', 'Results PSPNet', 'Results U-Net']
page = st.sidebar.radio('Go to', pages)

if page == pages[0]:
    st.write('### The cleaned-up labels dataframe')
    st.write('Every picture has 4 rows with the an RLE encoded mask on it')

if page == pages[1]:
    st.write('### Augmentation')

if page == pages[2]:
    st.write('### results of the PSPNet model')

if page == pages[3]:
    st.write('### results of the UNet model')
