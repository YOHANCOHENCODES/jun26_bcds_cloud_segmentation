from pathlib import Path

import torch
from loaders import load_model_pspnet

import streamlit as st

st.set_page_config(page_title='Cloud Segmentation', layout='wide')

# # PROJECT_ROOT is the 2nd parent of current file
# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# MODEL_PATH = 'models/pspnet_resnet34_binary.pth'
# MODEL_PATH = PROJECT_ROOT / MODEL_PATH

if not MODEL_PATH.exists():
    st.error(f'Model not found: {MODEL_PATH}')
    st.caption(f'Working directory: {Path.cwd()}')
    st.stop()


pg = st.navigation(
    [
        st.Page('views/input_data.py', title='Input data'),
        st.Page('views/augmentation.py', title='Data Augmentation'),
        st.Page('views/pspnet.py', title='Results PSPNet'),
        st.Page('views/unet.py', title='Results U-Net'),
    ]
)

pg.run()
# st.title('Cloud Segmentation')
# st.sidebar.title('Contents')
# pages = ['Input data', 'Data Augmentation', 'Results PSPNet', 'Results U-Net']
# page = st.sidebar.radio('Go to', pages)


# if page == pages[0]:
#     st.write('### The cleaned-up labels dataframe')
#     st.write('Every picture has 4 rows with the an RLE encoded mask on it')
#     df = pd.read_csv('../data/raw/train.csv')

# if page == pages[1]:
#     st.write('### Augmentation')

# if page == pages[2]:
#     st.write('### PSPNet results')

# if page == pages[3]:
#     st.write('### UNet results')
