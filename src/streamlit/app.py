import streamlit as st

st.set_page_config(page_title='Cloud Segmentation', layout='wide')

pg = st.navigation(
    [
        st.Page('views/input_data_cleaned.py', title='Input data (cleaned)'),
        st.Page('views/segmentation_masks.py', title='Segmentation masks'),
        st.Page('views/augmentation.py', title='Data Augmentation'),
        st.Page('views/pspnet.py', title='Results PSPNet'),
        st.Page('views/unet.py', title='Results U-Net'),
    ]
)

pg.run()
