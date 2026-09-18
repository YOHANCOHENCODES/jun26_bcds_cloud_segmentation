import streamlit as st

st.set_page_config(page_title='Cloud Segmentation', layout='wide')

pg = st.navigation(
    [
        st.Page('views/eda.py', title='EDA'),
        st.Page('views/preprocessing.py', title='Data Preprocessing'),
        st.Page('views/segmentation_masks.py', title='Segmentation masks'),
        st.Page('views/pspnet.py', title='Results PSPNet'),
        st.Page('views/unet.py', title='Results U-Net'),
    ]
)

pg.run()
