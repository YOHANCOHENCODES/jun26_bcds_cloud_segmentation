import streamlit as st

st.set_page_config(page_title='Cloud Segmentation', layout='wide')

pg = st.navigation(
    [
        st.Page('views/input_data.py', title='Input data'),
        st.Page('views/augmentation.py', title='Data Augmentation'),
        st.Page('views/pspnet.py', title='Results PSPNet'),
        st.Page('views/unet.py', title='Results U-Net'),
    ]
)

pg.run()
