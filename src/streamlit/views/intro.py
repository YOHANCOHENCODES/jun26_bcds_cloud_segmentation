from config import FIGURES_DIR

import streamlit as st

# Title page. Just the cover image, horizontally centered.
with st.container(horizontal=True, horizontal_alignment='center'):
    st.image(FIGURES_DIR / 'streamlit_intro.png', width=900)
