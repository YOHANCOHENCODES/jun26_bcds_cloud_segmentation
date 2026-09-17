from config import FIGURES_DIR

import streamlit as st

# This page only presents results - it does not run the U-Net. The trained
# Keras model stays out of the app on purpose: it would pull TensorFlow in
# next to torch, for plots that were already made during training and are
# sitting in `reports/figures/`. So they are shown from there.

st.write('## U-Net model results')

st.markdown("""
*TODO: Text about the architecture and the training setup.*
""")

st.write('### Distribution of the cloud labels')

st.image(
    FIGURES_DIR / 'distribution-of-cloud-labels.png',
    caption='How often each cloud type appears in the training labels',
    width=700,
)

st.markdown("""
*TODO: What the distribution means for the training.*
""")
