from config import FIGURES_DIR

import streamlit as st

# This page only presents results - it does not run the U-Net. The trained
# Keras model stays out of the app on purpose: TensorFlow next to torch does
# not fit into the 1 GB the deployment has, so the plots are the ones that
# were exported to `reports/figures/` during training.

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
