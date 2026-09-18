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
    FIGURES_DIR / 'mahsa_unet_architecture.png',
    caption='UNet architecture',
    width=700,
)

st.image(
    FIGURES_DIR / 'mahsa_loss_and_dice_plots.png',
    caption='Loss and Dice plots',
    width=700,
)

st.image(
    FIGURES_DIR / 'mahsa_prediction_on_test_images.png',
    caption='Predictions on test images',
    width=700,
)

st.markdown("""
*TODO: What the distribution means for the training.*
""")
