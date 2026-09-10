from config import TRAIN_IMAGES_DIR
from loaders import load_labels

import streamlit as st
from src.data.cloud_segmentation_image import CloudSegmentationImage
from src.visualization.image_display import draw_segmap_on_image

st.write('### The cleaned-up labels dataframe')

df = load_labels()
st.dataframe(df.head(8))

# Display the image with ID 4 (in the Labels data frame)
csi = CloudSegmentationImage(df, 4, TRAIN_IMAGES_DIR)
image = csi.get_image_rgb()
if image is None:
    st.error(f'The image {csi.get_image_path()} could not be read')
    st.stop()

st.write(f'### Image {csi.get_name()}')
st.caption(f'{csi.get_image_path()} - {image.shape}, {image.dtype}')

original, overlay = st.columns(2)
original.image(image, caption='Original')

# The notebooks render this overlay with `display_masks_overlay()`, which only
# wraps `draw_segmap_on_image()` in a matplotlib figure. Here the plain RGB
# array it returns goes straight to `st.image()`, which gives us zoom and
# fullscreen for free.
segmap_image = draw_segmap_on_image(csi, alpha=0.4, draw_label_names=True)
if segmap_image is not None:
    overlay.image(segmap_image, caption='Segmentation map')
