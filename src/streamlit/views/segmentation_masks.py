from config import TRAIN_IMAGES_DIR
from loaders import (
    first_row_id,
    load_train_image_names,
    load_train_labels,
)

import streamlit as st
from src.data.cloud_segmentation_image import CloudSegmentationImage
from src.visualization.image_display import draw_segmap_on_image

st.write('### What does RLE encoded mean?')
st.markdown("""
> In order to reduce the submission file size, our metric uses *run-length
> encoding* on the pixel values. Instead of submitting an exhaustive list of
> indices for your segmentation, you will submit pairs of values that contain a
> start position and a run length. E.g. `1 3` implies starting at pixel 1 and
> running a total of 3 pixels (1,2,3). The pixels are numbered from top to
> bottom, then left to right.

*Source:*
[Kaggle - Understanding Clouds from Satellite Images](https://www.kaggle.com/competitions/understanding_cloud_organization/overview/evaluation)
""")

st.write('### Displaying the segmentation masks')

train_labels_df = load_train_labels()
image_name = st.selectbox('Training image', load_train_image_names())

csi = CloudSegmentationImage(
    train_labels_df, first_row_id(train_labels_df, image_name), TRAIN_IMAGES_DIR
)
loaded_image_rgb = csi.get_image_rgb()
if loaded_image_rgb is None:
    st.error(f'The image {csi.get_image_path()} could not be read')
    st.stop()

# Create an rgb variant of the image with the masks drawn on it
segmap_image_rgb = draw_segmap_on_image(csi)
if segmap_image_rgb is None:
    st.error(f'The segmentation map of {csi.get_name()} could not be drawn')
    st.stop()

container_left, container_right = st.columns(2)
container_left.image(loaded_image_rgb, caption='Original')
container_right.image(segmap_image_rgb, caption='With Masks Overlay')
