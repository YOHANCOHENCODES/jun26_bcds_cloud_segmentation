from config import TRAIN_IMAGES_DIR
from loaders import (
    first_row_id,
    load_labels_head_processed,
    load_train_image_names,
    load_train_labels,
)

import streamlit as st
from src.data.cloud_segmentation_image import CloudSegmentationImage

# The same rows the previous page shows raw, now with `Image_Label` split up
head_df = load_labels_head_processed()

# One of the example training images, to put a picture next to the table. The
# page is about the labels; the image is an illustration, so it stays the
# first example instead of asking for a choice here
train_labels_df = load_train_labels()
image_name = load_train_image_names()[0]

csi = CloudSegmentationImage(
    train_labels_df, first_row_id(train_labels_df, image_name), TRAIN_IMAGES_DIR
)
loaded_image_rgb = csi.get_image_rgb()
if loaded_image_rgb is None:
    st.error(f'The image {csi.get_image_path()} could not be read')
    st.stop()


st.write('### The cleaned-up labels dataframe')
st.markdown("""
- It's *impractical* to have the labels attached to the file name
- We *separated them* into 2 columns, `Image` and `Label`
""")

st.dataframe(head_df)

st.write('### An image of our training data')
st.image(loaded_image_rgb, csi.get_name())
