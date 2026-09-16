from config import TRAIN_IMAGES_DIR
from loaders import load_labels

import streamlit as st
from src.data.cloud_segmentation_image import CloudSegmentationImage
from src.visualization.image_display import draw_segmap_on_image

# Load the labels
df = load_labels()

# Create a CloudSegmentationImage from of the picture that is 'linked' to
# rowId=4 in the labels DataFrame
csi = CloudSegmentationImage(df, 4, TRAIN_IMAGES_DIR)
loaded_image_rgb = csi.get_image_rgb()
if loaded_image_rgb is None:
    st.error(f'The image {csi.get_image_path()} could not be read')
    st.stop()


st.write('### The cleaned-up labels dataframe')
st.markdown("""
- It's *impractical* to have the labels attached to the file name
- We *separated them* into 2 columns, `Image` and `Label`
""")

st.dataframe(df.head(8))

st.write('### An image of our training data')
st.image(loaded_image_rgb, csi.get_name())
# st.caption(f'{csi.get_image_path()} - {loaded_image_rgb.shape}')
