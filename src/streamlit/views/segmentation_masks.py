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

# Create an rgb variant of the image with the masks drawn on it
segmap_image_rgb = draw_segmap_on_image(csi)
if segmap_image_rgb is None:
    st.error(f'The segmentation map of {csi.get_name()} could not be drawn')
    st.stop()

container_left, container_right = st.columns(2)
container_left.image(loaded_image_rgb, caption='Original')
container_right.image(segmap_image_rgb, caption='With Masks Overlay')

st.write('### The cleaned-up labels dataframe')
st.markdown("""
- It's *impractical* to have the labels attached to the file name
- We *separated them* into 2 columns, `Image` and `Label`
""")

st.dataframe(df.head(8))

st.write(f'### Image {csi.get_name()}')
st.caption(
    f'{csi.get_image_path()} - {loaded_image_rgb.shape}, {loaded_image_rgb.dtype}'
)
