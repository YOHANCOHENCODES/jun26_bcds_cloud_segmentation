import albumentations as A
from config import TRAIN_IMAGES_DIR
from loaders import load_labels

import streamlit as st
from src.data.cloud_segmentation_image import CloudSegmentationImage
from src.visualization.image_display import draw_segmap_on_image

df = load_labels()
csi = CloudSegmentationImage(df, 4, TRAIN_IMAGES_DIR)
loaded_image_rgb = csi.get_image_rgb()
loaded_image_masks = csi.get_mask_stack()
if loaded_image_rgb is None:
    st.error(f'The image {csi.get_image_path()} could not be read')
    st.stop()

loaded_image_rgb_with_segmap = draw_segmap_on_image(csi)
if loaded_image_rgb_with_segmap is None:
    st.error(f'The segmentation map of {csi.get_name()} could not be drawn')
    st.stop()

container_loaded, container_augmented = st.columns(2)
container_loaded.image(
    loaded_image_rgb_with_segmap,
    caption='Original image with Segmentation masks',
)

pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.8),
        A.Rotate(p=0.6),
    ]
)


transformed_data = pipeline(image=loaded_image_rgb, masks=loaded_image_masks)

augmented_image_rgb = transformed_data['image']
augmented_image_masks = transformed_data['masks']

augmented_csi = CloudSegmentationImage.from_image_and_masks(
    image_rgb=augmented_image_rgb,
    mask_stack=augmented_image_masks,
    image_name=csi.get_name(),
    image_path=csi.get_image_path().parent,
)


augmented_image_with_segmap = draw_segmap_on_image(augmented_csi)
if augmented_image_with_segmap is None:
    st.error(f'The segmentation map of {csi.get_name()} could not be drawn')
    st.stop()

container_augmented.image(
    augmented_image_with_segmap,
    caption='Augmented image with Segmentation masks',
)

st.write('### The cleaned-up labels dataframe')

st.write(f'### Image {csi.get_name()}')
st.caption(
    f'{csi.get_image_path()} - {loaded_image_rgb.shape}, {loaded_image_rgb.dtype}'
)
