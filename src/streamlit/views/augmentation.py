import albumentations as A
from config import TRAIN_IMAGES_DIR
from loaders import load_labels

import streamlit as st
from src.data.cloud_segmentation_image import CloudSegmentationImage
from src.visualization.image_display import draw_segmap_on_image

st.write('### The cleaned-up labels dataframe')


df = load_labels()
csi = CloudSegmentationImage(df, 4, TRAIN_IMAGES_DIR)
loaded_image = csi.get_image_rgb()
loaded_image_masks = csi.get_mask_stack()
if loaded_image is None:
    st.error(f'The image {csi.get_image_path()} could not be read')
    st.stop()

st.write(f'### Image {csi.get_name()}')
st.caption(
    f'{csi.get_image_path()} - {loaded_image.shape}, {loaded_image.dtype}'
)

loaded_image_with_segmap = draw_segmap_on_image(
    csi, alpha=0.4, draw_label_names=True
)

container_loaded, container_augmented = st.columns(2)
if loaded_image_with_segmap is not None:
    container_loaded.image(
        loaded_image_with_segmap,
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

    # NOTE: The `pipeline()` result, here stored into `transformed_data`, is the
    # albumentations result dictionary. Here, it contains the keys
    # 'image' and 'masks' (because these are the ones we gave in)
    transformed_data = pipeline(image=loaded_image, masks=loaded_image_masks)

    transformed_image = transformed_data['image']
    transformed_masks = transformed_data['masks']

    csi_augmented = CloudSegmentationImage.from_image_and_masks(
        image_rgb=transformed_image,
        mask_stack=transformed_masks,
        image_name=csi.get_name(),
        image_path=csi.get_image_path().parent,
    )

    augmented_image_with_segmap = draw_segmap_on_image(
        csi_augmented, alpha=0.4, draw_label_names=True
    )

    if augmented_image_with_segmap is not None:
        container_augmented.image(
            augmented_image_with_segmap,
            caption='Augmented image with Segmentation masks',
        )
    # container_loaded.image(loaded_image, caption='')
