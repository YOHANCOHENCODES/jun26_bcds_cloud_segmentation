import albumentations as A
from config import TRAIN_IMAGES_DIR
from loaders import load_labels

import streamlit as st
from src.data.cloud_segmentation_image import CloudSegmentationImage
from src.visualization.image_display import draw_segmap_on_image

# Create the augmentation pipeline
pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.8),
        A.Rotate(p=0.6),
    ]
)


def augment(image_rgb, image_masks):
    # Call ('apply') the Augmentation pipeline
    transformed_data = pipeline(image=image_rgb, masks=image_masks)

    # Extract the augmented rgb and masks from the pipeline result
    augmented_image_rgb = transformed_data['image']
    augmented_image_masks = transformed_data['masks']

    # Create a cloud segmentation image from the augmented rgb + masks that is
    # 'virtually linked' to the original image by the name and the file path.
    augmented_csi = CloudSegmentationImage.from_image_and_masks(
        image_rgb=augmented_image_rgb,
        mask_stack=augmented_image_masks,
        image_name=csi.get_name(),
        image_path=csi.get_image_path().parent,
    )

    # Create an rgb variant of the augmented image with the masks drawn on it
    augmented_image_with_segmap = draw_segmap_on_image(augmented_csi)
    if augmented_image_with_segmap is None:
        st.error(f'The segmentation map of {csi.get_name()} could not be drawn')
        st.stop()
    return augmented_image_with_segmap


# Load the labels
df = load_labels()

# Create a CloudSegmentationImage from of the picture that is 'linked' to
# rowId=4 in the labels DataFrame
csi = CloudSegmentationImage(df, 4, TRAIN_IMAGES_DIR)

# Get the rgb of the image - and also its masks
loaded_image_rgb = csi.get_image_rgb()
loaded_image_masks = csi.get_mask_stack()
if loaded_image_rgb is None:
    st.error(f'The image {csi.get_image_path()} could not be read')
    st.stop()

# Create an rgb variant of the image with the masks drawn on it
loaded_image_rgb_with_segmap = draw_segmap_on_image(csi)
if loaded_image_rgb_with_segmap is None:
    st.error(f'The segmentation map of {csi.get_name()} could not be drawn')
    st.stop()

# Create the Streamlit page title
st.write('## Augmentation')

st.write(f'### Image {csi.get_name()}')
st.caption(
    f'{csi.get_image_path()} - {loaded_image_rgb.shape}, {loaded_image_rgb.dtype}'
)

# Create a Streamlit 'container' to have the original images on the left and the
# augmented  ones at the right
container_left, container_right = st.columns(2)

# Run the augmentation once on first load, then again on every button click -
# the pipeline uses random probabilities, so each run can produce a different
# result
if 'augmented_image_with_segmap' not in st.session_state:
    st.session_state.augmented_image_with_segmap = augment(
        loaded_image_rgb, loaded_image_masks
    )

if st.button('Regenerate augmentation'):
    st.session_state.augmented_image_with_segmap = augment(
        loaded_image_rgb, loaded_image_masks
    )

augmented_image_with_segmap = st.session_state.augmented_image_with_segmap

# Display the images
container_left.image(loaded_image_rgb_with_segmap, caption='Original')
container_right.image(augmented_image_with_segmap, caption='Augmented')
