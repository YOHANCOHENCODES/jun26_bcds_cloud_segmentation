"""Data augmentation: what the training pipeline does to an image, and a live
example of it on a training image of your choice."""

import albumentations as A
import numpy as np
import pandas as pd
from config import TRAIN_IMAGES_DIR
from loaders import (
    first_row_id,
    load_train_image_names,
    load_train_labels,
)

import streamlit as st
from src.data.cloud_segmentation_image import CloudSegmentationImage
from src.visualization.image_display import draw_segmap_on_image

# --------------------------------------------------------------------------- #
# The augmentation pipeline
# --------------------------------------------------------------------------- #

pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.8),
        A.Rotate(p=0.6),
    ]
)


def augment(csi: CloudSegmentationImage) -> CloudSegmentationImage:
    """Run the pipeline once on `csi` and return the result as a new image.

    Image and masks go through the pipeline in the same call, so a geometric
    transformation moves both of them the same way. The result is linked to the
    original by name and path, so it can be displayed like any other image.
    """
    transformed_data = pipeline(
        image=csi.get_image_rgb(), masks=csi.get_mask_stack()
    )
    return CloudSegmentationImage.from_image_and_masks(
        image_rgb=transformed_data['image'],
        mask_stack=transformed_data['masks'],
        image_name=csi.get_name(),
        image_path=csi.get_image_path().parent,
    )


def with_masks_drawn(csi: CloudSegmentationImage) -> np.ndarray:
    """The image with its masks drawn on top. Stops the page if that fails."""
    image_rgb = draw_segmap_on_image(csi)
    if image_rgb is None:
        st.error(f'The segmentation map of {csi.get_name()} could not be drawn')
        st.stop()
    return image_rgb


# --------------------------------------------------------------------------- #
# What the pipeline does
# --------------------------------------------------------------------------- #

st.title('Preprocessing: Data Augmentation')

tab_why, tab_library, tab_pipeline = st.tabs(
    ['Why Augmentation?', 'The library: albumentations', 'The pipeline']
)

with tab_why:
    # The `\\` at the end of a quote line is a Markdown hard line break, so the
    # quote keeps its three lines instead of being reflowed into one block. (In
    # the Python source it has to be written as a double backslash, otherwise
    # Python reads it as a line continuation.)
    st.markdown("""
> *Data augmentation artificially increases the size of the training set*\\
> *by generating many realistic variants of each training instance.*\\
> *This reduces overfitting, making this a regularization technique.*
>
> — Aurélien Géron, **Hands-On Machine Learning with Scikit-Learn and PyTorch**,
> 2025 release, p. 438

So, basically, we use it to

- get the model to be more tolerant to variations in the data
- show the model a different variant of each image in every epoch
""")

with tab_library:
    st.markdown("""
We explored some (few) libraries and found the albumentations package quite promising and ready for
the task. Using it, gives us these benefits:

- It is well prepared for image augmentation
- augments the masks, which in our case are the labels. More precisely,
    - geometric transformations (flips, rotation) are applied identically to the image and to all of
      its masks,
    - photometric transformations (brightness, contrast) only applied to the image
- is actively maintained and runs with current Python versions
    - In contrast for example to the `imgaug` package
""")

with tab_pipeline:
    st.markdown(
        'Each transformation has its own probability `p`, so an image receives '
        'a different combination of transformations every time it is requested.'
    )
    st.code(
        """import albumentations as A

pipeline = A.Compose([
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.8),
    A.Rotate(p=0.6),
])

transformed_data = pipeline(
    image=csi.get_image_rgb(),
    masks=csi.get_mask_stack(),
)
""",
        language='python',
    )
    st.markdown("""
**Training data only.** Augmentation is applied to the training data only, to increase the variety of
the images and improve generalisation. The validation data is only resized and normalised, never
augmented, so that its score stays comparable across epochs and reflects performance on unmodified
images.
""")

st.divider()

# --------------------------------------------------------------------------- #
# The same pipeline, live
# --------------------------------------------------------------------------- #

st.header('Live example')

train_labels_df = load_train_labels()
image_name = st.selectbox('Training image', load_train_image_names())

csi = CloudSegmentationImage(
    train_labels_df, first_row_id(train_labels_df, image_name), TRAIN_IMAGES_DIR
)
if csi.get_image_rgb() is None:
    st.error(f'The image {csi.get_image_path()} could not be read')
    st.stop()

st.caption(f'{csi.get_image_path()} - {csi.get_image_rgb().shape}')

# The pipeline draws new random numbers on every call, so the augmented image
# is kept in the session state: it is recomputed on a click of the button and
# whenever another image is chosen, but not on every unrelated rerun.
if st.button('Regenerate augmentation') or (
    st.session_state.get('augmented_image_name') != image_name
):
    st.session_state.augmented_image_name = image_name
    st.session_state.augmented_image = with_masks_drawn(augment(csi))

container_left, container_right = st.columns(2)
container_left.image(with_masks_drawn(csi), caption='Original')
container_right.image(st.session_state.augmented_image, caption='Augmented')
