import albumentations as A
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
# 2.2.6 Data Augmentation - the report section that introduces the live example
# below. Moved here from `eda.py`, where chapter 2 otherwise lives.
# --------------------------------------------------------------------------- #

AUGMENTATIONS = [
    (
        'HorizontalFlip',
        0.30,
        'Geometric',
        'Satellite images have no canonical orientation, so a mirrored scene '
        'stays physically plausible.',
    ),
    (
        'VerticalFlip',
        0.30,
        'Geometric',
        'Same reasoning as the horizontal flip; doubles the orientation '
        'variety at no cost.',
    ),
    (
        'RandomBrightnessContrast',
        0.80,
        'Photometric',
        'Simulates different illumination and imaging conditions. Applied to '
        'the image only, never to the masks.',
    ),
    (
        'Rotate',
        0.60,
        'Geometric',
        "Cloud formations occur in any orientation; reduces the model's "
        'dependence on a fixed layout.',
    ),
]

PIPELINE_SNIPPET = """import albumentations as A

pipeline = A.Compose([
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.8),
    A.Rotate(p=0.6),
])

# Call the pipeline giving it an image and the stack of its masks
# NOTE: `csi` is an instance of our custom class `CloudSegmentationImage` which
# gathers the image together with the labels (the masks)
transformed_data = pipeline(
    image=csi.get_image_rgb(),
    masks=csi.get_mask_stack(),
)
"""

DATASET_SNIPPET = """train_dataset = CloudSegmentationDataset(
    train_labels_df, TRAIN_PATH, pipeline=pipeline
)

# No augmentation for validation, so it reflects real-world performance
val_dataset = CloudSegmentationDataset(val_labels_df, TRAIN_PATH, pipeline=None)
"""


def _section(title: str, number: str = '') -> None:
    label = f'{number} {title}'.strip()
    st.markdown(f'### {label}')


def _takeaway(text: str) -> None:
    """Highlighted one-line conclusion, used at the end of the section."""
    st.info(f'**Takeaway** - {text}')


def _s226_augmentation() -> None:
    _section('Data Augmentation', '2.2.6')

    st.markdown("""
Augmentation was applied **to the training data only**, to increase the variety of the images and
improve generalisation. The validation data is only resized and normalised, never augmented, so that
its score stays comparable across epochs and reflects performance on unmodified images.
""")

    t1, t2, t3 = st.tabs(['Pipeline', 'Why Albumentations', 'Mask handling'])

    with t1:
        st.markdown(
            'Each transformation has its own probability `p`, so an image receives a different '
            'combination of transformations every time it is requested.'
        )
        st.dataframe(
            pd.DataFrame(
                AUGMENTATIONS, columns=['Transformation', 'p', 'Type', 'Reason']
            ),
            hide_index=True,
            width='stretch',
            column_config={
                'p': st.column_config.NumberColumn(format='%.2f', width='small')
            },
        )
        st.code(PIPELINE_SNIPPET, language='python')
        st.caption('Figure 7 of the report: the augmentation pipeline.')
        st.markdown("""
**On-the-fly instead of stored augmentation.** The transformations are applied inside the dataset
whenever a training image is requested. The dataset on disk does not grow, but the model sees a
different version of the same image in different epochs.

**The probabilities were chosen heuristically** as a reasonable initial configuration and were not
systematically optimised - a with/without comparison and a probability sweep are open next steps.
""")

    with t2:
        st.markdown("""
**Albumentations** was chosen because it applies geometric transformations **consistently to an image
and all of its masks**, which is essential for segmentation: the spatial alignment between input and
labels must survive every transformation.

- **Geometric** transformations (flip, rotate) are applied **identically** to the image and to all four
  masks.
- **Photometric** transformations (brightness, contrast) are applied **to the image only**, because they
  change appearance but must not change the labels.

`imgaug` was considered early in the project; Albumentations provided the required functionality and
fitted the Python environment used here.

**Package versus import name** (relevant for reproducibility): the project installs
**`albumentationsx`** as specified in `requirements.txt`, which provides a compatible interface and is
imported as `albumentations`.
""")
        st.code('import albumentations as A', language='python')

    with t3:
        st.markdown("""
Three details keep the labels valid under augmentation:

1. **Simultaneous transformation** - a geometric transformation is applied to the image and its masks
   in the same call, so alignment is preserved.
2. **Corner filling on rotation** - newly created corner regions are filled with the background value,
   which is `0` in the masks, so **no artificial cloud regions** are introduced.
3. **Nearest-neighbour resampling for masks** - this keeps the masks strictly binary; any interpolating
   filter would create invalid intermediate values between 0 and 1.

The pipeline is passed to the custom `CloudSegmentationDataset` and applied in its `__getitem__`
method, so augmentation happens on the fly during training - and only for the training set.
""")
        st.code(DATASET_SNIPPET, language='python')
        st.caption(
            'Figure 8 of the report: augmentation for training, no augmentation for validation.'
        )

    _takeaway(
        'Every transformation is physically plausible for satellite imagery, and the masks follow the '
        'geometry exactly - the two conditions under which augmentation helps instead of hurting.'
    )


# --------------------------------------------------------------------------- #
# The live example
# --------------------------------------------------------------------------- #

# Create the augmentation pipeline
pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.8),
        A.Rotate(p=0.6),
    ]
)


def augment(csi: CloudSegmentationImage, image_rgb, image_masks):
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


# Create the Streamlit page title
st.write('## Augmentation')

# The report section comes first, the interactive example below it
_s226_augmentation()
st.divider()

st.write('### Live example')

train_labels_df = load_train_labels()
image_name = st.selectbox('Training image', load_train_image_names())

csi = CloudSegmentationImage(
    train_labels_df, first_row_id(train_labels_df, image_name), TRAIN_IMAGES_DIR
)

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

st.write(f'#### Image {csi.get_name()}')
st.caption(
    f'{csi.get_image_path()} - {loaded_image_rgb.shape}, {loaded_image_rgb.dtype}'
)

# Create a Streamlit 'container' to have the original images on the left and the
# augmented  ones at the right
container_left, container_right = st.columns(2)

# Run the augmentation once on first load and again on every button click - the
# pipeline uses random probabilities, so each run can produce a different
# result. The image name is kept alongside it, so that picking another image
# augments that one instead of showing the previous result.
regenerate = st.button('Regenerate augmentation')
if regenerate or st.session_state.get('augmented_image_name') != image_name:
    st.session_state.augmented_image_name = image_name
    st.session_state.augmented_image_with_segmap = augment(
        csi, loaded_image_rgb, loaded_image_masks
    )

augmented_image_with_segmap = st.session_state.augmented_image_with_segmap

# Display the images
container_left.image(loaded_image_rgb_with_segmap, caption='Original')
container_right.image(augmented_image_with_segmap, caption='Segmentation map')
