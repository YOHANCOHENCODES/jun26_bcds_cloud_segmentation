from typing import Optional

import cv2
import numpy as np
import torch
from config import FIGURES_DIR, TEST_IMAGES_DIR
from loaders import load_model_pspnet, load_test_image_names

import streamlit as st
from src.data.cloud_segmentation_image import (
    LABEL_CLASS_IDS,
    CloudSegmentationImage,
)
from src.models.predict_model import (
    MIN_DETECTION_AREA,
    detected_labels,
    predict_mask_stack,
)
from src.visualization.image_display import draw_segmap_on_image

# The image size the model was trained on. An uploaded image is resized to it,
# because the normalization and the encoder expect exactly this input.
IMAGE_SHAPE = (384, 576)

LABEL_NAMES = list(LABEL_CLASS_IDS.keys())


def decode_image_rgb(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode the bytes of an image file into an RGB array of IMAGE_SHAPE."""
    buffer = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return None

    image_rgb = cv2.cvtColor(
        src=image_bgr,
        code=cv2.COLOR_BGR2RGB,
    )

    if image_rgb.shape[:2] != IMAGE_SHAPE:
        height, width = IMAGE_SHAPE
        image_rgb = cv2.resize(
            image_rgb,
            (width, height),
        )

    return image_rgb


@st.cache_data(show_spinner='Running the model on the image...')
def predict_masks(image_rgb: np.ndarray, _model, _device) -> np.ndarray:
    """Cached wrapper around predict_mask_stack()."""
    return predict_mask_stack(
        _model,
        image_rgb,
        _device,
    )


st.write('## PSPNet and its results')

# st.markdown(f"""
# The trained PSPNet (`resnet34` encoder) is applied to an image from the
# **test** directory. Those images have *no labels*, so there is nothing to
# decode a mask from: the masks below are what the model itself predicts.

# A pixel can belong to several cloud types at once, so this is a **multi-label**
# problem - each of the 4 outputs is an independent binary mask, thresholded at
# `0.5`. *Detected* means that the predicted mask of a cloud type covers at least
# {MIN_DETECTION_AREA} pixels.
# """)


tab_properties, tab_architecture, tab_library = st.tabs(
    ['PSPNet properties', 'Architecture', 'The library: smp']
)

with tab_properties:
    st.markdown("""
- **The Pyramid Pooling Module is the idea of the network.**
    - It pools the feature map at four scales at once - `1×1`, `2×2`, `3×3` and `6×6`
    - The coarsest level is a global pooling over the whole image
    - The finer ones split it into sub-regions
- **Every level is brought back to a common shape.**
    - Each one is reduced by a `1×1` convolution
    - then bilinearly upsampled to the size of the feature map
    - and finally all of them are concatenated
- **This gives the network a global prior of the scene.**
    - It reads the fine detail and the overall context of an image at the same time
    - instead of deciding pixel by pixel from a local neighbourhood
- **The backbone is a pre-trained ResNet with dilated convolutions.**
    - Dilation widens the receptive field without adding parameters
    - so more context reaches the pooling module
- **That is the difference to an FCN.**
    - An FCN works at a single scale
    - PSPNet combines the local and the global view

*Sources: Zhao et al., [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105), CVPR 2017 -
the original paper - and
[GeeksforGeeks: PSPNet - Pyramid Scene Parsing Network for Image Segmentation](https://www.geeksforgeeks.org/computer-vision/pspnet-pyramid-scene-parsing-network-for-image-segmentation/)*
""")

with tab_architecture:
    # The SVG export of `06-model-pspnet-rosner-architecture.drawio`, not the
    # PNG: `st.image()` resamples a raster image down to the requested width
    # before sending it, which turns the thin lines and the small labels of the
    # diagram into mush. An .svg path is inlined as a data URI untouched, so
    # the width below only scales the vector and costs no sharpness.
    #
    #   drawio --export --format svg --svg-theme light --border 10 \
    #       --output reports/figures/rosner-pspnet-architecture.svg \
    #       reports/figures/06-model-pspnet-rosner-architecture.drawio
    st.image(
        FIGURES_DIR / 'rosner-pspnet-architecture.svg',
        width=800,
    )

with tab_library:
    st.markdown('TODO')

st.divider()

# --------------------------------------------------------------------------- #
# The model, live
# --------------------------------------------------------------------------- #

st.header('Live example')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = load_model_pspnet(device)


# Pick one of the test images shipped with the app.
# They are unlabeled, so there is nothing to compare the prediction against.
test_image_names = load_test_image_names()


if test_image_names:
    selected_name = st.selectbox(
        'Test image',
        test_image_names,
    )
else:
    selected_name = None
    st.info(
        'No test images available here. Please upload a satellite image '
        'instead.'
    )


uploaded_file = st.file_uploader(
    'Or upload an own image',
    type=['jpg', 'jpeg', 'png'],
)


# An uploaded image wins over the selected one.
if uploaded_file is not None:
    image_name = uploaded_file.name
    image_bytes = uploaded_file.getvalue()

elif selected_name is not None:
    image_name = selected_name
    image_bytes = (TEST_IMAGES_DIR / selected_name).read_bytes()

else:
    st.stop()


image_rgb = decode_image_rgb(image_bytes)


if image_rgb is None:
    st.error(f'The image {image_name} could not be read')
    st.stop()


# Let the model predict the masks of the image.
mask_stack = predict_masks(
    image_rgb,
    model,
    device,
)

labels = detected_labels(
    mask_stack,
    LABEL_NAMES,
)


# The predicted masks have the same shape as the decoded masks of a labeled
# image, so the image and its predictions can be wrapped in a
# CloudSegmentationImage and drawn like a ground-truth segmentation map.
csi_predicted = CloudSegmentationImage.from_image_and_masks(
    image_rgb=image_rgb,
    mask_stack=mask_stack,
    image_name=image_name,
    # Bookkeeping only: the image itself is handed over above, so this path
    # is never read from.
    image_path=TEST_IMAGES_DIR,
)


predicted_image_rgb = draw_segmap_on_image(csi_predicted)


if predicted_image_rgb is None:
    st.error(f'The segmentation map of {image_name} could not be drawn')
    st.stop()


st.write(f'### Image {image_name}')

st.write(f'**Detected:** {", ".join(labels) if labels else "none"}')


container_left, container_right = st.columns(2)


container_left.image(
    image_rgb,
    caption='Original',
)


container_right.image(
    predicted_image_rgb,
    caption='Predicted masks',
)
