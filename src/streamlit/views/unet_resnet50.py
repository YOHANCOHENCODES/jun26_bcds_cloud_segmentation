import cv2
import numpy as np
import torch
from typing import Optional

from config import TEST_IMAGES_DIR, TRAIN_IMAGES_DIR
from loaders import (
    load_model_unet_resnet50,
    load_test_image_names,
    load_train_image_names,
)

import streamlit as st
from src.data.cloud_segmentation_image import (
    LABEL_CLASS_IDS,
    CloudSegmentationImage,
)
from src.models.predict_model import (
    MIN_DETECTION_AREA,
    UNET_ENCODER,
    detected_labels,
    predict_mask_stack,
)
from src.visualization.image_display import draw_segmap_on_image


# The image size the model was trained on. An uploaded image is resized to
# it, because the normalization and the encoder expect exactly this input.
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
        encoder_name=UNET_ENCODER,
    )


st.write('## U-Net (ResNet50) predictions')

st.markdown(f"""
The trained U-Net (`resnet50` encoder) is applied to a satellite image. Pick
an image from the **test** set (unlabeled) or the **train** set below, or
upload your own - the masks shown are always what the model itself predicts,
not a ground-truth comparison.

A pixel can belong to several cloud types at once, so this is a **multi-label**
problem - each of the 4 outputs is an independent binary mask, thresholded at
`0.5`. *Detected* means that the predicted mask of a cloud type covers at least
{MIN_DETECTION_AREA} pixels.
""")


device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

model = load_model_unet_resnet50(device)


image_source = st.radio(
    'Image source',
    ['Test images', 'Train images'],
    horizontal=True,
)


if image_source == 'Test images':
    image_names = load_test_image_names()
    images_dir = TEST_IMAGES_DIR
else:
    image_names = load_train_image_names()
    images_dir = TRAIN_IMAGES_DIR


if image_names:
    selected_name = st.selectbox(
        'Image',
        image_names,
    )
else:
    selected_name = None
    st.info(
        'No images available here. Please upload a satellite image instead.'
    )


uploaded_file = st.file_uploader(
    'Or upload an own image',
    type=['jpg', 'jpeg', 'png'],
)


# An uploaded image wins over the selected one - it is the more recent choice
if uploaded_file is not None:
    image_name = uploaded_file.name
    image_bytes = uploaded_file.getvalue()

elif selected_name is not None:
    image_name = selected_name
    image_bytes = (images_dir / selected_name).read_bytes()

else:
    st.stop()


image_rgb = decode_image_rgb(image_bytes)


if image_rgb is None:
    st.error(
        f'The image {image_name} could not be read'
    )
    st.stop()


mask_stack = predict_masks(
    image_rgb,
    model,
    device,
)

labels = detected_labels(
    mask_stack,
    LABEL_NAMES,
)


csi_predicted = CloudSegmentationImage.from_image_and_masks(
    image_rgb=image_rgb,
    mask_stack=mask_stack,
    image_name=image_name,
    # Bookkeeping only: the image itself is handed over above, so this path
    # is never read from - and an upload has no directory of its own anyway.
    image_path=images_dir,
)


predicted_image_rgb = draw_segmap_on_image(
    csi_predicted
)


if predicted_image_rgb is None:
    st.error(
        f'The segmentation map of {image_name} could not be drawn'
    )
    st.stop()


st.write(
    f'### Image {image_name}'
)

st.write(
    f'**Detected:** {", ".join(labels) if labels else "none"}'
)


container_left, container_right = st.columns(2)


container_left.image(
    image_rgb,
    caption='Original',
)


container_right.image(
    predicted_image_rgb,
    caption='Predicted masks (U-Net + ResNet50)',
)