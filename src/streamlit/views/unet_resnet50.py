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


# ============================================================
# CONSTANTS
# ============================================================

# The image size the model was trained on.
IMAGE_SHAPE = (384, 576)

LABEL_NAMES = list(LABEL_CLASS_IDS.keys())


# ============================================================
# IMAGE DECODING
# ============================================================

def decode_image_rgb(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode image bytes into an RGB array of IMAGE_SHAPE."""

    buffer = np.frombuffer(image_bytes, np.uint8)

    image_bgr = cv2.imdecode(
        buffer,
        cv2.IMREAD_COLOR,
    )

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


# ============================================================
# MODEL PREDICTION
# ============================================================

@st.cache_data(show_spinner="Running the model on the image...")
def predict_masks(
    image_rgb: np.ndarray,
    _model,
    _device,
) -> np.ndarray:
    """Cached wrapper around predict_mask_stack()."""

    return predict_mask_stack(
        _model,
        image_rgb,
        _device,
        encoder_name=UNET_ENCODER,
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.write("## U-Net (ResNet50) Predictions")


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("""
### 🏗 Model Architecture
""")

architecture_col1, architecture_col2 = st.columns([1.25, 1])


# ------------------------------------------------------------
# ARCHITECTURE IMAGE
# ------------------------------------------------------------

with architecture_col1:

    st.image(
        "assets/architecture/unet_resnet50_architecture.png",
        caption="U-Net with ResNet50 encoder",
        use_container_width=True,
    )


# ------------------------------------------------------------
# ARCHITECTURE DESCRIPTION
# ------------------------------------------------------------

with architecture_col2:

    st.markdown("""
    **U-Net + ResNet50 encoder**

    - ImageNet-pretrained weights
    - ResNet50 encoder for feature extraction
    - U-Net decoder for spatial reconstruction
    - Skip connections between encoder and decoder
    - 4 output classes:
      **Fish, Flower, Gravel, Sugar**
    - **Sigmoid activation → Multi-label segmentation**
    """)


col1, col2, col3 = st.columns(3)


# ============================================================
# TRAINING
# ============================================================

with col1:
    st.markdown("""
    ### ⚙️ Training

    - BCE + Dice Loss
    - AdamW
    - Learning rate: **8 × 10⁻⁵**
    - Up to **25 epochs**
    - Early stopping
    """)


# ============================================================
# VALIDATION RESULTS
# ============================================================

with col2:
    st.markdown("""
    ### 📊 Validation Results

    **Macro Dice**

    # 0.6118

    **Mean IoU**

    # 0.4432
    """)


# ============================================================
# FALSE POSITIVES + INTERPRETABILITY
# ============================================================

with col3:
    st.markdown("""
    ### 🔍 Key Finding

    Occasional high-confidence
    **false positives** were observed.

    **99.1%** of FP cases also
    contained ≥1 correct prediction.

    **Interpretability**

    - **Grad-CAM++**
    - **Occlusion Sensitivity**
    - Shared visual features may explain
      additional class predictions.
    """)


st.markdown("---")


# ============================================================
# PREDICTION INFORMATION
# ============================================================

st.markdown("""
### 🔎 Prediction

Select an image from the **test** or **train** set, or upload your own image.

The masks shown below are the model's predictions, not ground truth.

A pixel can belong to several cloud types at once, so each of the four
outputs is treated as an independent binary mask and thresholded at **0.5**.
""")


# ============================================================
# DEVICE & MODEL
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = load_model_unet_resnet50(device)


# ============================================================
# IMAGE SOURCE
# ============================================================

image_source = st.radio(
    "Image source",
    ["Test images", "Train images"],
    horizontal=True,
)


if image_source == "Test images":

    image_names = load_test_image_names()
    images_dir = TEST_IMAGES_DIR

else:

    image_names = load_train_image_names()
    images_dir = TRAIN_IMAGES_DIR


# ============================================================
# IMAGE SELECTION
# ============================================================

if image_names:

    selected_name = st.selectbox(
        "Image",
        image_names,
    )

else:

    selected_name = None

    st.info(
        "No images available here. Please upload a satellite image instead."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Or upload an own image",
    type=["jpg", "jpeg", "png"],
)


# Uploaded image has priority over selected image.
if uploaded_file is not None:

    image_name = uploaded_file.name
    image_bytes = uploaded_file.getvalue()

elif selected_name is not None:

    image_name = selected_name
    image_bytes = (
        images_dir / selected_name
    ).read_bytes()

else:

    st.stop()


# ============================================================
# IMAGE DECODING
# ============================================================

image_rgb = decode_image_rgb(
    image_bytes
)


if image_rgb is None:

    st.error(
        f"The image {image_name} could not be read."
    )

    st.stop()


# ============================================================
# MODEL PREDICTION
# ============================================================

mask_stack = predict_masks(
    image_rgb,
    model,
    device,
)


# ============================================================
# DETECTED LABELS
# ============================================================

labels = detected_labels(
    mask_stack,
    LABEL_NAMES,
)


# ============================================================
# CREATE SEGMENTATION IMAGE
# ============================================================

csi_predicted = CloudSegmentationImage.from_image_and_masks(
    image_rgb=image_rgb,
    mask_stack=mask_stack,
    image_name=image_name,

    # Bookkeeping only:
    # the image itself is already provided above.
    image_path=images_dir,
)


predicted_image_rgb = draw_segmap_on_image(
    csi_predicted
)


if predicted_image_rgb is None:

    st.error(
        f"The segmentation map of {image_name} could not be drawn."
    )

    st.stop()


# ============================================================
# DISPLAY PREDICTION RESULTS
# ============================================================

st.write(
    f"### Image {image_name}"
)


st.write(
    f"**Detected:** "
    f"{', '.join(labels) if labels else 'none'}"
)


container_left, container_right = st.columns(2)


with container_left:

    st.image(
        image_rgb,
        caption="Original",
        use_container_width=True,
    )


with container_right:

    st.image(
        predicted_image_rgb,
        caption="Predicted masks (U-Net + ResNet50)",
        use_container_width=True,
    )


# ============================================================
# INTERPRETABILITY
# ============================================================

st.markdown("---")

st.markdown("""
### 🧠 Interpretability

To investigate the additional class predictions, we also examined
the model using **Grad-CAM++** and **Occlusion Sensitivity**.

**Grad-CAM++** highlights image regions that influence a class prediction,
while **Occlusion Sensitivity** measures the effect of masking image regions.

Both methods indicated that some visual features are shared between
different cloud classes. This feature overlap may explain why the model
sometimes predicts additional classes that are not present in the ground truth.
""")


# ============================================================
# INTERPRETABILITY IMAGES
# ============================================================

interp_col1, interp_col2 = st.columns(2)


with interp_col1:

    st.markdown("#### Grad-CAM++")

    st.image(
        "assets/interpretability/gradcam_plus_plus.png",
        use_container_width=True,
    )


with interp_col2:

    st.markdown("#### Occlusion Sensitivity")

    st.image(
        "assets/interpretability/occlusion_sensitivity.png",
        use_container_width=True,
    )


# ============================================================
# FINAL NOTE
# ============================================================

st.caption(
    "Interpretability analysis is used here as a qualitative analysis "
    "to better understand false-positive predictions."
)