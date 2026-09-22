from config import FIGURES_DIR

import streamlit as st

# This page only presents results - it does not run the U-Net. The trained
# Keras model stays out of the app on purpose: it would pull TensorFlow in
# next to torch, for plots that were already made during training and are
# sitting in `reports/figures/`. So they are shown from there.

# st.write('## U-Net model')

# st.markdown("""
# *TODO: Text about the architecture and the training setup.*
# """)

st.markdown(
    """
    <p style="
        font-size:50px;
        font-weight:600;
        text-align:left;
    ">
        U-Net architecture
    </p>
    """,
    unsafe_allow_html=True,
)

st.image(
    FIGURES_DIR / "mahsa_unet_architecture.png",
    width=1500,
)


st.markdown(
    """
    <p style="
        font-size:50px;
        font-weight:600;
        text-align:left;
    ">
        Model's validation plots
    </p>
    """,
    unsafe_allow_html=True,
)

st.image(
    FIGURES_DIR / "mahsa_loss_and_dice_plots.png",
    width=1500,
)

# st.image(
#     FIGURES_DIR / 'mahsa_prediction_on_test_images.png',
#     caption='Predictions on test images',
#     width=1000,
# )

# st.markdown("""
# *TODO: What the distribution means for the training.*
# """)


from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from tensorflow import keras


# ============================================================
# PATHS
# ============================================================

# unet.py is here:
# project/src/streamlit/views/unet.py
#
# therefore parents[3] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_unet_cloud_segmentation_4class.keras"
)

TEST_IMAGES_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "test_images"
)


# ============================================================
# SETTINGS
# ============================================================

# Cloud classes
LABEL_NAMES = [
    "Fish",
    "Flower",
    "Gravel",
    "Sugar",
]

# Probability threshold:
# pixel probability >= 0.5 --> cloud
THRESHOLD = 0.5

# Minimum number of predicted pixels required before
# saying that a cloud class was detected.
MIN_DETECTION_AREA = 500


# Colors used only for visualization.
# RGB
MASK_COLORS = np.array(
    [
        [255, 0, 0],      # Fish
        [0, 255, 0],      # Flower
        [0, 0, 255],      # Gravel
        [255, 255, 0],    # Sugar
    ],
    dtype=np.uint8,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_unet_model():
    """
    Load the trained Keras U-Net model once.

    compile=False is sufficient for inference and avoids
    requiring the training loss/metrics.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    return model


# ============================================================
# LOAD IMAGE NAMES
# ============================================================

@st.cache_data
def load_test_image_names():

    if not TEST_IMAGES_DIR.exists():
        return []

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
    }

    names = [
        p.name
        for p in TEST_IMAGES_DIR.iterdir()
        if p.suffix.lower() in extensions
    ]

    return sorted(names)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def decode_image_rgb(
    image_bytes: bytes,
) -> Optional[np.ndarray]:
    """
    Decode image bytes into an RGB NumPy array.
    """

    buffer = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image_bgr = cv2.imdecode(
        buffer,
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        return None

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return image_rgb


def prepare_image_for_model(
    image_rgb: np.ndarray,
    model,
):
    """
    Resize and normalize the image exactly to the model input.

    Expected model shape:
        (None, height, width, 3)

    Example:
        (None, 384, 576, 3)
    """

    input_shape = model.input_shape

    height = input_shape[1]
    width = input_shape[2]

    resized = cv2.resize(
        image_rgb,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    # IMPORTANT:
    # This must be the SAME normalization used during training.
    image_float = (
        resized.astype(np.float32) / 255.0
    )

    # Add batch dimension:
    #
    # (384, 576, 3)
    #       ↓
    # (1, 384, 576, 3)

    batch = np.expand_dims(
        image_float,
        axis=0,
    )

    return resized, batch


# ============================================================
# MODEL PREDICTION
# ============================================================

@st.cache_data(
    show_spinner="Running U-Net prediction..."
)
def predict_masks(
    image_rgb: np.ndarray,
    _model,
):

    resized_image, batch = (
        prepare_image_for_model(
            image_rgb,
            _model,
        )
    )

    prediction = _model.predict(
        batch,
        verbose=0,
    )

    prediction = np.asarray(prediction)

    # Remove batch dimension
    prediction = prediction[0]

    # Normally Keras segmentation output:
    #
    # (height, width, 4)
    #
    # but this also handles:
    #
    # (4, height, width)

    if (
        prediction.ndim == 3
        and prediction.shape[0] == 4
        and prediction.shape[-1] != 4
    ):
        prediction = np.transpose(
            prediction,
            (1, 2, 0),
        )

    if prediction.ndim != 3:
        raise ValueError(
            "Unexpected model output shape: "
            f"{prediction.shape}"
        )

    if prediction.shape[-1] != 4:
        raise ValueError(
            "Expected 4 output channels, "
            f"but model returned {prediction.shape}"
        )

    # Convert probability maps to binary masks
    binary_masks = (
        prediction >= THRESHOLD
    ).astype(np.uint8)

    return (
        resized_image,
        prediction,
        binary_masks,
    )


# ============================================================
# DETECT LABELS
# ============================================================

def get_detected_labels(
    masks: np.ndarray,
):

    labels = []

    for index, label in enumerate(
        LABEL_NAMES
    ):

        pixel_count = int(
            masks[:, :, index].sum()
        )

        if pixel_count >= MIN_DETECTION_AREA:
            labels.append(label)

    return labels


# ============================================================
# DRAW SEGMENTATION OVERLAY
# ============================================================

def create_segmentation_overlay(
    image_rgb: np.ndarray,
    masks: np.ndarray,
):

    overlay = image_rgb.copy()

    alpha = 0.45

    for class_index in range(
        len(LABEL_NAMES)
    ):

        mask = (
            masks[:, :, class_index]
            .astype(bool)
        )

        if not np.any(mask):
            continue

        color = MASK_COLORS[
            class_index
        ].astype(np.float32)

        overlay_pixels = (
            (1 - alpha)
            * overlay[mask].astype(np.float32)
            + alpha * color
        )

        overlay[mask] = np.clip(
            overlay_pixels,
            0,
            255,
        ).astype(np.uint8)

    return overlay


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.title(
    "U-Net predictions on unseen images"
)

st.markdown(
    f"""
The trained **U-Net** model is applied to an image from the
test dataset or to an image uploaded by the user.

The model predicts four independent cloud masks:

**Fish (red), Flower (green), Gravel (blue) and Sugar (yellow)**

This is a **multi-label segmentation problem**, meaning that
a pixel can belong to more than one cloud class.

Each predicted probability map is thresholded at
`{THRESHOLD}`.

A cloud type is reported as **detected** when its predicted
mask contains at least **{MIN_DETECTION_AREA} pixels**.
"""
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_unet_model()

except Exception as e:

    st.error(
        f"Could not load U-Net model:\n\n{e}"
    )

    st.stop()


# Optional debugging information
# Remove later if you do not want it displayed.

with st.expander(
    "Model information"
):

    st.write(
        "Model:",
        MODEL_PATH.name,
    )

    st.write(
        "Input shape:",
        model.input_shape,
    )

    st.write(
        "Output shape:",
        model.output_shape,
    )


# ============================================================
# SELECT TEST IMAGE
# ============================================================

image_names = (
    load_test_image_names()
)

if image_names:

    selected_name = st.selectbox(
        "Test image",
        image_names,
    )

else:

    selected_name = None

    st.warning(
        "No test images were found."
    )


# ============================================================
# UPLOAD IMAGE
# ============================================================

uploaded_file = st.file_uploader(
    "Or upload your own image",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
)


# Uploaded image has priority
if uploaded_file is not None:

    image_name = (
        uploaded_file.name
    )

    image_bytes = (
        uploaded_file.getvalue()
    )

elif selected_name is not None:

    image_name = selected_name

    image_path = (
        TEST_IMAGES_DIR
        / selected_name
    )

    image_bytes = (
        image_path.read_bytes()
    )

else:

    st.stop()


# ============================================================
# DECODE IMAGE
# ============================================================

image_rgb = decode_image_rgb(
    image_bytes
)

if image_rgb is None:

    st.error(
        f"Could not read image: "
        f"{image_name}"
    )

    st.stop()


# ============================================================
# RUN MODEL
# ============================================================

try:

    (
        model_image,
        probabilities,
        masks,
    ) = predict_masks(
        image_rgb,
        model,
    )

except Exception as e:

    st.error(
        f"Prediction failed:\n\n{e}"
    )

    st.stop()


# ============================================================
# DETECT CLOUD CLASSES
# ============================================================

detected = get_detected_labels(
    masks
)


# ============================================================
# CREATE OVERLAY
# ============================================================

predicted_image = (
    create_segmentation_overlay(
        model_image,
        masks,
    )
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

st.write(
    f"## Image {image_name}"
)

if detected:

    st.write(
        "**Detected:** "
        + ", ".join(detected)
    )

else:

    st.write(
        "**Detected:** none"
    )


left, right = st.columns(2)


with left:

    st.image(
        model_image,
        caption="Original",
        use_container_width=True,
    )


with right:

    st.image(
        predicted_image,
        caption="Predicted masks (U-Net)",
        use_container_width=True,
    )


# ============================================================
# OPTIONAL: SHOW EACH MASK
# ============================================================

with st.expander(
    "Show individual predicted masks"
):

    cols = st.columns(4)

    for index, label in enumerate(
        LABEL_NAMES
    ):

        cols[index].image(
            masks[:, :, index]
            * 255,
            caption=label,
            clamp=True,
            use_container_width=True,
        )