from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch

from src.data.cloud_segmentation_dataset import (
    DEFAULT_ENCODER_NAME,
    DEFAULT_ENCODER_WEIGHTS,
    get_normalization,
)

# Minimum number of predicted pixels for a cloud type to count as "detected"
# in an image. Matches `MIN_LABELED_SEGMENT_AREA` in
# `src.visualization.image_display`, which uses the same idea for labeling
# ground-truth segments.
MIN_DETECTION_AREA = 500

# Architecture the checkpoints in `models/` were trained with. A saved
# `state_dict` holds the weights only, so the model has to be rebuilt with
# exactly these settings before the weights fit back into it.
PSPNET_ENCODER = DEFAULT_ENCODER_NAME
NUM_CLASSES = 4


def load_pspnet(
    path: Path,
    device: torch.device,
    encoder_name: str = PSPNET_ENCODER,
    classes: int = NUM_CLASSES,
) -> torch.nn.Module:
    """Rebuild the PSPNet architecture and load the trained weights that
    `torch.save(model.state_dict(), path)` wrote to `path` into it.

    The returned model sits on `device` and is in eval mode, ready to be
    passed to `predict_mask_stack`. `encoder_weights` stays `None` here: the
    ImageNet weights the encoder started training from are overwritten by the
    checkpoint anyway, so there is no point in downloading them again.
    """
    model = smp.PSPNet(
        encoder_name=encoder_name, encoder_weights=None, classes=classes
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# Architecture the U-Net checkpoint was trained with (see load_unet below).
UNET_ENCODER = 'resnet50'


def load_unet(
    path: Path,
    device: torch.device,
    encoder_name: str = UNET_ENCODER,
    classes: int = NUM_CLASSES,
) -> torch.nn.Module:
    """Rebuild the U-Net (ResNet50 encoder) architecture and load the trained
    weights that `torch.save(model.state_dict(), path)` wrote to `path` into
    it. Mirrors `load_pspnet`, but for U-Net + ResNet50: same idea, different
    architecture and encoder. `encoder_weights` stays `None` for the same
    reason as in `load_pspnet` - the checkpoint already carries the
    fine-tuned weights.
    """
    model = smp.Unet(
        encoder_name=encoder_name, encoder_weights=None, classes=classes
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def predict_mask_stack(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
    encoder_name: str = PSPNET_ENCODER,
    encoder_weights: str = DEFAULT_ENCODER_WEIGHTS,
) -> np.ndarray:
    """Run `model` on a single RGB image and return its predicted per-label
    mask stack, shape `(num_classes, H, W)`, thresholded to 0/1.

    `image_rgb` must already have the size and channel order (RGB) the model
    was trained on; it is normalized here the same way
    `CloudSegmentationDataset` normalizes training images, so
    `encoder_name`/`encoder_weights` have to match the ones used for training.
    """
    image_tensor = torch.from_numpy(np.ascontiguousarray(image_rgb))
    image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0

    mean, std = get_normalization(encoder_name, encoder_weights)
    image_tensor = (image_tensor - mean) / std

    image_tensor = image_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(image_tensor))

    return (probs.squeeze(0) > threshold).cpu().numpy().astype(np.uint8)


def detected_labels(
    mask_stack: np.ndarray,
    label_names: list[str],
    min_area: int = MIN_DETECTION_AREA,
) -> list[str]:
    """Return the label names (in the order matching `mask_stack`'s channels)
    whose predicted mask covers at least `min_area` pixels."""
    return [
        label_name
        for channel, label_name in enumerate(label_names)
        if mask_stack[channel].sum() >= min_area
    ]