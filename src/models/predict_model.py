import numpy as np
import torch

from src.data.cloud_segmentation_dataset import IMAGENET_MEAN, IMAGENET_STD

# Minimum number of predicted pixels for a cloud type to count as "detected"
# in an image. Matches `MIN_LABELED_SEGMENT_AREA` in
# `src.visualization.image_display`, which uses the same idea for labeling
# ground-truth segments.
MIN_DETECTION_AREA = 500


def predict_mask_stack(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    """Run `model` on a single RGB image and return its predicted per-label
    mask stack, shape `(num_classes, H, W)`, thresholded to 0/1.

    `image_rgb` must already have the size and channel order (RGB) the model
    was trained on; it is normalized here the same way
    `CloudSegmentationDataset` normalizes training images.
    """
    image_tensor = torch.from_numpy(np.ascontiguousarray(image_rgb))
    image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
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
