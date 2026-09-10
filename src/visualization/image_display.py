import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

from src.data.cloud_segmentation_image import (
    LABEL_CLASS_IDS,
    CloudSegmentationImage,
)

# The RGB colors used to draw each cloud label onto an image
LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    'fish': (255, 0, 0),
    'flower': (0, 255, 0),
    'gravel': (0, 0, 255),
    'sugar': (255, 255, 0),
}

# The smallest pixel area a segment must have to get its label name drawn
MIN_LABELED_SEGMENT_AREA = 500


def display_image(csi: CloudSegmentationImage):
    image = csi.get_image_rgb()
    print(csi.get_image_path(), image.shape, image.dtype)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def display_masks_by_rows(csi: CloudSegmentationImage):
    display(Markdown(f'## Image {csi.get_name()}'))
    for mask in csi.get_all_masks():
        img = csi.get_image_rgb()
        if img is None:
            continue

        # Create a figure with 3 sublots in one row
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        # Display the image
        axes[0].imshow(img)
        axes[0].set_title(csi.image_name)

        # Display the mask
        axes[1].imshow(mask.mask_pixels, cmap='gray')
        axes[1].set_title(f'Mask: {csi.image_name}')

        # Display the image with a transparent overlay
        axes[2].imshow(img)
        axes[2].imshow(mask.mask_pixels, cmap='Reds', alpha=0.4)
        axes[2].set_title(f'Mask region labeled as "{mask.label_name}"')

        # Turn off the scales for all axes
        for ax in axes:
            ax.axis('off')
        plt.show()


def display_masks_overlay(csi: CloudSegmentationImage):
    result = draw_segmap_on_image(csi, alpha=0.4, draw_label_names=True)
    if result is None:
        return

    plt.imshow(result)
    plt.axis('off')
    plt.title(f'Segmentation map of {csi.get_name()}')
    plt.show()


def draw_segmap_on_image(
    csi: CloudSegmentationImage,
    label_colors: dict[str, tuple[int, int, int]] | None = None,
    alpha: float = 0.5,
    draw_label_names: bool = False,
) -> np.ndarray | None:
    """Overlay the combined segmentation map of a `CloudSegmentationImage` on
    its image.

    `label_colors` maps a lowercase label name to an RGB color and defaults to
    `LABEL_COLORS`. `alpha` is the blend strength (0 = invisible, 1 = solid
    color). With `draw_label_names` the name of the label is written into the
    top left corner of every segment (see `draw_label_names_on_image()`).

    NOTE: Overlapping masks are already resolved by
    `CloudSegmentationImage.get_segmap()`, so each pixel only shows the label
    with the highest class id.
    """
    if label_colors is None:
        label_colors = LABEL_COLORS

    segmap = csi.get_segmap()

    img = csi.get_image_rgb()
    if img is None:
        return None

    overlay = img.copy()

    for label_name, class_id in LABEL_CLASS_IDS.items():
        if label_name in label_colors:
            overlay[segmap == class_id] = label_colors[label_name]

    # Alpha-blend the image with the colored overlay
    blended = (img * (1 - alpha) + overlay * alpha).astype(np.uint8)

    if draw_label_names:
        blended = draw_label_names_on_image(blended, segmap)

    return blended


def draw_label_names_on_image(
    image: np.ndarray,
    segmap: np.ndarray,
    min_segment_area: int = MIN_LABELED_SEGMENT_AREA,
) -> np.ndarray:
    """Write the label name into the center of every segment of `segmap`.

    The mask of a label can fall apart into several separate regions, so every
    connected region gets a name of its own. Regions smaller than
    `min_segment_area` pixels are skipped, because their name would be bigger
    than the region itself.

    NOTE: `image` and `segmap` must have the same height and width.
    """
    labeled = image.copy()

    # Scale the text with the image, so it stays readable on any image size
    font_scale = max(0.4, image.shape[0] / 800)
    thickness = max(2, round(2 * font_scale))

    for label_name, class_id in LABEL_CLASS_IDS.items():
        mask = (segmap == class_id).astype(np.uint8)
        _, _, stats, centroids = cv2.connectedComponentsWithStats(mask)

        # The first row of `stats`/`centroids` is the background of `mask`, so
        # it is left out. `centroids[i]` is the center of mass of region `i` -
        # unlike a bounding box corner, it stays inside the region even when
        # the mask is rotated or otherwise non-rectangular.
        for i in range(1, len(stats)):
            if stats[i, cv2.CC_STAT_AREA] < min_segment_area:
                continue
            center_x, center_y = centroids[i]
            draw_text(
                labeled, label_name, center_x, center_y, font_scale, thickness
            )

    return labeled


def draw_text(
    image: np.ndarray,
    text: str,
    center_x: float,
    center_y: float,
    font_scale: float,
    thickness: int,
) -> None:
    """Draw `text` into `image` (in place), centered on `(center_x,
    center_y)`."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    # `cv2.putText()` expects the bottom left corner of the text, so it has to
    # be offset by half the text's own size to end up centered on the point
    origin = (
        int(center_x - text_width / 2),
        int(center_y + (text_height + baseline) / 2),
    )

    # A thicker black text below the white one acts as an outline and keeps the
    # name readable on both the image and the colors of the overlay
    cv2.putText(
        image,
        text,
        origin,
        font,
        font_scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
