import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.data.image_masks import LABEL_CLASS_IDS, ImageMask, ImageMasks

# The RGB colors used to draw each cloud label onto an image
LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    'fish': (255, 0, 0),
    'flower': (0, 255, 0),
    'gravel': (0, 0, 255),
    'sugar': (255, 255, 0),
}

# The smallest pixel area a segment must have to get its label name drawn
MIN_LABELED_SEGMENT_AREA = 500


def display_masked_image(image_data: ImageMask):
    '''Display given image with 3 visualizations: The image (left), the mask 
    (center) and the image with the mask as an overlay (right)'''
    # Create a figure with 3 sublots in one row
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # Read the image
    img = plt.imread(image_data.path / image_data.filename)

    # Display the image
    axes[0].imshow(img)
    axes[0].set_title(image_data.filename)

    # Display the mask
    axes[1].imshow(image_data.mask_pixels, cmap='gray')
    axes[1].set_title(f'Mask: {image_data.filename}')

    # Display the image with a transparent overlay
    axes[2].imshow(img)
    axes[2].imshow(image_data.mask_pixels, cmap='Reds', alpha=0.4)
    axes[2].set_title(f'Mask region labeled as "{image_data.label_name}"')

    # Turn off the scales for all axes
    for ax in axes:
        ax.axis('off')
    plt.show()


def display_seg_masked_image(image_masks: ImageMasks):
    result = draw_segmap_on_image(
        image_masks, alpha=0.4, draw_label_names=True)

    plt.imshow(result)
    plt.axis('off')
    plt.title(f'Segmentation map of {image_masks.get_name()}')
    plt.show()


def draw_mask_on_image(
        image_data: ImageMask,
        color: tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.5
) -> np.ndarray:
    '''Overlay an `ImageMask`'s binary mask on its image.

    `color` is the RGB color used for the mask and `alpha` is the blend
    strength (0 = invisible, 1 = solid color).
    '''
    img = plt.imread(image_data.path / image_data.filename)
    overlay = img.copy()
    overlay[image_data.mask_pixels > 0] = color

    return (img * (1 - alpha) + overlay * alpha).astype(np.uint8)


def draw_segmap_on_image(
        image_masks: ImageMasks,
        label_colors: dict[str, tuple[int, int, int]] | None = None,
        alpha: float = 0.5,
        draw_label_names: bool = False
) -> np.ndarray:
    '''Overlay the combined segmentation map of an `ImageMasks` on its image.

    `label_colors` maps a lowercase label name to an RGB color and defaults to
    `LABEL_COLORS`. `alpha` is the blend strength (0 = invisible, 1 = solid
    color). With `draw_label_names` the name of the label is written into the
    top left corner of every segment (see `draw_label_names_on_image()`).

    NOTE: Overlapping masks are already resolved by `ImageMasks.get_segmap()`,
    so each pixel only shows the label with the highest class id.
    '''
    if label_colors is None:
        label_colors = LABEL_COLORS

    segmap = image_masks.get_segmap()

    img = plt.imread(image_masks.get_image_path())
    overlay = img.copy()

    for label_name, class_id in LABEL_CLASS_IDS.items():
        if label_name in label_colors:
            overlay[segmap == class_id] = label_colors[label_name]

    blended = (img * (1 - alpha) + overlay * alpha).astype(np.uint8)

    if draw_label_names:
        blended = draw_label_names_on_image(blended, segmap)

    return blended


def draw_label_names_on_image(
        image: np.ndarray,
        segmap: np.ndarray,
        min_segment_area: int = MIN_LABELED_SEGMENT_AREA
) -> np.ndarray:
    '''Write the label name into the top left corner of every segment of
    `segmap`.

    The mask of a label can fall apart into several separate regions, so every
    connected region gets a name of its own. Regions smaller than
    `min_segment_area` pixels are skipped, because their name would be bigger
    than the region itself.

    NOTE: `image` and `segmap` must have the same height and width.
    '''
    labeled = image.copy()

    # Scale the text with the image, so it stays readable on any image size
    font_scale = max(0.4, image.shape[0] / 800)
    thickness = max(2, round(2 * font_scale))

    for label_name, class_id in LABEL_CLASS_IDS.items():
        mask = (segmap == class_id).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)

        # The first row of `stats` is the background of `mask`, so it is left
        # out. The other rows hold one region each as (x, y, width, height,
        # area), with (x, y) being its top left corner.
        for x, y, _, _, area in stats[1:]:
            if area < min_segment_area:
                continue
            _draw_text(labeled, label_name, x, y, font_scale, thickness)

    return labeled


def _draw_text(
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        font_scale: float,
        thickness: int
) -> None:
    '''Draw `text` into `image` (in place) with its top left corner at
    `(x, y)`.'''
    font = cv2.FONT_HERSHEY_SIMPLEX
    (_, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness)

    # `cv2.putText()` expects the bottom left corner of the text, so the text
    # has to be moved down by its own height to start at `(x, y)`
    origin = (int(x), int(y) + text_height + baseline)

    # A thicker black text below the white one acts as an outline and keeps the
    # name readable on both the image and the colors of the overlay
    cv2.putText(image, text, origin, font, font_scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
