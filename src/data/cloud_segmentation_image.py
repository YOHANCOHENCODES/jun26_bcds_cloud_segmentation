from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.utils.rle import decode_rle

# Class ids of the cloud labels in a combined segmentation map (0 is the
# background). The order also defines the priority when masks overlap: A label
# further down overwrites the ones above it.
LABEL_CLASS_IDS: dict[str, int] = {
    'fish': 1,
    'flower': 2,
    'gravel': 3,
    'sugar': 4,
}


@dataclass
class DecodedMask:
    """Data of a single image for one label."""

    mask_pixels: np.ndarray
    label_name: str

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        image_id: int,
        image_shape=(384, 576),
    ) -> 'DecodedMask':
        """Create an `ImageMask` instance from given `DataFrame`. The
        `DataFrame` must be suitable to our data: It must contain the columns
        `EncodedPixels` (with rle) and `Label` (cloud type as str).
        """
        encoded_mask_pixels = df.iloc[image_id]['EncodedPixels']
        label = df.iloc[image_id]['Label']
        decoded_mask_pixels = decode_rle(encoded_mask_pixels, image_shape)
        return cls(decoded_mask_pixels, label)


class CloudSegmentationImage:
    def __init__(
        self,
        labels_df: pd.DataFrame,
        requested_image_id_in_df: int,
        image_path: Path,
        image_shape=(384, 576),
    ):
        """Build a `CloudSegmentationImage` for the image referenced by
        `requested_image_id_in_df`, decoding its masks from `labels_df`.
        """
        self.df = labels_df
        self.image_path = image_path
        self.image_shape = image_shape

        self.image_rgb: np.ndarray | None = None

        # Get the name of the 'requested-by-its-id' image
        self.image_name = self._get_image_name_from_id(requested_image_id_in_df)

        # Get all id's (rows) for this name
        image_ids = self._get_image_ids(self.image_name)

        # Create an empty dictionary
        self.image_masks: dict[str, DecodedMask] = {}

        # Iterate the image_id's and create an `ImageMask` for each of them
        for id in image_ids:
            image_mask = DecodedMask.from_dataframe(labels_df, id, image_shape)
            self.image_masks[image_mask.label_name.lower()] = image_mask

    @classmethod
    def from_image_and_masks(
        cls,
        image_rgb: np.ndarray,
        mask_stack: np.ndarray,
        image_name: str,
        image_path: Path,
    ) -> 'CloudSegmentationImage':
        """Build a `CloudSegmentationImage` directly from an already-decoded
        image and a mask stack (e.g. produced by an augmentation pipeline),
        bypassing the usual DataFrame/RLE-based construction.
        """
        # Create the instance without calling __init__
        instance = cls.__new__(cls)
        instance.df = None
        instance.image_path = image_path
        instance.image_shape = image_rgb.shape[:2]
        instance.image_rgb = image_rgb
        instance.image_name = image_name
        instance.image_masks = {
            label_name: DecodedMask(
                mask_stack[channel], label_name.capitalize()
            )
            for channel, label_name in enumerate(LABEL_CLASS_IDS)
        }
        return instance

    def get_image_rgb(self):
        # If there in no imag RGB data yet, these are loaded first
        if self.image_rgb is None:
            image_full_path = str(self.image_path / self.image_name)
            image_bgr = cv2.imread(image_full_path)
            if image_bgr is None:
                return None
            # Convert the BGR data that has been loaded by cv2.imread() to RGB
            self.image_rgb = cv2.cvtColor(src=image_bgr, code=cv2.COLOR_BGR2RGB)
        return self.image_rgb

    def get_fish_mask(self) -> DecodedMask:
        return self.image_masks['fish']

    def get_flower_mask(self) -> DecodedMask:
        return self.image_masks['flower']

    def get_gravel_mask(self) -> DecodedMask:
        return self.image_masks['gravel']

    def get_sugar_mask(self) -> DecodedMask:
        return self.image_masks['sugar']

    def get_all_masks(self) -> list[DecodedMask]:
        return list(self.image_masks.values())

    def get_name(self) -> str:
        return self.image_name

    def get_image_path(self) -> Path:
        """Return the full path of the image all these masks belong to."""
        return self.image_path / self.image_name

    def get_segmap(self) -> np.ndarray:
        """Combine the masks of all labels into a single segmentation map, in
        which every pixel holds the class id of its label (see
        `LABEL_CLASS_IDS`) and 0 means background.

        NOTE: A pixel can hold only one class id. So where masks overlap, the
        label with the higher class id wins and the other ones are not
        visible in the result.
        """
        segmap = np.zeros(self.image_shape, dtype=np.int32)

        for label_name, class_id in LABEL_CLASS_IDS.items():
            image_mask = self.image_masks.get(label_name)

            # The `DataFrame` may not hold a row for every label of this image
            if image_mask is None:
                continue
            segmap[image_mask.mask_pixels > 0] = class_id
        return segmap

    def get_mask_stack(self) -> np.ndarray:
        """Stack the masks of all labels into a single array of the shape
        `(len(LABEL_CLASS_IDS), height, width)`, which holds 1 where a label
        is present and 0 where it is not.

        Channel `i` belongs to the label `list(LABEL_CLASS_IDS)[i]`, so the
        channel of a label is its class id minus 1.

        Unlike `get_segmap()` this keeps overlapping masks intact, because
        every label gets a channel of its own. It is the form a multi-label
        segmentation model needs.
        """
        height, width = self.image_shape
        stack = np.zeros((len(LABEL_CLASS_IDS), height, width), dtype=np.uint8)

        for channel, label_name in enumerate(LABEL_CLASS_IDS):
            image_mask = self.image_masks.get(label_name)

            # The `DataFrame` may not hold a row for every label of this image
            if image_mask is None:
                continue

            stack[channel] = image_mask.mask_pixels > 0

        return stack

    def _get_image_name_from_id(self, id: int) -> str:
        return str(self.df.at[id, 'Image'])

    def _get_image_ids(self, image_name: str) -> list[int]:
        return self.df.loc[self.df['Image'] == image_name].index.tolist()
