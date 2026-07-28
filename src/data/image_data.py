from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.rle import decode_rle


@dataclass
class ImageData:
    """Data of a single image for one label."""
    path: Path
    filename: str
    mask_pixels: np.ndarray
    label_name: str

    @classmethod
    def from_dataframe(
            cls,
            df: pd.DataFrame,
            image_id: int,
            image_path: Path,
            image_shape=(384, 576)
    ) -> "ImageData":
        """Create an `ImageData` instance from given `DataFrame`. The
        `DataFrame` must be suitable to our data: It must contain the columns 
        `Image` (image file name), `EncodedPixels` (with rle), `Label` 
        (name as str).
        """
        filename = df.iloc[image_id]['Image']
        encoded_mask_pixels = df.iloc[image_id]['EncodedPixels']
        label = df.iloc[image_id]['Label']
        decoded_mask_pixels = decode_rle(encoded_mask_pixels, image_shape)
        return cls(image_path, filename, decoded_mask_pixels, label)
