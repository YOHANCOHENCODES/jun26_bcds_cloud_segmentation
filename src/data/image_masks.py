from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.rle import decode_rle


@dataclass
class ImageMask:
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
    ) -> "ImageMask":
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


class ImageMasks:
    def __init__(
            self,
            df: pd.DataFrame,
            image_id: int,
            image_path: Path,
            image_shape=(384, 576)
    ):
        self.df = df

        # Get the name of the 'requested-by-its-id' image
        self.image_name = self._get_image_name_from_id(image_id)

        # Get all id's (rows) for this name
        image_ids = self._get_image_ids(self.image_name)

        # Create an empty dictionary
        self.image_data: dict[str, ImageMask] = {}

        # Iterate the image_id's and create an `ImageData` for each of them
        for id in image_ids:
            image_data = ImageMask.from_dataframe(
                df, id, image_path, image_shape)
            self.image_data[image_data.label_name.lower()] = image_data

    def get_fish_mask(self) -> ImageMask:
        return self.image_data['fish']

    def get_flower_mask(self) -> ImageMask:
        return self.image_data['flower']

    def get_gravel_mask(self) -> ImageMask:
        return self.image_data['gravel']

    def get_sugar_mask(self) -> ImageMask:
        return self.image_data['sugar']

    def get_all_masks(self) -> list[ImageMask]:
        return list(self.image_data.values())

    def get_name(self) -> str:
        return self.image_name

    def _get_image_name_from_id(self, id: int) -> str:
        return str(self.df.at[id, 'Image'])

    def _get_image_ids(self, image_name: str) -> list[int]:
        return self.df.loc[self.df["Image"] == image_name].index.tolist()
