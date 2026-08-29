from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.cloud_segmentation_image import CloudSegmentationImage

# Mean/std the `resnet34` encoder was pretrained with on ImageNet. Images must
# be normalized with these before being fed into the model.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CloudSegmentationDataset(Dataset):
    """PyTorch `Dataset` over the Cloud Segmentation images.

    For every unique image referenced in `labels_df`, this builds a
    `CloudSegmentationImage`, optionally runs it through an albumentations
    `pipeline` (applied identically to the image and its per-label mask
    stack) and returns an `(image, mask)` tensor pair ready for training.
    """

    def __init__(
        self,
        labels_df: pd.DataFrame,
        image_dir: Path,
        pipeline: A.Compose | None = None,
        image_shape: tuple[int, int] = (384, 576),
        normalize: bool = True,
    ):
        self.labels_df = labels_df
        self.image_dir = image_dir
        self.pipeline = pipeline
        self.image_shape = image_shape
        self.normalize = normalize

        # One row id per unique image is enough: `CloudSegmentationImage`
        # looks up all of that image's rows (one per label) itself. Keeping
        # the first occurrence (instead of e.g. `groupby`) preserves the
        # image order of `labels_df`.
        first_rows = labels_df.drop_duplicates(subset='Image', keep='first')
        self.row_ids = first_rows.index.tolist()

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row_id = self.row_ids[idx]
        csi = CloudSegmentationImage(
            self.labels_df, row_id, self.image_dir, self.image_shape
        )

        image = csi.get_image_rgb()
        if image is None:
            raise FileNotFoundError(f'Image not found: {csi.get_image_path()}')

        # (num_classes, H, W), values 0/1 - kept as separate channels because
        # cloud types can overlap (see `CloudSegmentationImage.get_mask_stack`)
        masks = csi.get_mask_stack()

        if self.pipeline is not None:
            transformed = self.pipeline(image=image, masks=masks)
            image = transformed['image']
            masks = np.array(transformed['masks'])

        image_tensor = torch.from_numpy(np.ascontiguousarray(image))
        image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0
        if self.normalize:
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std

        mask_tensor = torch.from_numpy(masks.astype(np.float32))

        return image_tensor, mask_tensor
