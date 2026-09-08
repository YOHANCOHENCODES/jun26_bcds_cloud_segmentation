from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from segmentation_models_pytorch.encoders import get_preprocessing_params
from torch.utils.data import Dataset

from src.data.cloud_segmentation_image import CloudSegmentationImage

# Encoder the models are built with (see `smp.PSPNet(encoder_name=...)`) and
# the name of the pretrained weights whose normalization statistics apply.
DEFAULT_ENCODER_NAME = 'resnet34'
DEFAULT_ENCODER_WEIGHTS = 'imagenet'


def get_normalization(
    encoder_name: str = DEFAULT_ENCODER_NAME,
    encoder_weights: str = DEFAULT_ENCODER_WEIGHTS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the `(mean, std)` an image must be normalized with before it is
    fed into `encoder_name`, shaped `(C, 1, 1)` so they broadcast over a
    `(C, H, W)` image tensor.

    The values are looked up from `segmentation_models_pytorch` instead of
    being hardcoded, so they stay in sync with the encoder: a different
    `encoder_name` or `encoder_weights` brings its own statistics along.

    NOTE: `encoder_weights` names the *pretrained setting* the statistics come
    from, not the weights actually loaded. When a trained checkpoint is loaded
    (`encoder_weights=None` in `smp.PSPNet`), the images still have to be
    normalized the way they were during training.
    """
    params = get_preprocessing_params(encoder_name, pretrained=encoder_weights)
    mean = torch.tensor(params['mean']).view(-1, 1, 1)
    std = torch.tensor(params['std']).view(-1, 1, 1)
    return mean, std


class CloudSegmentationDataset(Dataset):
    """PyTorch `Dataset` over the Cloud Segmentation images.

    For every unique image referenced in `labels_df`, this builds a
    `CloudSegmentationImage`, optionally runs it through an albumentations
    `pipeline` (applied identically to the image and its per-label mask
    stack) and returns an `(image, mask)` tensor pair ready for training.

    `encoder_name`/`encoder_weights` must match the ones the model is built
    with, because they decide the mean/std the images are normalized with
    (see `get_normalization`).
    """

    def __init__(
        self,
        labels_df: pd.DataFrame,
        image_dir: Path,
        pipeline: A.Compose | None = None,
        image_shape: tuple[int, int] = (384, 576),
        normalize: bool = True,
        encoder_name: str = DEFAULT_ENCODER_NAME,
        encoder_weights: str = DEFAULT_ENCODER_WEIGHTS,
    ):
        self.labels_df = labels_df
        self.image_dir = image_dir
        self.pipeline = pipeline
        self.image_shape = image_shape
        self.normalize = normalize

        # Looked up once here instead of per `__getitem__` call: the encoder
        # cannot change over the lifetime of the dataset.
        self.mean, self.std = get_normalization(encoder_name, encoder_weights)

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
            image_tensor = (image_tensor - self.mean) / self.std

        mask_tensor = torch.from_numpy(masks.astype(np.float32))

        return image_tensor, mask_tensor
