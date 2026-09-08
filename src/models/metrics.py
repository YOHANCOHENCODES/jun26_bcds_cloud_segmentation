"""Dice and IoU metrics for the multilabel cloud masks.

The training loss is `BCEWithLogitsLoss + DiceLoss`, which is what the
optimizer minimizes - it is *not* a score that can be reported as "Dice".
This module computes the actual metrics on top of the same 0/1 masks the
model produces at inference time.
"""

from collections.abc import Iterable

import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from src.data.cloud_segmentation_image import LABEL_CLASS_IDS

# The threshold `predict_mask_stack` turns probabilities into a 0/1 mask with.
# Metrics are computed with the same value, so the numbers describe the masks
# the model actually produces.
DEFAULT_THRESHOLD = 0.5

# Score for a class whose tp/fp/fn are all 0, i.e. the class is absent from the
# ground truth and nothing was predicted for it. Both Dice and IoU are 0/0
# there; `segmentation_models_pytorch` counts that as a perfect prediction and
# this follows that convention.
ZERO_DIVISION_SCORE = 1.0


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Element-wise `numerator / denominator`, substituting
    `ZERO_DIVISION_SCORE` wherever the denominator is 0."""
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp(min=1.0),
        torch.full_like(numerator, ZERO_DIVISION_SCORE),
    )


class SegmentationStats:
    """Accumulates the confusion-matrix counts of a whole epoch (or a whole
    `DataLoader`) and turns them into Dice and IoU scores.

    Counts are collected per batch via `update`, which is cheap enough to run
    inside the training loop on the logits that are computed anyway - no extra
    forward pass is needed.

    Two aggregations are reported, because they answer different questions on
    this dataset (many masks are empty, and the cloud types differ a lot in
    size):

    * **micro** sums the pixel counts over all images *and* all 4 cloud types
      before dividing once. Large classes dominate it, so it reads as "how
      many of all cloud pixels did we get right".
    * **macro** computes one score per cloud type (over all images) and
      averages those 4 scores. Every cloud type counts the same, regardless of
      how many pixels it covers.

    Both aggregate over the whole dataset rather than per image, which avoids
    the 0/0 case that images without a given cloud type would otherwise
    produce on nearly every batch.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        label_names: Iterable[str] | None = None,
    ):
        self.threshold = threshold
        self.label_names = list(
            label_names if label_names is not None else LABEL_CLASS_IDS
        )
        # One (batch_size, num_classes) tensor per processed batch
        self._tp: list[torch.Tensor] = []
        self._fp: list[torch.Tensor] = []
        self._fn: list[torch.Tensor] = []

    def update(self, logits: torch.Tensor, masks: torch.Tensor) -> None:
        """Add one batch of raw model outputs (logits, *not* probabilities)
        and their ground-truth masks, both shaped `(N, num_classes, H, W)`."""
        # `get_stats` does not apply an activation itself, so the logits are
        # squashed here the same way `predict_mask_stack` does it.
        probs = torch.sigmoid(logits.detach())

        tp, fp, fn, _ = smp.metrics.get_stats(
            probs,
            masks.detach().long(),
            mode='multilabel',
            threshold=self.threshold,
        )

        # Moved off the GPU right away: the counts are tiny, and keeping them
        # on the device would pin memory for the rest of the epoch.
        self._tp.append(tp.cpu())
        self._fp.append(fp.cpu())
        self._fn.append(fn.cpu())

    def class_counts(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the `(tp, fp, fn)` pixel counts summed over all images, one
        entry per cloud type (shape `(num_classes,)`)."""
        if not self._tp:
            raise ValueError('No batches accumulated - call update() first.')

        return (
            torch.cat(self._tp).sum(dim=0).float(),
            torch.cat(self._fp).sum(dim=0).float(),
            torch.cat(self._fn).sum(dim=0).float(),
        )

    def dice_per_class(self) -> torch.Tensor:
        """Dice score per cloud type: `2*tp / (2*tp + fp + fn)`."""
        tp, fp, fn = self.class_counts()
        return _ratio(2 * tp, 2 * tp + fp + fn)

    def iou_per_class(self) -> torch.Tensor:
        """IoU (Jaccard) per cloud type: `tp / (tp + fp + fn)`."""
        tp, fp, fn = self.class_counts()
        return _ratio(tp, tp + fp + fn)

    def scores(self) -> dict[str, float]:
        """Return the micro- and macro-aggregated Dice and IoU as a flat dict
        of plain floats, ready to be appended to a history."""
        tp, fp, fn = self.class_counts()

        return {
            'dice_micro': _ratio(
                2 * tp.sum(), 2 * tp.sum() + fp.sum() + fn.sum()
            ).item(),
            'dice_macro': self.dice_per_class().mean().item(),
            'iou_micro': _ratio(
                tp.sum(), tp.sum() + fp.sum() + fn.sum()
            ).item(),
            'iou_macro': self.iou_per_class().mean().item(),
        }

    def per_class_dataframe(self) -> pd.DataFrame:
        """Return Dice and IoU per cloud type as a `DataFrame`, indexed by
        label name - the per-class breakdown for the report."""
        return pd.DataFrame(
            {
                'dice': self.dice_per_class().tolist(),
                'iou': self.iou_per_class().tolist(),
            },
            index=pd.Index(self.label_names, name='label'),
        )


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[SegmentationStats, float | None]:
    """Run `model` over all of `loader` and return the accumulated
    `SegmentationStats` together with the mean loss (or `None` if no
    `criterion` was given).

    Use this to score a checkpoint that was trained without metric tracking:
    it needs nothing but the model and the validation loader, so no retraining
    is required.
    """
    model.eval()
    stats = SegmentationStats(threshold=threshold)

    total_loss = 0.0
    num_images = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)

        stats.update(logits, masks)

        if criterion is not None:
            # Weighted by batch size, so a smaller last batch does not count
            # as much as a full one.
            total_loss += criterion(logits, masks).item() * images.size(0)
        num_images += images.size(0)

    mean_loss = total_loss / num_images if criterion is not None else None
    return stats, mean_loss
