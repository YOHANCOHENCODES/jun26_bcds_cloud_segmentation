"""Build the data the Streamlit app ships with, from the full dataset.

`data/raw/` is ~1.3 GB and not part of the repository, so the app cannot read
it after a deployment. It reads `data/presentation/` instead - always, also
locally, so that what you see while developing is what the deployed app shows.

This script fills that directory from `data/raw/`. Run it whenever the raw
data or the selection below changes:

    python -m src.data.build_presentation_data

What it writes, and why that is enough to present the *whole* dataset:

`labels_slim.csv.gz`
    Every one of the 22184 label rows, but without the `EncodedPixels`
    strings, which make up ~99% of the 44 MB of `train.csv`. What the strings
    are needed for in an overview - whether a label is present and how much of
    the image it covers - is precomputed into `HasMask` and `MaskArea`. So
    counts, ratios and distributions still get computed live over the complete
    dataset; only the mask *geometry* is gone. ~120 KB.

`labels_raw_head.csv`
    The first rows of `train.csv` completely unchanged, RLE strings included,
    for the page that shows what the raw labels look like.

`train_images/` + `train_labels.csv`
    A few training images with their full RLE rows, so masks can actually be
    decoded and drawn. These are *examples* - the numbers on the pages come
    from `labels_slim.csv.gz`, never from this selection.
"""

from pathlib import Path
from shutil import copyfile

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
RAW_LABELS_PATH = RAW_DIR / 'train.csv'
RAW_TRAIN_IMAGES_DIR = RAW_DIR / 'train_images'

PRESENTATION_DIR = PROJECT_ROOT / 'data' / 'presentation'

# Training images that are shipped as examples. Chosen so that the mask
# overlays actually show something: every one of them carries several labels
# whose masks are large enough to be recognizable, and together they cover the
# range from two labels up to all four.
EXAMPLE_IMAGE_NAMES = [
    '002be4f.jpg',  # Fish, Flower, Sugar
    'f9d19e3.jpg',  # all four labels, none of them small
    'b24b4f6.jpg',  # all four labels
    'dd41467.jpg',  # Gravel, Sugar - two large, clearly separated masks
]

# Number of raw label rows kept for the "this is what train.csv looks like"
# page. Two images' worth, to show that every image brings one row per label.
NUM_RAW_HEAD_ROWS = 8


def mask_area(rle_string: object) -> int:
    """Number of pixels the RLE encoded mask `rle_string` covers.

    The RLE holds `(start, length)` pairs, so the covered area is the sum of
    the lengths - the mask itself never has to be decoded for this. A missing
    label (`NaN` instead of a string) covers nothing.
    """
    if not isinstance(rle_string, str):
        return 0
    return sum(int(length) for length in rle_string.split()[1::2])


def split_image_label(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the `Image_Label` column by the two columns `Image` and `Label`
    it packs together, the way `src.data.labels.load_processed_labels` does.
    """
    df = df.copy()
    df[['Image', 'Label']] = df['Image_Label'].str.split('_', n=1, expand=True)
    return df.drop(columns='Image_Label')


def build_slim_labels(raw_df: pd.DataFrame) -> pd.DataFrame:
    """The complete labels table without the RLE strings (see module doc)."""
    slim_df = split_image_label(raw_df)
    slim_df['MaskArea'] = raw_df['EncodedPixels'].map(mask_area)
    slim_df['HasMask'] = slim_df['MaskArea'] > 0
    return slim_df.drop(columns='EncodedPixels')


def build_example_labels(raw_df: pd.DataFrame) -> pd.DataFrame:
    """The full rows - RLE included - of the example images, in the order of
    `EXAMPLE_IMAGE_NAMES` so that the app's image selection follows it.
    """
    df = split_image_label(raw_df)
    example_df = df[df['Image'].isin(EXAMPLE_IMAGE_NAMES)].copy()

    # `isin` keeps the order of `train.csv`; sort by the position in
    # `EXAMPLE_IMAGE_NAMES` instead, and drop the row numbers of the full
    # table - `CloudSegmentationImage` addresses rows by position in *this*
    # table
    order = {name: i for i, name in enumerate(EXAMPLE_IMAGE_NAMES)}
    example_df = example_df.sort_values(
        'Image', key=lambda images: images.map(order), kind='stable'
    )
    return example_df.reset_index(drop=True)


def copy_example_images(target_dir: Path) -> None:
    """Copy the example images over, and remove images of a previous run that
    are no longer selected.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    for stale_path in target_dir.glob('*.jpg'):
        if stale_path.name not in EXAMPLE_IMAGE_NAMES:
            stale_path.unlink()
            print(f'  removed {stale_path.name}')

    for image_name in EXAMPLE_IMAGE_NAMES:
        source_path = RAW_TRAIN_IMAGES_DIR / image_name
        if not source_path.is_file():
            raise FileNotFoundError(f'Example image missing: {source_path}')
        copyfile(source_path, target_dir / image_name)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write `df` to `path`, gzipped when the name says so.

    `mtime=0` keeps the gzip bytes identical between runs on unchanged data -
    otherwise the embedded timestamp would show up as a diff in every build.
    """
    compression = (
        {'method': 'gzip', 'mtime': 0} if path.suffix == '.gz' else 'infer'
    )
    df.to_csv(path, index=False, compression=compression)
    print(f'  {path.relative_to(PROJECT_ROOT)}  ({path.stat().st_size:,} B)')


def main() -> None:
    if not RAW_LABELS_PATH.is_file():
        raise FileNotFoundError(
            f'{RAW_LABELS_PATH} not found. This script needs the full dataset '
            'in data/raw/; the Streamlit app does not.'
        )

    raw_df = pd.read_csv(RAW_LABELS_PATH)
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f'Reading {RAW_LABELS_PATH.relative_to(PROJECT_ROOT)} '
        f'({len(raw_df):,} rows)'
    )

    write_csv(
        build_slim_labels(raw_df), PRESENTATION_DIR / 'labels_slim.csv.gz'
    )
    write_csv(
        raw_df.head(NUM_RAW_HEAD_ROWS),
        PRESENTATION_DIR / 'labels_raw_head.csv',
    )
    write_csv(
        build_example_labels(raw_df), PRESENTATION_DIR / 'train_labels.csv'
    )

    copy_example_images(PRESENTATION_DIR / 'train_images')
    print(f'  {len(EXAMPLE_IMAGE_NAMES)} example images copied')


if __name__ == '__main__':
    main()
