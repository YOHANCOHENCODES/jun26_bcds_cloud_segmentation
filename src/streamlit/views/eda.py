"""Chapter 2 - Exploratory Data Analysis and Data Preparation.

Streamlit page of the project presentation. All text lives in the constants and
section functions below, all figures are read from `FIGURES_DIR`
(`reports/figures/`), so nothing else in the app has to change when the chapter
text is edited.

Only Figure 5 is still read from a file. Everything else is computed live from
`data/presentation/`: the label rows as real dataframes (the way
`input_data_raw.py` and `input_data_cleaned.py` do it), Figures 3 and 4 as
charts counted from the complete labels table, and Figure 6 from the masks of a
chosen training image.

Figure files used by this page:
    streamlit_eda_sensor_gap.png
"""

import altair as alt
import pandas as pd
from typing import Optional

from config import FIGURES_DIR, TRAIN_IMAGES_DIR
from loaders import (
    first_row_id,
    load_labels_head_processed,
    load_labels_raw_head,
    load_labels_slim,
    load_train_image_names,
    load_train_labels,
)

import streamlit as st
from src.data.cloud_segmentation_image import CloudSegmentationImage
from src.visualization.image_display import draw_segmap_on_image

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

CLOUD_CLASSES = ['Fish', 'Flower', 'Gravel', 'Sugar']

FACTS = {
    'n_images': 5546,
    'n_records': 22184,
    'n_with_mask': 11836,
    'n_without_mask': 10348,
    'pct_with_mask': 53.35,
    'original_size': '1400 x 2100',
    'processed_size': '576 x 384',
    'image_shape': '384 x 576 x 3',
    'mask_shape': '384 x 576 x 4',
    'n_train': 4991,
    'n_val': 555,
    'batch_size': 8,
    'random_state': 42,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

AXIS_LABEL_SIZE = 17
AXIS_TITLE_SIZE = 19

BAR_BAND_SIZE = 56
AXIS_BAND_SIZE = 60

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _figure(filename: str, caption: str) -> None:
    """Show a report figure with its caption, or a clear placeholder if
    missing."""
    path = FIGURES_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(
            f'Figure not found: `{path.name}` - expected in `{FIGURES_DIR}`.'
        )


def _section(title: str, number: str = '') -> None:
    label = f'{number} {title}'.strip()
    st.markdown(f'### {label}')


def _takeaway(text: str) -> None:
    """Highlighted one-line conclusion, used at the end of each section."""
    st.info(f'**Takeaway** - {text}')


def _mask_counts() -> pd.DataFrame:
    """Available masks per cloud type - Figure 3 of the report, counted from
    the complete labels table instead of read from a picture."""
    labels_df = load_labels_slim()

    counts = (
        labels_df[labels_df['HasMask']]
        .groupby('Label')
        .size()
        .reindex(CLOUD_CLASSES)
    )

    return pd.DataFrame(
        {
            'Cloud type': counts.index,
            'Available masks': counts.to_numpy(),
            'Share': 100 * counts.to_numpy() / counts.sum(),
        }
    )


def _bar_chart(
    df: pd.DataFrame,
    *,
    category: str,
    value: str,
    category_title: Optional[str] = None,
    value_title: Optional[str] = None,
    horizontal: bool = False,
) -> None:
    """One-series bar chart with enlarged labels."""

    category_axis = alt.Y if horizontal else alt.X
    value_axis = alt.X if horizontal else alt.Y

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            category_axis(
                f'{category}:O',
                title=category_title,
                sort=None,
                axis=alt.Axis(
                    labelOverlap=False,
                    labelAngle=0,
                ),
            ),
            value_axis(
                f'{value}:Q',
                title=value_title,
            ),
            tooltip=[category, value],
        )
        .properties(
            height=len(df) * BAR_BAND_SIZE + AXIS_BAND_SIZE
        )
        .configure_axis(
            labelFontSize=AXIS_LABEL_SIZE,
            titleFontSize=AXIS_TITLE_SIZE,
        )
    )

    # Compatible with the Streamlit version used in the Python 3.9 environment.
    st.altair_chart(
        chart,
        use_container_width=True,
    )


def _cloud_types_per_image() -> pd.DataFrame:
    """How many cloud types are annotated per image - Figure 4 of the report,
    counted from the complete labels table."""
    labels_df = load_labels_slim()

    per_image = labels_df.groupby('Image')['HasMask'].sum()
    counts = per_image.value_counts().sort_index()

    return pd.DataFrame(
        {
            'Cloud types': counts.index.astype(int),
            'Images': counts.to_numpy(),
        }
    )


# --------------------------------------------------------------------------- #
# 2.1 Exploratory Data Analysis
# --------------------------------------------------------------------------- #


def _kpi_row() -> None:
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        'Satellite images',
        f'{FACTS["n_images"]:,}',
    )

    c2.metric(
        'Annotation records',
        f'{FACTS["n_records"]:,}',
    )

    c3.metric(
        'Records with a mask',
        f'{FACTS["pct_with_mask"]:.2f} %',
        help=(
            f'{FACTS["n_with_mask"]:,} of {FACTS["n_records"]:,} records '
            'carry an RLE mask.'
        ),
    )

    c4.metric(
        'Cloud categories',
        len(CLOUD_CLASSES),
        help=', '.join(CLOUD_CLASSES),
    )


def _s211_overview() -> None:
    _section('Data Set Overview', '2.1.1')

    st.markdown(f"""
The project uses the Kaggle **Understanding Clouds from Satellite Images** dataset: satellite images
with cloud segmentation annotations for four cloud organisation patterns - **Fish, Flower, Gravel**
and **Sugar**.

- **{FACTS['n_images']:,} unique satellite images** and **{FACTS['n_records']:,} annotation records**,
  {FACTS['n_images']:,} records per cloud category.
- Segmentation annotations are stored in **Run-Length Encoding (RLE)** in the `EncodedPixels` column
  of `train.csv`.
- Original resolution **{FACTS['original_size']} pixels**, resized for this project to
  **{FACTS['processed_size']} pixels** (NumPy shape `{FACTS['image_shape']}`).
- Only **{FACTS['pct_with_mask']:.2f} %** of the records contain a valid RLE mask. A missing value means
  that this cloud category was not annotated in that image - it is a negative label, not missing data.
""")

    with st.expander('Dataset files'):
        st.markdown("""
| File | Content | Used |
|---|---|---|
| `train.csv` | RLE masks for every image / cloud-label pair | yes |
| `train_images.zip` | training satellite images | yes |
| `test_images.zip` | test satellite images | yes |
| `sample_submission.csv` | example of the required output format | no |
""")

    with st.expander('Where the images and the labels come from'):
        st.markdown("""
- The images come from **NASA Worldview** and were captured by the **TERRA** and **AQUA**
  polar-orbiting satellites with the **MODIS** instrument.
- An image can be composed from two successive orbits. The uncovered area between them appears as a
  **black band** - a sensor gap, not a valid image region, and therefore not a valid prediction area.
- The ground truth was created by **crowdsourcing with 68 scientists**. Each image was labelled by
  about **three** of them, and the final mask is the **union** of their annotations after removing the
  black band.
- Because several patterns may overlap in the same pixel, the task is **multi-label segmentation**,
  not multi-class segmentation.
""")

    _takeaway(
        'A moderate-sized, pixel-level annotated dataset with coarse, crowdsourced labels - which sets '
        'both the modelling approach (multi-label) and the realistic accuracy ceiling.'
    )


def _s212_structure() -> None:
    _section('Dataset Structure', '2.1.2')

    st.markdown("""
The training annotations arrive in a CSV with two columns: `Image_Label` and `EncodedPixels`.
""")

    st.dataframe(
        load_labels_raw_head(),
        use_container_width=True,
    )

    st.caption(
        'Figure 1: Initial dataframe of the training dataset'
    )

    st.markdown("""
`Image_Label` mixes two pieces of information, so it was split into `Image` and `Label` -
for example `0011165.jpg_Fish` becomes `0011165.jpg` and `Fish`.
""")

    st.dataframe(
        load_labels_head_processed(),
        use_container_width=True,
    )

    st.caption(
        'Figure 2: Dataset structure after separating image identifiers and cloud categories'
    )

    st.markdown("""
Each image appears up to four times, once per cloud category. The RLE strings were later decoded into
multi-channel masks, so that every image is associated with one pixel-level target containing the
segmentation of all four categories.
""")

    _takeaway(
        'One row per image-and-category pair becomes one image with a four-channel target.'
    )


def _s213_class_distribution() -> None:
    _section('Class Distribution', '2.1.3')

    df = _mask_counts()
    masks = df['Available masks']

    left, right = st.columns(
        [3, 2],
        gap='large',
    )

    with left:
        _bar_chart(
            df,
            category='Cloud type',
            value='Available masks',
            category_title=None,
            value_title='Available masks',
            horizontal=True,
        )

        st.caption(
            'Figure 3: Distribution of available segmentation masks by cloud type'
        )

    with right:
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={
                'Available masks': st.column_config.NumberColumn(
                    format='%d'
                ),
                'Share': st.column_config.ProgressColumn(
                    format='%.1f %%',
                    min_value=0.0,
                    max_value=35.0,
                ),
            },
        )
        st.caption(
            f'Largest / smallest ratio: {masks.max() / masks.min():.2f}'
        )
    st.markdown("""
**Sugar** has the most annotated masks, **Flower** the fewest. The imbalance is **moderate**
(a ratio of about 1.6, not 1:10), so no class-balancing technique was applied - neither a
class-weighted loss nor resampling. In addition, the Dice component of the loss used in the later
models is normalised by region size and is therefore already less sensitive to class frequency than
plain cross-entropy.
""")

    with st.expander('Was that the right decision? (defense note)'):
        st.markdown("""
The results support it. If frequency drove performance, **Flower** would be the weakest class -
it is in fact the **strongest** in every model, while **Fish** is the weakest despite having more
masks than Flower. The limiting factor is therefore class **difficulty** and label ambiguity,
not class frequency, so frequency-based weighting would have addressed the wrong problem.
""")

    _takeaway(
        'Moderate imbalance, deliberately left untreated - and the per-class results justify it.'
    )


def _s214_mask_availability() -> None:
    _section(
        'Mask Availability and Cloud Categories per Image',
        '2.1.4',
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        f'Records with a mask ({FACTS["pct_with_mask"]:.2f} %)',
        f'{FACTS["n_with_mask"]:,}',
    )

    c2.metric(
        f'Records without a mask ({100 - FACTS["pct_with_mask"]:.2f} %)',
        f'{FACTS["n_without_mask"]:,}',
    )

    c3.metric(
        'Cloud types per image',
        '1 to 4',
        help='Most images contain two cloud types.',
    )

    st.markdown(f"""
Of the {FACTS['n_records']:,} annotation records, **{FACTS['n_with_mask']:,} carry a segmentation mask**
and **{FACTS['n_without_mask']:,} do not**. A missing value does **not** mean that the image is
cloud-free - it means that this one cloud category was not annotated there.
""")

    per_image_df = _cloud_types_per_image()

    left, right = st.columns(
        [3, 2],
        gap='large',
    )

    with left:
        _bar_chart(
            per_image_df,
            category='Cloud types',
            value='Images',
            category_title='Cloud types in the image',
            value_title='Images',
        )

        st.caption(
            'Figure 4: Number of cloud types per image'
        )

    with right:
        st.dataframe(
            per_image_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                'Cloud types': st.column_config.NumberColumn(
                    format='%d'
                ),
                'Images': st.column_config.NumberColumn(
                    format='%d'
                ),
            },
        )

    st.markdown("""
Images contain between **one and four** cloud types. Most contain **two**, followed by three and one,
and only a few contain all four. Most scenes therefore hold several patterns at once, which is why the
task is framed as multi-label segmentation and why the model has to distinguish categories **within**
one image.
""")

    _takeaway(
        'Multi-label by nature: the model must separate several co-occurring patterns per scene.'
    )


def _s215_sensor_gap() -> None:
    _section('Sensor Gap Analysis', '2.1.5')

    st.markdown("""
Some images contain a **black band** caused by the uncovered area between two successive satellite
orbits. It is a sensor gap, not a cloud region.
""")

    _figure(
        'streamlit_eda_sensor_gap.png',
        'Figure 5: Example of sensor gap detection',
    )

    st.markdown("""
**What the figure shows**

- **Left** - the original satellite image.
- **Middle** - the detected sensor gap (white = gap).
- **Right** - the valid image region with its boundary.

The black band carries no valid satellite information and must not be interpreted as a cloud region.
The ground truth annotations already exclude these areas.

**Decision:** no additional sensor-gap mask was applied during training or prediction. A
brightness-based heuristic would also catch genuinely dark ocean and would silently remove valid
prediction area, so the modelling stays aligned with the provided ground truth.
""")

    _takeaway(
        'The gap was understood and deliberately not corrected - the labels already handle it.'
    )


# --------------------------------------------------------------------------- #
# 2.2 Data Preparation
# --------------------------------------------------------------------------- #

RLE_SNIPPET = '''def rle_to_mask(rle: str, height: int = 1400, width: int = 2100) -> np.ndarray:
    """Decode one RLE string into a binary mask of shape (height, width).

    A missing / NaN value is treated as an empty mask: the cloud category was
    not annotated in this image.
    """
    mask = np.zeros(height * width, dtype=np.uint8)

    if isinstance(rle, str) and rle.strip():
        values = [int(v) for v in rle.split()]
        starts, lengths = values[0::2], values[1::2]

        for start, length in zip(starts, lengths):
            mask[start - 1:start - 1 + length] = 1

    return mask.reshape(height, width, order="F")
'''

MASK_STACK_SNIPPET = '''CLOUD_CLASSES = ["Fish", "Flower", "Gravel", "Sugar"]

def build_mask_stack(labels_df, image, height, width):
    """Return the four-channel ground truth of one image: (H, W, 4)."""
    channels = []

    for cloud_type in CLOUD_CLASSES:
        row = labels_df[
            (labels_df.Image == image) &
            (labels_df.Label == cloud_type)
        ]

        rle = row.EncodedPixels.values[0] if len(row) else None
        channels.append(rle_to_mask(rle, height, width))

    return np.stack(channels, axis=-1)
'''

RESIZE_SNIPPET = """IMG_H, IMG_W = 384, 576

# Image: bilinear interpolation is fine for continuous pixel values
image = cv2.resize(
    image,
    (IMG_W, IMG_H),
    interpolation=cv2.INTER_LINEAR,
)

# Masks: NEAREST only - any interpolation would create invalid values
# between 0 and 1
mask_stack = cv2.resize(
    mask_stack,
    (IMG_W, IMG_H),
    interpolation=cv2.INTER_NEAREST,
)
"""

NORMALIZE_SNIPPET = """IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

image = image.astype("float32") / 255.0
image = (image - IMAGENET_MEAN) / IMAGENET_STD

# The masks are NOT normalised:
# they stay binary (1 = cloud category present).
"""

LOADER_SNIPPET = """from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

# Batch shapes handed to the model:
# images -> [8, 3, 384, 576]
# masks  -> [8, 4, 384, 576]
"""


def _s221_rle() -> None:
    _section('RLE Decoding', '2.2.1')

    st.markdown("""
The masks are stored as **Run-Length Encoding**: pairs of a start position and a run length. To use
them as segmentation targets they were decoded into **binary masks** with the same spatial size as the
processed images - `1` for pixels of that cloud category, `0` elsewhere. **Missing `EncodedPixels`
values become empty masks**, because they indicate that no mask is available for that image and
category.
""")

    with st.expander('What does RLE encoded mean?'):
        st.markdown("""
> In order to reduce the submission file size, our metric uses *run-length
> encoding* on the pixel values. Instead of submitting an exhaustive list of
> indices for your segmentation, you will submit pairs of values that contain a
> start position and a run length. E.g. `1 3` implies starting at pixel 1 and
> running a total of 3 pixels (1,2,3). The pixels are numbered from top to
> bottom, then left to right.

*Source:*
[Kaggle - Understanding Clouds from Satellite Images](https://www.kaggle.com/competitions/understanding_cloud_organization/overview/evaluation)
""")

    st.code(
        RLE_SNIPPET,
        language='python',
    )

    st.caption(
        'Two details that are easy to get wrong: RLE start positions are 1-based, and the mask '
        'is filled in column-major (Fortran) order.'
    )

    _takeaway(
        'RLE in, binary mask out - with missing values interpreted as a negative label.'
    )


def _s222_multichannel() -> None:
    _section(
        'Multi-channel Mask Creation',
        '2.2.2',
    )

    st.markdown(f"""
The four decoded binary masks of an image are stacked into **one four-channel mask**, one channel per
cloud category in the fixed order **{', '.join(CLOUD_CLASSES)}**. Every processed image therefore has
a ground truth of shape **`{FACTS['mask_shape']}`**.

Because the channels are independent, a pixel can belong to several categories at once - which is
exactly what the crowdsourced union labels contain. This is the reason the models end in **four sigmoid
channels** rather than a four-class softmax.
""")

    st.code(
        MASK_STACK_SNIPPET,
        language='python',
    )

    _takeaway(
        'Four independent channels - the data representation that makes the task multi-label.'
    )


def _s223_visualization() -> None:
    _section(
        'Image and Ground Truth Mask Visualization',
        '2.2.3',
    )

    st.markdown("""
Before training, the decoded annotations were visually checked against the satellite image to confirm
their spatial alignment. Pick an image to run that check here - the masks are decoded from the RLE
strings on the spot, not read from a stored picture.
""")

    train_labels_df = load_train_labels()

    image_name = st.selectbox(
        'Training image',
        load_train_image_names(),
    )

    csi = CloudSegmentationImage(
        train_labels_df,
        first_row_id(train_labels_df, image_name),
        TRAIN_IMAGES_DIR,
    )

    loaded_image_rgb = csi.get_image_rgb()

    if loaded_image_rgb is None:
        st.error(
            f'The image {csi.get_image_path()} could not be read'
        )
        return

    segmap_image_rgb = draw_segmap_on_image(csi)

    if segmap_image_rgb is None:
        st.error(
            f'The segmentation map of {csi.get_name()} could not be drawn'
        )
        return

    container_left, container_right = st.columns(2)

    container_left.image(
        loaded_image_rgb,
        caption='Original',
    )

    container_right.image(
        segmap_image_rgb,
        caption='With Masks Overlay',
    )

    st.caption(
        'Figure 6 of the report, as a live example: a satellite image with its ground truth '
        'segmentation masks.'
    )

    st.markdown("""
The masks show where the annotated cloud regions lie and that they line up with the image. A black
band in an image is the **sensor gap**; the ground truth already excludes it.
""")

    _takeaway(
        'A visual check is the cheapest way to catch an RLE decoding bug before training.'
    )


def _s224_resizing() -> None:
    _section(
        'Image Resizing',
        '2.2.4',
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        'Original resolution',
        FACTS['original_size'],
    )

    c2.metric(
        'Processed resolution',
        FACTS['processed_size'],
    )

    c3.metric(
        'Model input shape',
        FACTS['image_shape'],
    )

    st.markdown(f"""
The images were resized from **{FACTS['original_size']}** to **{FACTS['processed_size']}** pixels to
reduce the computational cost, and the **masks were resized identically** so that image and label stay
aligned. After resizing: image `{FACTS['image_shape']}`, mask `{FACTS['mask_shape']}`.

Two reasons make this a safe trade-off rather than a loss of information. The target classes are
**large-scale organisation patterns**, not fine texture, and the crowdsourced ground truth is coarse
in the first place. In addition, 384 and 576 are both **divisible by 32**, which the five-stage ResNet
encoders require in order to reconstruct the exact input size.
""")

    st.code(
        RESIZE_SNIPPET,
        language='python',
    )

    _takeaway(
        'A 3.6x downscale that keeps the class-relevant signal and fits a single GPU.'
    )


def _s225_normalization() -> None:
    _section(
        'Normalization',
        '2.2.5',
    )

    c1, c2 = st.columns(2)

    c1.metric(
        'ImageNet mean (R, G, B)',
        ', '.join(f'{v:.3f}' for v in IMAGENET_MEAN),
    )

    c2.metric(
        'ImageNet std (R, G, B)',
        ', '.join(f'{v:.3f}' for v in IMAGENET_STD),
    )

    st.markdown("""
The images were normalised with the **ImageNet statistics**, because the models use
**ImageNet-pretrained encoders**: this gives the network an input distribution that matches the
pretrained weights and makes training more stable. Normalising with our own dataset statistics instead
would have reduced the value of those weights.

The **masks are not normalised** - they stay binary, with `1` for the presence of a cloud category.
""")

    st.code(
        NORMALIZE_SNIPPET,
        language='python',
    )

    _takeaway(
        'Normalisation follows the encoder, not the dataset.'
    )


# --------------------------------------------------------------------------- #
# 2.3 Data Preparation Results
# --------------------------------------------------------------------------- #


def _s232_split() -> None:
    _section(
        'Train / Validation Split',
        '2.3.2',
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        'Training images (90 %)',
        f'{FACTS["n_train"]:,}',
    )

    c2.metric(
        'Validation images (10 %)',
        f'{FACTS["n_val"]:,}',
    )

    c3.metric(
        'Random state',
        FACTS['random_state'],
    )

    st.markdown(f"""
The dataset was split **90 / 10** into {FACTS['n_train']:,} training and {FACTS['n_val']:,} validation
images ({FACTS['n_images']:,} in total), with a fixed random state of **{FACTS['random_state']}** for
reproducibility.

The decisive detail is that the split is performed **at image level, on unique image identifiers** -
so all four label rows of an image stay on the same side and **no image can leak** between training and
validation. The same split is used for every model, which is what makes the comparison in Chapter 6
possible.

The split is a **plain random** split; a stratified split over the label combinations is a planned
optimisation step and deliberately not part of this baseline.
""")

    _takeaway(
        'One fixed, leakage-free split shared by all models - the basis of every later comparison.'
    )


def _s233_loaders() -> None:
    _section(
        'Data Loaders, Batch Size and Data Format',
        '2.3.3 - 2.3.5',
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        'Batch size',
        FACTS['batch_size'],
    )

    c2.metric(
        'Image batch',
        '[8, 3, 384, 576]',
    )

    c3.metric(
        'Mask batch',
        '[8, 4, 384, 576]',
    )

    st.markdown(f"""
Both datasets are loaded with a PyTorch **`DataLoader`** and a batch size of **{FACTS['batch_size']}**,
which balances computational efficiency against GPU memory at this input resolution.

- The **training** loader **shuffles**, so the image order differs in every epoch.
- The **validation** loader does **not** shuffle - the order is irrelevant, and a deterministic order
  keeps the epoch-to-epoch score comparable.
- Both use `num_workers=0` and `pin_memory=True` for efficient transfer to the GPU.

After preprocessing, a single image has shape `(3, 384, 576)` and its mask `(4, 384, 576)`: three RGB
channels in, four cloud-class channels out.
""")

    st.code(
        LOADER_SNIPPET,
        language='python',
    )

    _takeaway(
        'Input pipeline complete: batches of 8 images and their four-channel targets.'
    )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

SECTIONS = {
    '2.1 Exploratory Data Analysis': [
        _s211_overview,
        _s212_structure,
        _s213_class_distribution,
        _s214_mask_availability,
        _s215_sensor_gap,
    ],
    '2.2 Data Preparation': [
        _s221_rle,
        _s222_multichannel,
        _s223_visualization,
        _s224_resizing,
        _s225_normalization,
    ],
    '2.3 Data Preparation Results': [
        _s232_split,
        _s233_loaders,
    ],
}


def render(navigation: str = 'tabs') -> None:
    """Render the whole chapter.

    Parameters
    ----------
    navigation : {'tabs', 'flow'}
        `'tabs'` puts 2.1 / 2.2 / 2.3 on three tabs (good for a live talk),
        `'flow'` renders everything on one scrolling page (good for reading).
    """
    st.title('Exploratory Data Analysis and Data Preparation')
    st.markdown(
        '**From 22,184 RLE strings to batches of `[8, 3, 384, 576]` images with '
        '`[8, 4, 384, 576]` targets.** This chapter covers what the data looks like, what had to be '
        'repaired or reshaped, and which preprocessing decisions the later model results depend on.'
    )

    _kpi_row()
    st.divider()

    if navigation == 'tabs':
        for tab, (_name, sections) in zip(
            st.tabs(list(SECTIONS)),
            SECTIONS.items(),
        ):
            with tab:
                for i, section in enumerate(sections):
                    section()

                    if i < len(sections) - 1:
                        st.divider()
    else:
        for name, sections in SECTIONS.items():
            st.header(name)

            for section in sections:
                section()
                st.divider()


# This module is a page script of `app.py` (`st.navigation`), so it renders on
# import. `st.set_page_config()` stays in `app.py` - calling it here a second
# time would raise.

render()