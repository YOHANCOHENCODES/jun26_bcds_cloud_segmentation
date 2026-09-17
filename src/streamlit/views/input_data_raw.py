from loaders import load_labels_raw_head, load_labels_slim

import streamlit as st
from src.data.dataset_facts import NUM_TEST_IMAGES

# The complete labels table - these numbers describe the whole dataset, not
# the few files that are deployed with the app
labels_df = load_labels_slim()

num_train_images = labels_df['Image'].nunique()
num_rows = len(labels_df)
num_unlabeled_rows = int((~labels_df['HasMask']).sum())

test_percent = round(
    100 * NUM_TEST_IMAGES / (NUM_TEST_IMAGES + num_train_images)
)

# The raw rows, exactly as they come out of `train.csv`
raw_head_df = load_labels_raw_head()


st.write('### Basic overview of the images')
st.markdown(f"""
- There are **{num_train_images} training images** and
  **{NUM_TEST_IMAGES} test images**
- This makes a test size of **{test_percent}%**
- Only the training images are labeled - the test images come without any
  annotation, which is why the model predictions later on cannot be compared
  against a ground truth
""")

st.write('### The raw labels dataframe')
st.markdown(f"""
- `train.csv` has **{num_rows} rows** and **{raw_head_df.shape[1]} columns**:
  `Image_Label` and `EncodedPixels` - both of type `object`
- There are **4 rows per image**, one for each of the categories
  **Fish**, **Flower**, **Gravel** and **Sugar**
- The category is *appended to the file name* in the column `Image_Label`.
  So this column holds *two* informations that should be separated
- `EncodedPixels` is `NaN` when an image is not labeled for that category
  ({num_unlabeled_rows} of {num_rows} rows) - otherwise it
  contains the **RLE encoded** mask that the scientists marked in the image
""")

st.dataframe(raw_head_df)
