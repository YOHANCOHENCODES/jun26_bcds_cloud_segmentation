from config import TEST_IMAGES_DIR, TRAIN_IMAGES_DIR
from loaders import load_raw_labels

import streamlit as st

# Load the labels exactly as they come from `train.csv` - unprocessed
df = load_raw_labels()

# Count the images we got with the dataset
train_files_count = len(list(TRAIN_IMAGES_DIR.glob('*.jpg')))
test_files_count = len(list(TEST_IMAGES_DIR.glob('*.jpg')))
test_percent = 100 * test_files_count // (test_files_count + train_files_count)

# Pick the 8 rows that belong to the 2 first images - to show that every image
# comes with one row per cloud category
first_image_rows = df.head(8)


st.write('### Basic overview of the images')
st.markdown(f"""
- There are **{train_files_count} training images** and
  **{test_files_count} test images**
- This makes a test size of **{test_percent}%**
""")

st.write('### The raw labels dataframe')
st.markdown(f"""
- `train.csv` has **{df.shape[0]} rows** and **{df.shape[1]} columns**:
  `Image_Label` and `EncodedPixels` - both of type `object`
- There are **4 rows per image**, one for each of the categories
  **Fish**, **Flower**, **Gravel** and **Sugar**
- The category is *appended to the file name* in the column `Image_Label`.
  So this column holds *two* informations that should be separated
- `EncodedPixels` is `NaN` when an image is not labeled for that category
  ({df['EncodedPixels'].isna().sum()} of {df.shape[0]} rows) - otherwise it
  contains the **RLE encoded** mask that the scientists marked in the image
""")

st.dataframe(first_image_rows)
