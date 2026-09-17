"""Properties of the *full* Cloud Segmentation dataset.

The Streamlit app presents the dataset we worked with, not the handful of
files that happen to be deployed with it (see
`src.data.build_presentation_data`). Numbers that describe the whole dataset
therefore must not be measured at runtime by counting files in a directory -
they either get derived from the complete labels table in
`data/presentation/labels_slim.csv.gz`, or, when they are not in there at all,
they live here.

Only the test set falls into the second group: `train.csv` covers the training
images alone, so the number of test images cannot be read off the labels.

Source: https://www.kaggle.com/competitions/understanding_cloud_organization
"""

# Number of images in `data/raw/test_images`. Unlike the training images this
# cannot be derived from the labels, because the test images are unlabeled.
NUM_TEST_IMAGES = 3698
