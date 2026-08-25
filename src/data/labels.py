from pathlib import Path

import pandas as pd


def load_processed_labels(path: Path) -> pd.DataFrame:
    """Loads a Cloud Segmentation labels CSV file from given path. Splits the
    column `Image_label` into the two newly created columns `Image` and `Label`
    and returns the resulting df.
    """
    df = pd.read_csv(path)

    # Separate the label from the image name: Split the column 'Image_Label' at
    # '_' and store the resulting 2 values in newly created columns 'Image' and
    # 'Label'
    df[['Image', 'Label']] = df['Image_Label'].str.split('_', n=1, expand=True)

    # Remove the 'Image_Label' column; it's not needed anymore
    df = df.drop(columns='Image_Label')
    return df
