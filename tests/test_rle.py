import numpy as np

from src.utils.rle import decode_rle


def test_rle_decode_simple_run():
    # Pixels are numbered 1-based and column-wise (Fortran order):
    # pixel 1 = (0,0), pixel 2 = (1,0), pixel 3 = (0,1), pixel 4 = (1,1)
    # "2 2" means: starting at pixel 2, set two pixels -> pixels 2 and 3
    mask = decode_rle("2 2", shape=(2, 2))

    expected = np.array([[0, 1],
                         [1, 0]],
                        dtype=np.uint8)
    np.testing.assert_array_equal(mask, expected)


def test_rle_decode_run_with_a_bigger_shape():
    mask = decode_rle("2 3", shape=(3, 4))

    # NOTE:
    # To meet the Kaggle RLE conventions the array was filled by rows, not as
    # you might expect by columns. So, instead of looking like:
    #       [0, 1, 1, 1]
    #       [0, 0, 0, 0]
    #       [0, 0, 0, 0]
    # it looks like this:
    expected = np.array([[0, 1, 0, 0],
                         [1, 0, 0, 0],
                         [1, 0, 0, 0]],
                        dtype=np.uint8)
    np.testing.assert_array_equal(mask, expected)


def test_rle_decode_nan_returns_empty_mask():
    # In train.csv, EncodedPixels is NaN when an image has no label
    # for the category -> mask should be all zeros
    mask = decode_rle(float("nan"), shape=(4, 6))

    assert mask.shape == (4, 6)
    assert mask.dtype == np.uint8
    assert mask.sum() == 0
