import numpy as np


def decode_rle(rle_string, shape):
    """Decode a run-length-encoded segmentation mask.

    Args:
        rle_string: Space-separated (start, length) pairs with 1-based,
            column-wise pixel indices. Non-string values (e.g. NaN) yield
            an all-zero mask.
        shape: Mask shape as (height, width).

    Returns:
        np.ndarray of uint8 with the given shape; 1 = masked, 0 = background.
    """
    # Create an empty mask of given shape
    mask_height = shape[0]
    mask_width = shape[1]
    mask = np.zeros(mask_height * mask_width, dtype=np.uint8)

    # Return the empty mask if no proper RLE string
    if not isinstance(rle_string, str):
        return mask.reshape(shape, order="F")  # Fortran order = column-wise!

    # Decode the rle_string into the mask
    values = list(map(int, rle_string.split()))
    for i in range(0, len(values), 2):
        start = values[i]
        length = values[i + 1]
        mask[start - 1: start - 1 + length] = 1

    # Return decoded mask
    return mask.reshape(shape, order="F")  # Fortran order = column-wise!
