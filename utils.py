import numpy as np


def postprocess_image(
    img : np.ndarray,
    dtype : np.dtype,
    percentile : int = 99
    ) -> np.ndarray:
    """
    Normalize input image

    Parameters
    ----------
    image : np.ndarray
        Image array
    dtype : np.dtype
        Dtype of input image
    percentile : int 
        Percentile rank for normalized image

    Returns
    -------
    output : np.ndarray
        Normalized image
    """
    max_val = None
    if dtype == np.uint8:
        max_val = 2 ** 8 - 1
    elif dtype == np.uint16:
        max_val = 2 ** 16 - 1
    elif dtype == np.float32:
        max_val = 1
    else:
        raise NotImplementedError("Not implemented for dtype={}".format(dtype))
    img_normalized = (
        img
        * float(max_val)
        / np.maximum(max_val, np.percentile(img, percentile, axis=(0, 1)))
    )
    img_normalized[img_normalized > max_val] = max_val
    img_normalized[img_normalized < 0] = 0
    img_normalized = img_normalized.astype(dtype=dtype)
    return img_normalized


def load_img(
    filepath : str
    ) -> np.ndarray:
    """
    Load image

    Parameters
    ----------
    filepath : str
        Path to image file

    Returns
    -------
    output : np.ndarray
        Image array
    """
    import skimage.io as imgio
    from tifffile import TiffFileError

    try:
        return imgio.imread(filepath)
    except TiffFileError as tfe:
        raise ValueError("Couldn't load image", filepath, tfe)


def save_img(
    image : np.ndarray,
    filepath : str,
    dtype : np.dtype = np.uint8
    ) -> None:
    """
    Save image

    Parameters
    ----------
    image : np.ndarray
        Image array
    filepath : str
        Path to image file
    dtype : np.dtype
        Dtype of input image. Default value is np.uint8

    Returns
    -------
    output : None
    """
    import skimage.io as imgio

    if image.dtype in (np.dtype(np.uint8), np.dtype(np.uint16)):
        imgio.imsave(filepath, image)
    elif image.dtype == np.dtype(np.float32):
        image = (image * np.iinfo(dtype).max + 0.5).astype(dtype)
        imgio.imsave(filepath, image)
    else:
        raise ValueError(
            "save_img: not implemented for image dtype", image.dtype
        )
    if dtype not in (np.uint8, np.uint16):
        raise ValueError(
            "save_img: implemented for np.uint8 or np.uint16 only", image.dtype
        )