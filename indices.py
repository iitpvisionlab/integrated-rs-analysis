import numpy as np
from sklearn.decomposition import PCA


def sdi(
    rgbn : np.ndarray
    ) -> np.ndarray:
    """
    Calculate sdi index

    Parameters
    ----------
    rgbn : np.ndarray
        Array consisting of normalized red, green, blue and nir channels.

    Returns
    -------
    output : np.ndarray
        Array consisting of sdi index
    """
    pca = PCA(n_components=1)
    rgbn_ = rgbn.reshape(rgbn.shape[0] * rgbn.shape[1], rgbn.shape[2])
    pc = pca.fit_transform(rgbn_).reshape(-1).reshape(rgbn[..., 0].shape)
    sdi = ((1 - pc) + 1) / (((rgbn[..., 1] - rgbn[..., 2]) * rgbn[..., 0] + 1))
    norm_sdi = (sdi - np.min(sdi)) / (np.max(sdi - np.min(sdi)))
    sdi_m = np.clip((norm_sdi - rgbn[..., 0]), 0, 1)
    return sdi_m


def ndwi(
    green : np.ndarray,
    nir : np.ndarray
    ) -> np.ndarray:
    """
    Calculate ndwi index

    Parameters
    ----------
    green : np.ndarray
        Array consisting of normalized green channel.
    nir : np.ndarray
        Array consisting of normalized nir channel.

    Returns
    -------
    output : np.ndarray
        Array consisting of ndwi index
    """
    return (green - nir) / (green + nir)


def ndvi(
    red : np.ndarray,
    nir : np.ndarray
    ) -> np.ndarray:
    """
    Calculate ndvi index

    Parameters
    ----------
    red : np.ndarray
        Array consisting of normalized red channel.
    nir : np.ndarray
        Array consisting of normalized nir channel.

    Returns
    -------
    output : np.ndarray
        Array consisting of ndvi index
    """
    red_ = np.asarray(red, dtype=np.float32)
    nir_ = np.asarray(nir, dtype=np.float32)
    vi = (nir_ - red_) / (nir_ + red_)
    vi[~np.isfinite(vi)] = 0
    return vi
