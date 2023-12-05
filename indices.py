import numpy as np
from sklearn.decomposition import PCA


def sdi(rgbn):
    pca = PCA(n_components=1)
    rgbn_ = rgbn.reshape(rgbn.shape[0] * rgbn.shape[1], rgbn.shape[2])
    pc = pca.fit_transform(rgbn_).reshape(-1).reshape(rgbn[..., 0].shape)
    # norm_pc = (pc - np.min(pc)) / (np.max(pc) - np.min(pc))
    sdi = ((1 - pc) + 1) / (((rgbn[..., 1] - rgbn[..., 2]) * rgbn[..., 0] + 1))
    norm_sdi = (sdi - np.min(sdi)) / (np.max(sdi - np.min(sdi)))
    sdi_m = np.clip((norm_sdi - rgbn[..., 0]), 0, 1)
    return sdi_m


def ndwi(green, nir):
    return (green - nir) / (green + nir)


def ndvi(red, nir):
    red_ = np.asarray(red, dtype=np.float32)
    nir_ = np.asarray(nir, dtype=np.float32)
    vi = (nir_ - red_) / (nir_ + red_)
    vi[~np.isfinite(vi)] = 0
    return vi
