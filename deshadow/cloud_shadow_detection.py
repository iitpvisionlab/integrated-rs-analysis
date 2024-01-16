import numpy as np
import math as m

from cv2 import medianBlur
from skimage.feature import match_template
from skimage.filters import threshold_minimum

from indices import sdi as calculate_sdi
from indices import ndwi as calculate_ndwi


def match_shadow_cloud(
        shadow: np.ndarray,
        cloud: np.ndarray,
        window_size: int = 500,
        padding_size: int = 300
) -> np.ndarray:
    """
    Function for matching shadow and cloud.

    Parameters
    ----------
    shadow : (H, W) nd.ndarray
        Binary mask of shadow.
    cloud : (H, W) np.ndarray
        Binary mask of cloud.
    window_size : int
        Determines the square image fragment size used to match cloud and
        shadow masks. Increasing this value increases the quality of the
        match, but also increases the amount of RAM required.
        Default value is 500.
    padding_size : int
        Determines the area size of the cloud mask fragment in which matching
        will occur with the shadow mask fragment. For example,
        if window_size=500 and padding_size=300, then the matching search area
        will be a fragment of size (800, 800). Thus, the offset of the shadow
        mask relative to the cloud mask cannot be greater than padding_size.
        Default value is 300.

    Returns
    -------
    output : np.ndarray
        Binary shadow mask.
    """
    pad_cloud = np.pad(cloud, (padding_size, padding_size))
    tmp = np.zeros_like(cloud)
    x_rat = shadow.shape[0] / window_size
    y_rat = shadow.shape[1] / window_size
    for i in range(int(m.ceil(x_rat))):
        for j in range(int(m.ceil(y_rat))):
            if i == m.ceil(x_rat) - 1:
                x_size = shadow.shape[0]
            else:
                x_size = (i + 1) * window_size
            if j == m.ceil(y_rat) - 1:
                y_size = shadow.shape[0]
            else:
                y_size = (j + 1) * window_size
            small_tmp = shadow[
                i * window_size: x_size, j * window_size: y_size
            ]
            small_cloud = pad_cloud[
                i * window_size: x_size + padding_size * 2,
                j * window_size: y_size + padding_size * 2,
            ]
            res = match_template(small_cloud, small_tmp)
            x, y = np.unravel_index(np.argmax(res), res.shape)
            template_width, template_height = small_tmp.shape
            temp_cloud = small_cloud[
                x: x + template_width, y: y + template_height
            ]
            tmp[i * window_size: x_size, j * window_size: y_size] = temp_cloud
    return tmp & shadow


def get_masks(rgbn: np.ndarray, window_size: int = 500,
              padding_size: int = 300) -> tuple((np.ndarray, np.ndarray)):
    """
    Function for calculating cloud and shadow masks from red, green, blue and
    nir channels. Details of the method are presented in [1].

    Parameters
    ----------
    rgbn : (H, W, 4) np.ndarray
        Array consisting of normalized red, green, blue and nir channels.
    window_size : int
        Determines the square image fragment size used to match cloud and
        shadow masks. Increasing this value increases the quality of the
        match, but also increases the amount of RAM required.
        Default value is 500.
    padding_size : int
        Determines the area size of the cloud mask fragment in which matching
        will occur with the shadow mask fragment. For example,
        if window_size=500 and padding_size=300, then the matching search area
        will be a fragment of size (800, 800). Thus, the offset of the shadow
        mask relative to the cloud mask cannot be greater than padding_size.
        Default value is 300.

    Returns
    -------
    output : tuple of np.ndarray (H, W)
        Cloud and shadow masks in uint8.

    References
    ----------
    .. [1]  D. A. Bocharov, D. P. Nikolaev, M. A. Pavlova and V. A. Timofeev,
            “Cloud Shadows Detection and Compensation Algorithm on
            Multispectral Satellite Images for Agricultural Regions,”
            JCTE, vol. 67, no 6, pp. 728-739, 2022,
            :DOI:`10.1134/S1064226922060171`.
    """
    sdi = calculate_sdi(rgbn) * 255
    try:
        cloud_th = threshold_minimum(sdi)
        no_clouds = sdi[sdi > cloud_th]
        th = threshold_minimum(no_clouds)
    except RuntimeError:
        th = 0
        cloud_th = 0
        print("Could not find thresholds.")
    cloud = sdi < cloud_th
    shadow = sdi > th
    ndwi = calculate_ndwi(rgbn[..., 1], rgbn[..., 3])
    shadow_wo_water = np.clip(shadow * 255 - (ndwi > 0) * 255, 0, 255)
    temp_shadow = match_shadow_cloud(shadow, cloud, window_size=window_size,
                                     padding_size=padding_size)
    summa = np.clip(shadow_wo_water + temp_shadow * 255, 0, 255)
    blur = medianBlur((summa).astype(np.uint8), 5)
    shadow_mask = (np.clip(blur - cloud * 255, 0, 255)).astype(np.uint8)

    return ((cloud * 255).astype(np.uint8), shadow_mask)
