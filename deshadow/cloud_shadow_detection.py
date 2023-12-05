from indices import sdi as calculate_sdi
from indices import ndwi as calculate_ndwi
import numpy as np
from skimage.feature import match_template
import cv2
import math as m
from skimage import filters


def match_shadow_cloud(shadow, cloud):
    pad_cloud = np.pad(cloud, (300, 300))
    tmp = np.zeros_like(cloud)
    size = 500
    x_rat = shadow.shape[0] / size
    y_rat = shadow.shape[1] / size
    for i in range(int(m.ceil(x_rat))):
        for j in range(int(m.ceil(y_rat))):
            if i == m.ceil(x_rat) - 1:
                x_size = shadow.shape[0]
            else:
                x_size = (i + 1) * size
            if j == m.ceil(y_rat) - 1:
                y_size = shadow.shape[0]
            else:
                y_size = (j + 1) * size
            small_tmp = shadow[i * size : x_size, j * size : y_size]
            small_cloud = pad_cloud[i * size : x_size + 600, j * size : y_size + 600]
            res = match_template(small_cloud, small_tmp)
            x, y = np.unravel_index(np.argmax(res), res.shape)
            template_width, template_height = small_tmp.shape
            temp_cloud = small_cloud[x : x + template_width, y : y + template_height]
            tmp[i * size : x_size, j * size : y_size] = temp_cloud
    return tmp & shadow


def get_masks(rgbn):
    sdi = calculate_sdi(rgbn) * 255
    # cloud_th = filters.threshold_minimum(sdi)
    # cloud_th = 200
    # no_clouds = sdi[sdi > cloud_th]
    # th = filters.threshold_minimum(no_clouds)
    # th = 228
    # cloud = sdi < cloud_th
    # shadow = sdi > th
    cloud = sdi / sdi.max() < 0.1
    shadow = sdi / sdi.max() > 0.9
    # ndwi = calculate_ndwi(rgbn[..., 1], rgbn[..., 3])
    # shadow_wo_water = np.clip(shadow*255 - (ndwi > 0)*255, 0, 255)
    temp_cloud = match_shadow_cloud(shadow, cloud)
    # summa = np.clip(shadow_wo_water+(temp_cloud)*255, 0, 255)
    # blur = cv2.medianBlur((summa).astype(np.uint8), 5)
    blur = cv2.medianBlur((temp_cloud * 255).astype(np.uint8), 5)
    shadow_mask = (np.clip(blur - cloud * 255, 0, 255)).astype(np.uint8)

    return (cloud * 255).astype(np.uint8), shadow_mask
