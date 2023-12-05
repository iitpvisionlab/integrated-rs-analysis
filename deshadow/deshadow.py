import numpy as np
from skimage.measure import label, regionprops


def calculate_coefs_matrix(
    img: np.ndarray,  # raw band
    mask: np.ndarray,  # binary mask of single shadow
    cloud: np.ndarray,  # binary mask of all clouds
    shadow: np.ndarray,  # binary mask of all shadows
    offset: int = 30,  # offset for mask
):
    dx_mask, dy_mask = np.gradient(mask)
    coefs_matrix = list()
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x, y] and not cloud[x, y]:
                dx = int(dx_mask[x, y] * offset)
                dy = int(dy_mask[x, y] * offset)
                if dx or dy:
                    try:
                        in_ = img[x + dx, y + dy]
                        out = img[x - dx, y - dy]
                        if (
                            not cloud[x - dx, y - dy]
                            and not shadow[x - dx, y - dy]
                            and shadow[x + dx, y + dy]
                            and np.all(out) != 0
                        ):
                            coefs_matrix.append(in_ / out)
                    except IndexError:
                        pass
    return coefs_matrix


def calculate_coef(img, mask, cloud, shadow, offset=30, method="med"):
    coefs = calculate_coefs_matrix(img, mask, cloud, shadow, offset)
    if not coefs:
        return None
    if method == "med":
        return np.median(coefs, axis=0)
    elif method == "avr":
        return np.mean(coefs, axis=0)
    else:
        raise ValueError(f"unknown method {method}")


def deshadow_image(
    image, cloud_mask, shadow_mask, offset=30, method="med", use_overall=True
):
    not_processed = list()
    processed_image = image.copy()
    shadow_label = label(shadow_mask)
    regions = regionprops(shadow_label)
    valid_regions = [r for r in regions if r.area > np.pi * (offset**2) / 4]
    for reg in valid_regions:
        mask = (shadow_label == reg.label) * 1.0
        coef = calculate_coef(image, mask, cloud_mask, shadow_mask, offset, method)
        if coef is None or not np.all(coef):
            not_processed.append(reg.label)
        else:
            processed_image[shadow_label == reg.label] = (
                processed_image[shadow_label == reg.label] / coef
            )
    if use_overall:
        overall_coef = calculate_coef(
            image, shadow_mask, cloud_mask, shadow_mask, offset, method
        )
        for i in not_processed:
            processed_image[shadow_label == i] = (
                processed_image[shadow_label == i] / overall_coef
            )
    return processed_image
