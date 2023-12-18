import numpy as np
from skimage.measure import label, regionprops
from typing import Union


def calculate_vector_list(
    image: np.ndarray,
    current_mask: np.ndarray,
    cloud_mask: np.ndarray,
    shadow_mask: np.ndarray,
    offset: int,
) -> list:
    """
    Calculate list of vectors of shading coefficients for each channel.

    Parameters
    ----------
    image : (H, W, N) np.ndarray
        Multispectral image with shadows where H is height, W is width and N
        is number of channels.
    current_mask : (H, W) np.ndarray
        Binary mask of single shadow or all shadows for overall vector.
    cloud_mask : (H, W) np.ndarray
        Binary cloud mask.
    shadow_mask : (H, W) np.ndarray
        Binary shadowmask.
    offset : int
        Offset from shadow border.

    Returns
    -------
    output : list
        List of vectors of shading coefficients for each channel.
    """
    dx_mask, dy_mask = np.gradient(current_mask)
    vector_list = list()
    for x in range(current_mask.shape[0]):
        for y in range(current_mask.shape[1]):
            if current_mask[x, y] and not cloud_mask[x, y]:
                dx = int(dx_mask[x, y] * offset)
                dy = int(dy_mask[x, y] * offset)
                if dx or dy:
                    try:
                        in_ = image[x + dx, y + dy]
                        out = image[x - dx, y - dy]
                        if (
                            not cloud_mask[x - dx, y - dy]
                            and not shadow_mask[x - dx, y - dy]
                            and shadow_mask[x + dx, y + dy]
                            and np.all(out) != 0
                        ):
                            vector_list.append(in_ / out)
                    except IndexError:
                        pass
    return vector_list


def calculate_vector(
        image: np.ndarray,
        current_mask: np.ndarray,
        cloud_mask: np.ndarray,
        shadow_mask: np.ndarray,
        offset: int,
        method: str
) -> Union[np.ndarray, None]:
    """
    Estimate vector of shading coefficients for each channel.

    Parameters
    ----------
    image : (H, W, N) np.ndarray
        Multispectral image with shadows where H is height, W is width and N
        is number of channels.
    current_mask : (H, W) np.ndarray
        Binary mask of single shadow or all shadows for overall vector.
    cloud_mask : (H, W) np.ndarray
        Binary cloud_mask current_mask.
    shadow_mask : (H, W) np.ndarray
        Binary shadow_mask current_mask.
    offset : int
        Offset from shadow_mask border.
    method : str
        Method of shading coefficients vector calculation. There are
        implementations for two methods: `med` for robast median estimation
        and `avr` for average estimation.

    Returns
    -------
    output : Union[np.ndarray, None]
        Vector of shading coefficients for each channel or None.

    Notes
    -----
    Function returns None if it is impossible to estimate the vector,
    for example, due to surrounding clouds.
    """
    vector_list = calculate_vector_list(image, current_mask,
                                        cloud_mask, shadow_mask, offset)
    if len(vector_list) == 0:
        return None
    if method == "med":
        return np.median(vector_list, axis=0)
    elif method == "avr":
        return np.mean(vector_list, axis=0)
    else:
        raise ValueError(f"unknown method {method}.")


def deshadow_image(
    image: np.ndarray,
    cloud_mask: np.ndarray,
    shadow_mask: np.ndarray,
    offset: int = 30,
    method: str = "med",
    use_overall: bool = True
) -> np.ndarray:
    """
    Function for deshadowing multispectral image by estimation of shading
    coefficients vector. Details of the method are presented in [1].

    Parameters
    ----------
    image : (H, W, N) np.ndarray
        Multispectral image with shadows where H is height, W is width and N
        is number of channels.
    cloud_mask : (H, W) np.ndarray
        Binary cloud_mask current_mask.
    shadow_mask : (H, W) np.ndarray
        Binary shadow_mask current_mask.
    offset : int
        Offset from shadow_mask border. Defalut is 30.
    method : str
        Method of shading coefficients vector calculation. There are
        implementations for two methods: `med` for robast median estimation
        and `avr` for average estimation. Default is `med`.
    use_overall : bool
        If True, overall vector of shading coefficients will be calculated and
        applied to shadows for which it was not possible to calculate the
        individual vector. Default is True.

    Returns
    -------
    outpit : (H, W, N) np.ndarray
        Multispectral image without shadows.

    References
    ----------
    .. [1]  D. A. Bocharov, D. P. Nikolaev, M. A. Pavlova and V. A. Timofeev,
            “Cloud Shadows Detection and Compensation Algorithm on
            Multispectral Satellite Images for Agricultural Regions,”
            JCTE, vol. 67, no 6, pp. 728-739, 2022,
            :DOI:`10.1134/S1064226922060171`.
    """
    not_processed = list()
    processed_image = image.copy()
    shadow_label = label(shadow_mask)
    regions = regionprops(shadow_label)
    valid_regions = [r for r in regions if r.area > np.pi * (offset**2) / 4]
    for reg in valid_regions:
        current_mask = (shadow_label == reg.label) * 1.0
        coef_vector = calculate_vector(image, current_mask, cloud_mask,
                                       shadow_mask, offset, method)
        if coef_vector is None or not np.all(coef_vector):
            not_processed.append(reg.label)
        else:
            processed_image[shadow_label == reg.label] = (
                processed_image[shadow_label == reg.label] / coef_vector
            )
    if use_overall:
        overall_coef_vector = calculate_vector(
            image, shadow_mask, cloud_mask, shadow_mask, offset, method
        )
        for i in not_processed:
            processed_image[shadow_label == i] = (
                processed_image[shadow_label == i] / overall_coef_vector
            )
    return processed_image
