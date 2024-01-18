import numpy as np
from cv2.ximgproc import guidedFilter
from skimage.color import rgb2hsv, hsv2rgb
from scipy.ndimage import gaussian_filter
from skimage.metrics import mean_squared_error as mse


def get_dark_value(
    x : int,
    y : int,
    I : np.ndarray, 
    dx : int = 7,
    dy : int = 7
    ) -> int:
    """
    Get minimal value through all the channels in considered window for one pixel by its coords

    Returns
    -------
    output : int
        Dark value

    Parameters
    ----------
    x, y : int 
        Pixel coordinates
    I : numpy ndarray
        Image, shape (x_size, y_size, 3)
    dx, dy : int (optional)
        Window size. Defaults to (7, 7).
    """
    r = np.min(I[x - dx : x + dx + 1, y - dy : y + dy + 1])
    return r


def get_dark_channel(
    I : np.ndarray, 
    dx : int ,
    dy : int, 
    ) -> np.ndarray:
    """
    Return dark channnel of image

    Returns
    -------
    output : numpy ndarray
        Image (x_size, y_size)

    Parameters
    ----------
    I : numpy ndarray
        Image, shape (x_size, y_size)
    dx, dy : int
        Window size.
    """
    x_size, y_size = I.shape[:2]
    I_dc = np.zeros_like(I)
    pad_I = np.pad(I, ((dx, dx), (dy, dy), (0, 0)), mode="reflect")
    for x in range(dx, x_size + dx):
        for y in range(dy, y_size + dy):
            I_dc[x - dx, y - dy] = np.min(
                pad_I[x - dx : x + dx + 1, y - dy : y + dy + 1]
            )
    return I_dc[..., 0]


def window_min(
    I : np.ndarray, 
    dx : int ,
    dy : int, 
    ) -> np.ndarray:
    """
    Window minimum filter for an image

    Returns
    -------
    output : numpy ndarray
        Image (x_size, y_size)

    Parameters
    ----------
    I : numpy ndarray
        Image, shape (x_size, y_size)
    dx, dy : int
        Window size.
    """
    x_size, y_size = I.shape
    I_dc = np.zeros_like(I)
    pad_I = np.pad(I, ((dx, dx), (dy, dy)), mode="edge")
    for x in range(dx, x_size + dx):
        for y in range(dy, y_size + dy):
            I_dc[x - dx, y - dy] = get_dark_value(x, y, pad_I, dx, dy)

    return I_dc


def zhu_depth_estim(
    I : np.ndarray,
    dx : int = 7,
    dy : int = 7,
    r : int = 30,
    eps : float = 0.01,
    gf_on : bool = True
    ) -> np.ndarray:
    """
    Atmospheric Light Estimation Based Remote Sensing Image Dehazing by
    Z. Zhu et. al.: https://www.mdpi.com/2072-4292/13/13/2432/htm
    Depth map estimation

    Returns
    -------
    output : numpy ndarray
        Depth map (x_size, y_size)

    Parameters
    ----------
    I : numpy ndarray
        Image, shape (x_size, y_size, 3)
    dx, dy : int (optional)
        Window size. Defaults to (7, 7)
    r : int (optional)
        Guided filter radius. Defaults to 30.
    eps : float (optional)
        Reg param for guided filter. Defaults to 0.01.
    gf_on : bool (optional)
        Apply guided filter or not. Defaults to True.
    """

    I_hsv = rgb2hsv(I)
    w0 = 0.121779
    w1 = 0.959710
    w2 = -0.780245
    sigma = 0.041337

    d = (w0 + w1 * I_hsv[:, :, 2] + w2 * I_hsv[:, :, 1]).astype(np.float32)
    d = gaussian_filter(d, sigma)
    d = window_min(d, dx, dy)

    if gf_on:
        d = guidedFilter(I, d, r, eps)

    return d

def cadcp(
    I : np.ndarray,
    dx : int = 7,
    dy : int = 7,
    k : float = 0.95,
    r : int = 30,
    eps : float = 0.01,
    d_quantile : float = 0.999,
    gf_on : bool = True,
    a_mean : bool = True
) -> np.ndarray:
    """
    Color attenuation prior and Dark Channel Prior dehazing algorithm

    Returns
    -------
    output : numpy ndarray
        Dehazed image (x_size, y_size)

    Parameters
    ----------
    I : numpy ndarray
        Image, shape (x_size, y_size, 3)
    dx, dy : int (optional)
        Window size. Defaults to (7, 7)
    k : float (optional)
        Reg param for dehazed image. Defaults to 0.95.
    r : int (optional)
        Guided filter radius. Defaults to 30.
    eps : float (optional)
        Reg param for guided filter. Defaults to 0.01.
    d_quantile : float (optional)
        Quantile for veil color estimation. Defaults to 0.999.
    gf_on : bool (optional)
        Apply guided filter or not. Defaults to True.
    a_mean : bool (optional)
        Average to estimate veil color or not . Defaults to True.
    """

    d = zhu_depth_estim(I, dx, dy, r, eps, gf_on)

    q_d = np.quantile(d, d_quantile)
    I_intens = I.sum(axis=2)

    if a_mean:  
        a_rgb = (I[d >= q_d]).mean(axis=0)
    else:  
        I_max = I_intens[d >= q_d].max()
        a_rgb = I[(d >= q_d) & (I_intens == I_max)].mean(axis=0)
    A = a_rgb * np.ones(I.shape)

    I_a = I / A
    V = get_dark_channel(I_a, dx, dy).astype(np.float32)
    t = 1 - k * V
    t = guidedFilter(I, t, r, eps)
    t = np.dstack((t, t, t))

    J = (I - A) / (np.minimum(np.maximum(t, 0.1), 0.9)) + A
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J



def calculate_metrics(
    orig : np.ndarray,
    img : np.ndarray,
    metrics : dict = {"MSE" : mse}
    ) -> list:
    """
    Quality metrics calculator

    Returns
    -------
    output : list
        Metrics values

    Parameters
    ----------
    orig : numpy ndarray
        Clear image
    img : numpy ndarray
        Dehazed image
    metrics : dict (optional)
        Metrics functions. Defaults to MSE.
    """
    m_vals = []

    for m in metrics:
        m_vals.append(np.around(metrics[m](orig, img), decimals=3))
    return m_vals

