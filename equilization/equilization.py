import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt
import rasterio
from sklearn.metrics import mean_squared_error
from sklearn import linear_model


def fn(c, d, r, l):
    """Calculate metric of smoothness of a function

    Args:
        c (array) : 1st coefficient at current step
        d (array) : 2st coefficient at current step
        r (float) : scale coefficient
        l (float) : norm type
        
    Returns:
        float : metric of smoothness of a function
    """
    filter_c = np.array([c[i] - c[i-1] for i in range(1, len(c))])
    filter_d = np.array([d[i] - d[i-1] for i in range(1, len(d))])
    ec = np.abs(filter_c)**l
    ed = np.abs(filter_d)**l
    e = r * (np.sum(ec) + np.sum(ed))
    return e


def dn(x, l):
    """Calculate the term of regularization coefficient

    Args:
        x (array) : coefficient in need of regularization
        l (float) : norm of regularization
        
    Returns:
        numpy ndarray: term of regularization coefficient
    """
    dx = np.zeros((x.shape[0]+1))
    dx[0] = x[0]
    dx[1:] = x[:]
    dx =  np.array([dx[i] - dx[i-1] for i in range(1, len(dx))])
    s = -np.sign(dx)
    grx = l * s * np.abs(dx) ** (l - 1)
    dgrx = np.zeros((grx.shape[0]+1))
    dgrx[-1] = grx[-1]
    dgrx[:-1] = grx[:]
    return np.array([dgrx[i] - dgrx[i+1] for i in range(0, len(dgrx)-1)])



def hcn(a, b, c, d, r, l):
     """Calculate gradient for 1st affine model coefficient

    Args:
        a (array) : array of clues with normal illumination
        b (array) : array of clues with difficult illumination
        c (array) : 1st coefficient at current step
        d (array) : 2st coefficient at current step
        r (float) : scale regularization coefficient
        l (float) : norm of regularization
        
    Returns:
        numpy ndarray: gradient for 1st affine model coefficient
    """
     return np.sum(2*(b*c + d - a)*b, axis=0) + r*dn(c, l)


def hdn(a, b, c, d, r, l):
    """Calculate gradient for 2st affine model coefficient

    Args:
        a (array) : array of clues with normal illumination
        b (array) : array of clues with difficult illumination
        c (array) : 1st coefficient at current step
        d (array) : 2st coefficient at current step
        r (float) : scale regularization coefficient
        l (float) : norm of regularization
        
    Returns:
        numpy ndarray: gradient for 2st affine model coefficient
    """
    return np.sum(2*(b*c + d - a), axis=0) + r*dn(d,l)


def regularization_n(a, b, r, l, t, n):
    """Calculate coefficients for affine model with regularization

    Args:
        a (array) : array of clues with normal illumination
        b (array) : array of clues with difficult illumination
        r (float) : scale regularization coefficient
        l (float) : norm of regularization
        t (float) : marginal increase in regularization accuracy at adjacent stages
        n (int) : threshold for regularization steps
        
    Returns:
        numpy ndarray: multispectral image with gain correction
    """
    co = np.maximum(np.mean(a, axis=0)/(np.mean(b, axis=0)+0.01), 0.01)
    do = np.maximum(np.mean(a, axis=0) - co*np.mean(b, axis=0), 0.01)

    e = fn(co, do, r, l)
    s = 1.0
    i = 0
    lr = 0.00001
    y_change = e
    
    while i <= n and y_change >= t:
        tmp_c = co - lr * hcn(a, b, co, do, r, l)
        tmp_d = do - lr * hdn(a, b, co, do, r, l)
        tmp_y = fn(tmp_c, tmp_d, r, l)
        y_change = np.abs(tmp_y - e)
        e = tmp_y
        co = tmp_c
        do = tmp_d
        i += 1
    return co, do


def rad(path, gain):
    """Read multispectral image with gain correction

    Args:
        path (str) : path to image file
        gain (array) : array of gain coefficients
        
    Returns:
        numpy ndarray: multispectral image with gain correction
    """
    img = rasterio.open(path).read()
    return np.mean(img, axis=(1,2))/gain

def read_gain(gain_path):
    """Read *.gain file into array

    Args:
        gain_path (str) : path to *.gain file
        
    Returns:
        array : array of gain coefficients
    """
    
    with open(gain_path, 'r') as f:
        data = f.read()
    gain = []
    for i in data.split('\n'):
        g = [f for f in i.split(' ') if f]
        if g:
            gain.append(float(g[0]))
    return gain