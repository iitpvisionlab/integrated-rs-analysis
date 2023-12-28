import numpy as np
import rasterio



class AffineModel:
    c = 0
    d = 0

    def _error_func(self, c, d, r, l):
        filter_c = np.array([c[i] - c[i - 1] for i in range(1, len(c))])
        filter_d = np.array([d[i] - d[i - 1] for i in range(1, len(d))])
        ec = np.abs(filter_c) ** l
        ed = np.abs(filter_d) ** l
        e = r * (np.sum(ec) + np.sum(ed))
        return e

    def _dn(self, x, l):
        dx = np.zeros((x.shape[0] + 1))
        dx[0] = x[0]
        dx[1:] = x[:]
        dx = np.array([dx[i] - dx[i - 1] for i in range(1, len(dx))])
        s = -np.sign(dx)
        grx = l * s * np.abs(dx) ** (l - 1)
        dgrx = np.zeros((grx.shape[0] + 1))
        dgrx[-1] = grx[-1]
        dgrx[:-1] = grx[:]
        return np.array(
            [dgrx[i] - dgrx[i + 1] for i in range(0, len(dgrx) - 1)]
        )

    def _hc(self, a, b, c, d, r, l):
        return np.sum(2 * (b * c + d - a) * b, axis=0) + r * self._dn(c, l)

    def _hd(self, a, b, c, d, r, l):
        return np.sum(2 * (b * c + d - a), axis=0) + r * self._dn(d, l)

    def fit(self, a, b, r=0.0025, l=2.0, t=1e-14, lr=0.0001, n=1000000):
        """
        Calculate coefficients for affine model with regularization.

        Returns
        -------
        output : float, float
            Return k_1, k_2 coefficeint

        Parameters
        ----------
        a : array
            Array of clues with normal illumination
        b : array
            Array of clues with difficult illumination
        r : float
            Scale regularization coefficient
        l : float  
            Norm of regularization
        t : float  
            Marginal increase in regularization accuracy at adjacent stages
        n : int
            Threshold for regularization steps
        """
        co = np.maximum(
            np.mean(a, axis=0) / (np.mean(b, axis=0) + 0.0001), 0.0001
        )
        do = np.maximum(np.mean(a, axis=0) - co * np.mean(b, axis=0), 0.0001)

        e = self._error_func(co, do, r, l)
        y_change = e
        i = 0
        while i <= n and y_change >= t:
            tmp_c = co - lr * self._hc(a, b, co, do, r, l)
            tmp_d = do - lr * self._hd(a, b, co, do, r, l)
            tmp_y = self._error_func(tmp_c, tmp_d, r, l)
            y_change = np.abs(tmp_y - e)
            e = tmp_y
            co = tmp_c
            do = tmp_d
            i += 1
        self.c = co
        self.d = do
        return co, do

    def predict(self, x):
        """
        Calculate the irradiance spectrum under target conditions

        Returns
        -------
        output : array
            The irradiance spectrum under target conditions

        Parameters
        ----------
        x : array
            The irradiance spectrum under original shooting conditions
        """
        return self.c * x + self.d



def rad(path, gain):
    """
    Read multispectral image with gain correction

    Returns
    -------
    output : numpy ndarray
        Multispectral image with gain correction

    Parameters
    ----------
    path : str
        Path to image file
    gain : array
        Array of gain coefficients
    """
    img = rasterio.open(path).read()
    return np.mean(img, axis=(1,2))/gain

def read_gain(gain_path):
    """
    Read *.gain file into array

    Returns
    -------
    output : array
        Array of gain coefficients

    Parameters
    ----------
    gain_path : str
        Path to *.gain file
    """
    
    with open(gain_path, 'r') as f:
        data = f.read()
    gain = []
    for i in data.split('\n'):
        g = [f for f in i.split(' ') if f]
        if g:
            gain.append(float(g[0]))
    return gain