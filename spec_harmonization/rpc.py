import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from math import factorial

# regularizing coefficient


class RPC:
    def __init__(self, degree: int, reg_coef: float = 0.0) -> None:
        self.degree = degree
        self.reg_coef = reg_coef
        self.poly = PolynomialFeatures(self.degree, include_bias=False)
        self.weight = None

    def __str__(self) -> str:
        return f"RPC model:\ndegree:{self.degree},\nregularizing coefficient:{self.reg_coef}"

    def _set_dimension(self, X) -> None:
        self.dim = X.shape[1]

    def _set_indices(self) -> None:
        tmp_data = np.random.rand(1, self.dim)
        tmp_data = self.poly.transform(tmp_data)
        tmp_data = self._make_root_features(tmp_data)[0]
        indices = list(np.arange(self.dim))
        for i in range(self.dim, len(tmp_data)):
            if not (tmp_data[i] == 0).sum() == (self.dim - 1):
                indices.append(i)
        self.indices = indices

    def _make_root_features(self, Xp: np.ndarray) -> np.ndarray:
        """
        Transform PolynomialFeatures output using root polynomial approach
        from https://www.researchgate.net/publication/273462977_Color_Correction_Using_Root-Polynomial_Regression
        Arguments
        ---------
        Xp : ndarray
            input matrix transformed by PolynomialFeatures

        """
        count = 0

        # 1..k
        for curr_k in range(1, self.degree + 1):
            end_count = count + self._get_degree_count(curr_k)
            Xp[:, count:end_count] = Xp[:, count:end_count] ** (1 / curr_k)
            count = end_count
        return Xp

    def _get_degree_count(self, k: int):
        """
        How much number n features can produce elements with k degree
        Example: 1, a, b, a^2, ab, b^2
        for n = 2, k = 2 answer is 3 (a^2, ab, b^2)
        Arguments
        ---------
        n : number of features
        k : polynom degree

        Returns
        -------
        cnk: number of elements with k degree
        """

        def get_cnk(n, k):
            return factorial(n) / (factorial(k) * factorial(n - k))

        cnk = int(get_cnk(self.dim + k - 1, k))
        return cnk

    def _fit_model(
            self,
            X: np.ndarray, 
            Y: np.ndarray,
            reg_coef: float = 0.0
    ):
        n_s = X.shape[1]
        reg_ones = reg_coef * np.eye(n_s)
        X = np.concatenate([X, reg_ones])

        n_t = Y.shape[1]
        zeros = np.zeros([n_s, n_t])
        Y = np.concatenate([Y, zeros])

        sol = np.linalg.lstsq(X, Y, rcond=None)
        self.weight = sol[0]

    def fit(self, X, Y) -> None:
        Xp = self.poly.fit_transform(X)

        self._set_dimension(X)
        self._set_indices()

        Xrp = self._make_root_features(Xp)
        Xrp = Xrp[:, self.indices]
        self._fit_model(Xrp, Y, self.reg_coef)

    def predict(self, X) -> None:
        assert X.shape[1] == self.dim
        Xp = self.poly.fit_transform(X)
        Xrp = self._make_root_features(Xp)
        Xrp = Xrp[:, self.indices]
        Y = np.dot(Xrp, self.weight)
        return Y

    def save_model(self, path) -> None:
        ...

    @staticmethod
    def load_model() -> "RPC":
        return RPC()


# Deprecated

def calc_lin_coefs_reg(vals_mat_s, vals_mat_t,reg_coef=0):
    n_s = vals_mat_s.shape[1]
    reg_ones = reg_coef * np.eye(n_s)
    vals_mat_s = np.concatenate([vals_mat_s, reg_ones])

    n_t = vals_mat_t.shape[1]
    zeros = np.zeros([n_s, n_t])
    vals_mat_t = np.concatenate([vals_mat_t, zeros])

    sol = np.linalg.lstsq(vals_mat_s, vals_mat_t, rcond=None)
    coefs = sol[0]
    return coefs


def get_cnk(n, k):
    value = factorial(n) / (factorial(k) * factorial(n - k))
    return value


def get_degree_count(n, k):
    """
    How much number n features can produce elements with k degree
    Example: 1, a, b, a^2, ab, b^2
    for n = 2, k = 2 answer is 3 (a^2, ab, b^2)
    Arguments
    ---------
    n : number of features
    k : polynom degree

    Returns
    -------
    cnk: number of elements with k degree
    """
    cnk = int(get_cnk(n + k - 1, k))
    return cnk


def makeRootFeatures(Xp, n, k, bias):
    """
    Transform PolynomialFeatures output using root polynomial approach
    from https://www.researchgate.net/publication/273462977_Color_Correction_Using_Root-Polynomial_Regression
    Arguments
    ---------
    Xp : ndarray
         input matrix transformed by PolynomialFeatures
    n : int
        number of features
    bias : bool
           include bias or not
    k : int
        highest degree of Xp
    """
    count = 0
    if bias:
        count = 1

    # 1..k
    for curr_k in range(1, k + 1):
        end_count = count + get_degree_count(n, curr_k)
        Xp[:, count:end_count] = Xp[:, count:end_count] ** (1 / curr_k)
        count = end_count
    return Xp


def get_right_indices(power_arr, n, bias=False):
    if bias:
        N = n + 1
    else:
        N = n
    indices = list(np.arange(N))
    for i in range(N, len(power_arr)):
        if not (power_arr[i] == 0).sum() == (n - 1):
            indices.append(i)
    return indices


class RPCmodel:
    def __init__(self, degree, reg_coef, bias):
        self.degree = degree
        self.bias = bias
        self.reg_coef = reg_coef
        self.model = None
        self.indices = None

        self.poly = PolynomialFeatures(self.degree, include_bias=self.bias)

    def __str__(self):
        print(f"RPC: {self.degree}")

    def fit(self, X, Y):
        # create polynomial of self.degree
        n = len(X[0])
        print()
        Xp = self.poly.fit_transform(X)
        print(Xp.shape)
        # transform to root polynomial
        Xrp = makeRootFeatures(Xp, n, self.degree, bias=self.bias)
        # getting indices that should be excluded ex. (a, sqrt(a^2)), currently 'just works' method
        tmp_data = np.random.rand(1, n)
        tmp_data = self.poly.transform(tmp_data)
        tmp_data = makeRootFeatures(tmp_data, n, self.degree, bias=self.bias)
        indices = get_right_indices(tmp_data[0], n, bias=self.bias)
        print(tmp_data.shape)
        self.indices = indices
        Xrp = Xrp[:, self.indices]
        # OLS without additional intercept
        self.model = calc_lin_coefs_reg(Xrp, Y, reg_coef=self.reg_coef)

        return self

    def predict(self, X):
        n = len(X[0])
        k = self.degree
        Xp = self.poly.fit_transform(X)
        # transform to root polynomial
        Xrp = makeRootFeatures(Xp, n, k, bias=self.bias)
        Xrp = Xrp[:, self.indices]
        Y = np.dot(Xrp, self.model)
        return Y


class RPC_dep:
    def __init__(self, degree, reg_coef=0.0, bias=False):
        self.degree = degree
        self.bias = bias
        self.reg_coef = reg_coef
        self.model = RPCmodel(
            degree=self.degree, reg_coef=self.reg_coef, bias=self.bias
        )

    def __str__(self):
        bias_info = ""
        if self.bias:
            bias_info = " biased"
        return f"RPC: d:{self.degree}, r:{self.reg_coef}{bias_info}"

    def fit(self, df_x, df_y):
        self.columns = df_y.columns
        X = df_x.values
        print(X.shape)
        Y = df_y.values
        self.model.fit(X, Y)

        return self

    def predict(self, df_x):
        output = self.model.predict(df_x.values)
        answers = pd.DataFrame(output, index=df_x.index, columns=self.columns)
        return answers
