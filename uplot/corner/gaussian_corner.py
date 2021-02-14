import os
import pickle
import typing as tp
from functools import partial
from math import pi
from operator import itemgetter

import numpy as np
import sympy as sym

from ..utils import _move_batch_mat, _rowwise_combinations, vfloat


def covariance_ellipse(c00, c11, c10):
    D = np.sqrt((c00-c11)**2 + 4*c10**2)
    scale = np.sqrt(((c00+c11) + np.moveaxis(np.array((1, -1)) * D[..., None], -1, 0)) / 2)
    angle = np.where(D, 0.5 * np.arcsin(2 * c10 / D), 0.)
    return tp.cast(tp.Tuple[vfloat, vfloat, vfloat],
                   (*scale, np.where(c00-c11 >= 0, angle, pi / 2 - angle)))


def gaussian_corner(cov: np.ndarray, batch_at_front=True) -> tp.Tuple[tp.Tuple[int, int], tp.Tuple[tp.Tuple[np.ndarray, np.ndarray], np.ndarray]]:
    if batch_at_front:
        cov = _move_batch_mat(cov)
    for i, j in _rowwise_combinations(len(cov)):
        yield (i, j), covariance_ellipse(cov[i, i], cov[j, j], cov[i, j])


_trellipse_callable = tp.Callable[[vfloat, vfloat, vfloat, vfloat], vfloat]
TRELLIPSE_FNAME = os.path.join(os.path.dirname(__file__), 'trellipse.pickle')


def get_trellipse(fname: str = TRELLIPSE_FNAME):
    data = pickle.load(open(fname, 'rb'))
    return tp.cast(tp.Tuple[_trellipse_callable, _trellipse_callable, _trellipse_callable],
                   tuple(map(partial(sym.lambdify, data['args']), itemgetter('ow', 'oh', 'oa')(data))))



if __name__ == '__main__' and True:
    w, h, t = sym.symbols('w, h, t', positive=True)
    theta = sym.Symbol('theta')
    TR = sym.Matrix(((1, 0), (0, t))) @ sym.rot_axis3(theta)[:2, :2]
    U, L = (TR @ sym.Matrix(((w**2, 0), (0, h**2))) @ TR.T).diagonalize(normalize=True)

    # use covariance matrix as input
    # ------------------------------
    a, b = sym.symbols('a, b', positive=True)
    c = sym.Symbol('c')
    T = sym.Matrix(((1, 0), (0, t)))
    U, L = (T @ sym.Matrix(((a, c), (c, b))) @ T.T).diagonalize(normalize=True)

    # args = (w, h, theta, t)
    ow, oh = sym.sqrt(L[0, 0]), sym.sqrt(L[1, 1]) / t
    oa = sym.atan2(sym.sign(theta) * U[1, 1], U[1, 0])
    #
    # pickle.dump({'args': args, 'ow': ow, 'oh': oh, 'oa': oa},
    #             open(TRELLIPSE_FNAME, 'wb'))
    #
    # TikzGaussianCorner._ow, TikzGaussianCorner._oh, TikzGaussianCorner._oa = get_trellipse()
