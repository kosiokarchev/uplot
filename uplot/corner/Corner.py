import typing as tp

import matplotlib
import numpy as np
import scipy.stats
from frozendict import frozendict
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.ticker import Formatter, Locator

from .abstractcorner import AbstractCorner, AbstractGaussianCorner
from .gaussian_corner import covariance_ellipse
from ..utils import unshare, vfloat


class Corner(AbstractCorner[plt.Axes]):
    fig: plt.Figure

    def __init__(self, ndim=None, names=None, truths=None, labels=None,
                 truth_options=frozendict(), label_options=frozendict(), axs=None,
                 diag_locator: tp.Union[tp.Type[Locator], Locator] = plt.AutoLocator,
                 diag_formatter: tp.Union[tp.Type[Formatter], Formatter] = plt.NullFormatter,
                 **subplots_kwargs):
        super().__init__(ndim=ndim, names=names, truths=truths, labels=labels,
                         truth_options=truth_options, label_options=label_options, axs=axs)

        if self.axs is None:
            self.fig, self.axs = plt.subplots(self.ndim, self.ndim, sharex='col', sharey='row', **subplots_kwargs)
        else:
            self.fig = self.axs[0, 0].figure

        for ax in self.axs[np.triu_indices(self.ndim, 1)]:
            ax.remove()
        for ax in self.iter_diag:
            unshare(ax, 'y', diag_locator, diag_formatter)
            ax.yaxis.set_visible(True)
            ax.set_ylim(0)

        self.draw_truths(**self.truth_options)
        self.draw_labels(**self.label_options)

    _draw_label_x = staticmethod(plt.Axes.set_xlabel)
    _draw_label_y = staticmethod(plt.Axes.set_ylabel)
    _draw_truth_diag = staticmethod(plt.Axes.axvline)

    @staticmethod
    def _draw_truth_offdiag(ax: plt.Axes, truth_x, truth_y, **kwargs):
        ax.axvline(truth_x, **kwargs)
        ax.axhline(truth_y, **kwargs)


class GaussianCorner(AbstractGaussianCorner[plt.Axes], Corner):
    @property
    def _x(self):
        return self.__x

    @_x.setter
    def _x(self, value):
        self.__x = value
        self.__y = self._get_y()

    @property
    def _y(self):
        if self.__y is None:
            self.__y = self._get_y()
        return self.__y

    @staticmethod
    def _get_y(x):
        return scipy.stats.norm.pdf(x)

    __x = np.linspace(-5, 5)
    __y = _get_y.__func__(__x)

    def _draw_hist(self, m, v, _options=frozendict(), **options):
        scale = np.sqrt(v)
        return plt.Line2D(m + scale * self._x, self._y / scale, **{**options, **_options})
    
    def _draw_contour(self, x, y, w=1., h=1., angle=0., levels=(1.,), level_kwargs=(frozendict(),), _options=frozendict(), **options):
        return PatchCollection([
            Ellipse((x, y), * 2*level*np.array((w, h)), np.rad2deg(angle), **{**options, **_options, **lkwargs})
            for level, lkwargs in zip(levels, level_kwargs)
        ], match_original=True)

    def _get_ellipse_args(self, cov):
        return covariance_ellipse(*self._get_cov_elements(cov))

    def _draw(self, _drawing):
        for ax, elements in _drawing:
            for element in elements:
                if isinstance(element, matplotlib.collections.Collection):
                    ax.add_collection(element)
                elif isinstance(element, Line2D):
                    ax.add_line(element)
