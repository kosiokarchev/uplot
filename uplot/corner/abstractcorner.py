import typing as tp
from abc import ABC, abstractmethod
from itertools import chain, count, repeat
from operator import itemgetter

import numpy as np
from frozendict import frozendict

from ..utils import _move_batch_mat, _move_batch_vec, _rowwise_combinations, _to_nd_obj_array


class AbstractAxis(ABC):
    @abstractmethod
    def set_xlim(self, xmin=None, xmax=None):
        raise NotImplementedError()

    @abstractmethod
    def set_ylim(self, ymin=None, ymax=None):
        raise NotImplementedError

    def autoscale(self, axis=None):
        pass


_AxisType = tp.TypeVar('_AxisType')


class AbstractCorner(ABC, tp.Generic[_AxisType]):
    def __init__(self, ndim: int = None, names: tp.Iterable[str] = None,
                 truths: tp.Union[tp.Iterable[float], tp.Mapping[str, float]] = None,
                 labels: tp.Union[tp.Iterable[str], tp.Mapping[str, str]] = None,
                 truth_options=frozendict(), label_options=frozendict(), axs: np.ndarray = None, **kwargs):
        super().__init__(**kwargs)

        self.ndim = ndim
        self.axs = axs
        if self.axs is not None:
            if isinstance(self.ndim, int):
                assert self.axs.shape == (ndim, ndim)
            else:
                self.ndim = self.axs.shape[-1]
        elif self.ndim is None:
            raise ValueError('At least one of axs or ndim must be not None.')

        self.il, self.jl = np.tril_indices(self.ndim, -1)
        self.idxl = np.stack((self.il, self.jl), 0)

        self.names = list(names
                          or (isinstance(truths, tp.Mapping) and truths.keys())
                          or (isinstance(labels, tp.Mapping) and labels.keys())
                          or labels or range(ndim))
        self.labels, self.truths = (
            {name: val for name, val in zip(self.names, (isinstance(p, tp.Mapping) and itemgetter(*self.names)(p)) or p or repeat(None))}
            for p in (labels, truths)
        )  # type: tp.Mapping[str, str], tp.Mapping[str, float]

        self.truth_options = truth_options
        self.label_options = label_options

    def _draw_truth_diag(self, ax: _AxisType, truth: float, **kwargs) -> None:
        raise NotImplementedError()

    def _draw_truth_offdiag(self, ax: _AxisType, truth_x: float, truth_y: float, **kwargs) -> None:
        raise NotImplementedError()

    def _draw_label_x(self, ax: _AxisType, label: str, **kwargs) -> None:
        raise NotImplementedError()

    def _draw_label_y(self, ax: _AxisType, label: str, **kwargs) -> None:
        raise NotImplementedError()

    def draw_truths(self, **kwargs):
        for (i, j), ax in self.enum_all:
            if i == j:
                self._draw_truth_diag(ax, self.truths[self.names[i]], **kwargs)
            else:
                self._draw_truth_offdiag(ax, self.truths[self.names[j]], self.truths[self.names[i]], **kwargs)

    def draw_labels(self, **kwargs):
        kwargs = {**self.label_options, **kwargs}
        for i, label in enumerate(self.labels.values()):
            if i > 0:
                self._draw_label_y(self.axs[i, 0], label, **kwargs)
            self._draw_label_x(self.axs[-1, i], label, **kwargs)

    @property
    def iter_diag(self) -> tp.Iterable[_AxisType]:
        yield from self.axs.diagonal()

    @property
    def enum_diag(self) -> tp.Iterator[tp.Tuple[int, _AxisType]]:
        return zip(count(), self.iter_diag)

    @property
    def enum_offdiag(self) -> tp.Iterable[tp.Tuple[tp.Tuple[int, int], _AxisType]]:
        yield from (((i, j), self.axs[i, j]) for i, j in _rowwise_combinations(self.ndim))

    @property
    def iter_offdiag(self) -> tp.Iterable[_AxisType]:
        yield from map(itemgetter(-1), self.enum_offdiag)

    @property
    def enum_all(self) -> tp.Iterable[tp.Tuple[tp.Tuple[int, int], _AxisType]]:
        yield from (((i, j), self.axs[i, j]) for i, j in zip(*np.tril_indices_from(self.axs)))

    @property
    def iter_all(self) -> tp.Iterable[_AxisType]:
        yield from map(itemgetter(-1), self.enum_all)


class AbstractGaussianCorner(AbstractCorner[_AxisType], ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.draw_hist = np.vectorize(self._draw_hist, excluded='self', otypes=(object,),)
        self.draw_contour = np.vectorize(self._draw_contour, excluded='self', otypes=(object,),)

    @staticmethod
    def _1d_iter(arr):
        return np.reshape(arr, (len(arr), -1))

    @abstractmethod
    def _draw_hist(self, m: float, v: float, _options=frozendict(), **options):
        raise NotImplementedError()

    @abstractmethod
    def _draw_contour(self, x: float, y: float, *args,
                      levels: tp.Iterable[float] = (1.,), level_kwargs: tp.Iterable[tp.Mapping[str, tp.Any]] = (frozendict(),),
                      _options=frozendict(), **options):
        raise NotImplementedError()

    def _get_cov_elements(self, cov):
        return cov[self.jl, self.jl], cov[self.il, self.il], cov[self.il, self.jl]

    @abstractmethod
    def _get_ellipse_args(self, cov):
        raise NotImplementedError()

    @abstractmethod
    def _draw(self, _drawing: tp.Iterable[tp.Tuple[_AxisType, tp.Any]]):
        raise NotImplementedError()

    def draw(self, mean: np.ndarray, cov: np.ndarray, lims=None,
             sigma_levels: tp.Tuple[float] = (3, 2, 1),
             hist1d_options=frozendict(), contour_options=frozendict(), _options: tp.Tuple[tp.Mapping] = ({},),
             contour_level_options: tp.Tuple[tp.Mapping] = None,
             batch_at_front=True, lims_nsigma=3., lims_pad_fraction=0.02,
             lims_collapse_functions: tp.Tuple[tp.Callable[[np.ndarray], np.ndarray], tp.Callable[[np.ndarray], np.ndarray]] = (np.min, np.max),
             **extra_options):
        if batch_at_front:
            mean, cov = _move_batch_vec(mean), _move_batch_mat(cov)

        if lims is None:
            lims = np.stack([
                f(a.reshape(len(a), -1), axis=-1)
                for f, a in zip(lims_collapse_functions,
                                np.moveaxis(mean[..., None] + np.sqrt(np.moveaxis(cov.diagonal(), -1, 0))[..., None] * (-1, 1) * lims_nsigma, -1, 0))
            ], -1)
            lims += np.diff(lims, axis=-1) * (-1, 1) * lims_pad_fraction

        nbatch = mean.ndim - 1
        sigma_levels, contour_level_options = (
            _to_nd_obj_array(np.array(arr), nbatch - 1)
            for arr in
            (sigma_levels, contour_level_options if contour_level_options is not None else len(sigma_levels) * ({},))
        )
        _options = _to_nd_obj_array(np.array(_options), nbatch)
        hist1d_options, contour_options, extra_options = (
            {key: _to_nd_obj_array(np.array(val), nbatch) for key, val in d.items()}
            for d in (hist1d_options, contour_options, extra_options))

        self._draw(zip(chain(self.iter_diag, self.iter_offdiag),
                       chain(self._1d_iter(self.draw_hist(mean, _move_batch_vec(np.diagonal(cov)), _options=_options, **{**extra_options, **hist1d_options})),
                             self._1d_iter(self.draw_contour(
                                 *mean[self.idxl[::-1]], *self._get_ellipse_args(cov),
                                 sigma_levels, contour_level_options, _options=_options, **{**extra_options, **contour_options})))))

        if lims is not False:
            for (i, j), ax in self.enum_all:
                ax.set_xlim(*lims[j])
                if i > j:
                    ax.set_ylim(*lims[i])
                else:
                    ax.autoscale(axis='y')
                    ax.set_ylim(0, None)

        return self
