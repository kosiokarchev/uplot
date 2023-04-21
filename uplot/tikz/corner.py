import numpy as np
from frozendict import frozendict

from .axes import PGFAxisInGroup, PGFGroupplot
from .bases import Optionable
from .commands import AddplotExpression, AxhlineCommand, AxvlineCommand, EllipseCommand
from ..corner.abstractcorner import AbstractCorner, AbstractGaussianCorner


__all__ = 'PGFCorner', 'PGFGaussianCorner'


class PGFCorner(AbstractCorner[PGFAxisInGroup], PGFGroupplot):
    @staticmethod
    def _draw_label_x(ax, label, **kwargs):
        ax.options['xlabel'] = label

    @staticmethod
    def _draw_label_y(ax, label, **kwargs):
        ax.options['ylabel'] = label

    def _draw_truth_diag(self, ax, truth, **kwargs):
        ax.children.append(AxvlineCommand(truth, options={**self.truth_options, **kwargs}, zorder=100))

    def _draw_truth_offdiag(self, ax, truth_x, truth_y, **kwargs):
        ax.children.extend((AxvlineCommand(truth_x, options={**self.truth_options, **kwargs}, zorder=100),
                            AxhlineCommand(truth_y, options={**self.truth_options, **kwargs}, zorder=100)))

    def __init__(self, ndim=None, names=None, truths=None, labels=None,
                 truth_options=frozendict(), axs=None,
                 diagplot_args='', offdiag_args='',
                 zorder=0, **kwargs):
        super().__init__(ndim=ndim, names=names, truths=truths, labels=labels,
                         truth_options=truth_options, axs=axs, zorder=zorder, **kwargs)

        if self.axs is None:
            self.axs = np.reshape([PGFAxisInGroup() for _ in range(self.ndim**2)], (self.ndim, self.ndim))

        for (i, j), ax in np.ndenumerate(self.axs):
            ax.additional_options = ((i == j) and diagplot_args or
                                     (i < j) and 'group/empty plot' or
                                     (i > j) and offdiag_args or '')
        for ax in self.iter_diag:
            ax.options['ymajorticks'] = 'false'
        self.children = self.axs.reshape(-1)

        self.options['tight layout'] = ndim
        self.options['corner'] = None
        self.options['group style'] = f'group size={ndim} by {ndim}'

        self.draw_truths()
        self.draw_labels()


class PGFGaussianCorner(AbstractGaussianCorner[PGFAxisInGroup], PGFCorner):
    num_fmt = '{:.4e}'

    def _num_fmt(self, num):
        return self.num_fmt.format(num)

    def _draw_hist(self, m, v, _options=frozendict(), **options):
        return AddplotExpression(f'exp(-(x-{self._num_fmt(m)})^2 / 2 / {self._num_fmt(v)}) / {self._num_fmt(v**0.5)}',
                                 options={**options, **_options})

    def _draw_contour(self, x, y, a=1., b=1., c=0.,
                      levels=(1,), level_kwargs: EllipseCommand._LevelKwargsType = (frozendict(),),
                      _options=frozendict(), zorder=0, **options: Optionable._OptionsType):
        return EllipseCommand(*map(self._num_fmt, (x, y, a, b, c)),
                              levels=levels, level_kwargs=level_kwargs, options={**options, **_options}, zorder=zorder)

    _get_ellipse_args = AbstractGaussianCorner._get_cov_elements

    def _draw(self, _drawing):
        for ax, commands in _drawing:
            ax.children.extend(commands)
