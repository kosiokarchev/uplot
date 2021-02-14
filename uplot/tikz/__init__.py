import typing as tp
from abc import ABC
from functools import partial
from itertools import chain
from operator import is_not

import numpy as np
from frozendict import frozendict

from ..corner.abstractcorner import AbstractAxis, AbstractCorner, AbstractGaussianCorner
from ..utils import vfloat


class Orderable:
    def __init__(self, *args, zorder=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.zorder = zorder


class Printable(Orderable):
    def __init__(self, *args, zorder=0, **kwargs):
        super().__init__(*args, zorder=zorder, **kwargs)
        self.children: tp.Sequence[tp.Union[Printable, Command, str]] = []

    @property
    def header(self) -> tp.Optional[str]:
        return None

    @property
    def footer(self) -> tp.Optional[str]:
        return None

    @staticmethod
    def _sort_key(printable_or_string):
        return printable_or_string.zorder if isinstance(printable_or_string, Orderable) else 0

    def print(self) -> tp.Iterable[str]:
        return filter(partial(is_not, None), chain(
            (self.header,),
            ('\t' + c
             for ch in sorted(self.children, key=self._sort_key)
             for c in (ch.print() if isinstance(ch, Printable) else str(ch).split('\n'))),
            (self.footer,)
        ))

    def __str__(self):
        return '\n'.join(self.print())


class Optionable:
    _OptionsType = tp.Union[tp.MutableMapping[str, tp.Union[str, tp.Any]], tp.Iterable[str]]

    def __init__(self, *args, options: _OptionsType = None, additional_options='', **kwargs):
        super().__init__(*args, **kwargs)

        self.options: Optionable._OptionsType = (
            options if isinstance(options, tp.Mapping)
            else {key: None for key in options} if options else {}
        )
        self.additional_options = additional_options

    def _set_options(self, delete_if_None=True, **kwargs):
        for key, val in kwargs.items():
            if val is None:
                if delete_if_None:
                    self.options.pop(key, None)
            else:
                self.options[key] = val

    @staticmethod
    def optional(arg: str):
        return arg and f'[{arg}]'

    @staticmethod
    def format_options(options: str = None, **kwoptions: str):
        return ', '.join(filter(bool, (', '.join(f'{key}={{{val}}}' if val is not None else key
                                                 for key, val in kwoptions.items()),
                                       options)))

    @property
    def formatted_options(self):
        return self.format_options(self.additional_options, **self.options)


class Command(Optionable, Orderable, ABC):
    name: str

    def __init__(self, command_body, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.command_body = command_body

    def __str__(self):
        return '\\' + self.name + self.optional(self.formatted_options) + self.command_body + ';'


class DrawCommand(Command):
    name = 'draw'


class AxvlineCommand(DrawCommand):
    def __init__(self, x, ymin=0., ymax=1., ydata=0., **kwargs):
        super().__init__(f' ({{rel axis cs:0,{ymin}}} -| {{axis cs:{x},{ydata}}}) |- (rel axis cs:0,{ymax})', **kwargs)


class AxhlineCommand(DrawCommand):
    def __init__(self, y, xmin=0., xmax=1., xdata=0., **kwargs):
        super().__init__(f' ({{rel axis cs:{xmin},0}} |- {{axis cs:{xdata},{y}}}) -| (rel axis cs:{xmax},0)', **kwargs)


class EllipseCommand(Command):
    name = 'plotellipseabc'

    _LevelKwargsType = tp.Iterable[tp.Union[str, Optionable._OptionsType]]

    def __init__(self, x: str, y: str, a: str, b: str, c: str,
                 levels: tp.Iterable[float] = (), level_kwargs: _LevelKwargsType = (),
                 *args, **kwargs):
        levels, level_args = map(list, (levels, level_kwargs))
        super().__init__(
            f'({x}, {y})'
            + ''.join(f'{{{_}}}' for _ in (a, b, c))
            + self.optional(', '.join(map(str, levels)))
            + self.optional(', '.join(f'{{{{{arg if isinstance(arg, str) else self.format_options(**arg)}}}}}' for arg in level_args)),
            *args, **kwargs)


class AddplotCommand(Command, ABC):
    name = 'addplot'


class AddplotExpression(AddplotCommand):
    def __init__(self, expression: str, *args, **kwargs):
        super().__init__(f' {{{expression}}}', *args, **kwargs)


# TODO: This has so much potential!
class AddplotCoordinates(AddplotCommand):
    def __init__(self, points: tp.Union[tp.Iterable[tp.Iterable[tp.Tuple[float, float]]], np.ndarray], coordsys=None, *args, **kwargs):
        coordsys = f'{coordsys}:' if coordsys else ''
        super().__init__(' coordinates {{{}}}'.format(
            '\n'.join(' '.join(f'({coordsys}{p[0]}, {p[0]})' for p in ch) for ch in np.atleast_3d(points))),
            *args, **kwargs)


class PGFAbstractAxis(Optionable, AbstractAxis):
    def set_xlim(self, xmin=None, xmax=None):
        self._set_options(xmin=xmin, xmax=xmax)

    def set_ylim(self, ymin=None, ymax=None):
        self._set_options(ymin=ymin, ymax=ymax)


class PrintableEnvironment(Optionable, Printable, ABC):
    name: str = None

    @property
    def header(self):
        return rf'\begin{{{self.name}}}{self.optional(self.formatted_options)}'

    @property
    def footer(self):
        return rf'\end{{{self.name}}}'


class PGFAxis(PrintableEnvironment, PGFAbstractAxis):
    name = 'axis'


class PGFGroupplot(PrintableEnvironment):
    name = 'groupplot'
    children: tp.Sequence['PGFAxisInGroup']


class PGFAxisInGroup(PGFAbstractAxis, Printable):
    @property
    def header(self):
        return fr'\nextgroupplot{self.optional(self.formatted_options)}'


class PGFCorner(AbstractCorner[PGFAxisInGroup], PGFGroupplot):
    @staticmethod
    def _draw_label_x(ax, label):
        ax.options['xlabel'] = label

    @staticmethod
    def _draw_label_y(ax, label):
        ax.options['ylabel'] = label

    def _draw_truth_diag(self, ax, truth, **kwargs):
        ax.children.append(AxvlineCommand(truth, options={**self.truth_options, **kwargs}))

    def _draw_truth_offdiag(self, ax, truth_x, truth_y, **kwargs):
        ax.children.extend((AxvlineCommand(truth_x, options={**self.truth_options, **kwargs}),
                            AxhlineCommand(truth_y, options={**self.truth_options, **kwargs})))


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
        self.children = self.axs.flat

        self.options['tight layout'] = ndim
        self.options['corner'] = None
        self.options['group style'] = f'group size={ndim} by {ndim}'

        self.draw_truths()
        self.draw_labels()


class PGFGaussianCorner(AbstractGaussianCorner[PGFAxisInGroup], PGFCorner):
    num_fmt = '{:.4e}'

    def _num_fmt(self, num):
        return self.num_fmt.format(num)

    def _draw_hist(self, m, v, **kwargs):
        return AddplotExpression(f'exp(-(x-{self._num_fmt(m)})^2 / 2 / {self._num_fmt(v)}) / {self._num_fmt(v**0.5)}',
                                 options=kwargs)

    def _draw_contour(self, x, y, a=1., b=1., c=0.,
                      levels=(1,), level_kwargs: EllipseCommand._LevelKwargsType = (frozendict(),),
                      zorder=0, **kwargs: Optionable._OptionsType):
        return EllipseCommand(*map(self._num_fmt, (x, y, a, b, c)),
                              levels=levels, level_kwargs=level_kwargs, options=kwargs, zorder=zorder)

    _get_ellipse_args = AbstractGaussianCorner._get_cov_elements

    def _draw(self, _drawing):
        for ax, commands in _drawing:
            ax.children.extend(commands)
