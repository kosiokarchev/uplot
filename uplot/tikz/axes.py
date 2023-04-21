import typing as tp

from ..corner.abstractcorner import AbstractAxis
from .bases import Optionable, Printable, PrintableEnvironment


class PGFAbstractAxis(Printable, Optionable, AbstractAxis):
    defaults = {**Optionable.defaults, **{
        'xmode': 'linear', 'ymode': 'linear'
    }}

    def set_xlim(self, xmin=None, xmax=None):
        self._set_options(xmin=xmin, xmax=xmax)

    def set_ylim(self, ymin=None, ymax=None):
        self._set_options(ymin=ymin, ymax=ymax)


class PGFAxis(PrintableEnvironment, PGFAbstractAxis):
    name = 'axis'


class PGFGroupplot(PrintableEnvironment):
    name = 'groupplot'
    children: tp.Sequence['PGFAxisInGroup']


class PGFAxisInGroup(PGFAbstractAxis):
    @property
    def header(self):
        return fr'\nextgroupplot{self.optional(self.formatted_options)}'
