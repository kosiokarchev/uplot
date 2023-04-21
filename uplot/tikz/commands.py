import typing as tp

import numpy as np
from frozendict import frozendict

from .bases import Command, Optionable
from .utils import points_to_coords


__all__ = 'DrawCommand', 'AxhlineCommand', 'AxvlineCommand', 'EllipseCommand',\
          'AddplotCommand', 'AddplotExpression', 'AddplotCoordinates'



class Subcommand(Command):
    def __init__(self, name, command_body='', *args, **kwargs):
        self.name = name
        super().__init__(command_body, *args, **kwargs)

    @property
    def formatted_name(self):
        return self.name

    @property
    def end(self):
        return ''


class DrawCommand(Command):
    name = 'draw'


class AxvlineCommand(DrawCommand):
    def __init__(self, x, ymin=0., ymax=1., ydata=0., **kwargs):
        super().__init__(f'({{rel axis cs:0,{ymin}}} -| {{axis cs:{x},{ydata}}}) |- (rel axis cs:0,{ymax})', **kwargs)


class AxhlineCommand(DrawCommand):
    def __init__(self, y, xmin=0., xmax=1., xdata=0., **kwargs):
        super().__init__(f'({{rel axis cs:{xmin},0}} |- {{axis cs:{xdata},{y}}}) -| (rel axis cs:{xmax},0)', **kwargs)


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

    @property
    def formatted_command_body(self):
        return self.command_body


class AddplotCommand(Command):
    name = 'addplot'
    subcommand: str

    def __init__(self, command_body, subcommand=None, subcommand_options=frozendict(), *args, **kwargs):
        if subcommand is not None:
            self.subcommand = subcommand
        self.sub = Subcommand(self.subcommand, f'{{{command_body}}}', options=subcommand_options)
        super().__init__(*args, **kwargs)

    @property
    def formatted_command_body(self):
        return f' {self.sub}'


class AddplotExpression(AddplotCommand):
    subcommand = ''


class AddplotCoordinates(AddplotCommand):
    num_fmt = '.4e'
    subcommand = 'coordinates'

    def __init__(self, points: tp.Union[tp.Iterable[tp.Iterable[tp.Tuple[float, float]]], np.ndarray], coordsys=None, *args, **kwargs):
        super().__init__(command_body=points_to_coords(points, self.num_fmt, coordsys), *args, **kwargs)


class AddplotGraphics(AddplotCommand):
    subcommand = 'graphics'
