from __future__ import annotations

from itertools import cycle
from typing import Iterable, Optional, Union

import numpy as np
from matplotlib import rcParams
from matplotlib.colors import to_rgb
from more_itertools import spy


rc = {
    'ls_num_fmt': '.1f',
    'clr_gray_num_fmt': '.3f'
}
_mpl_color_shorthands = {
    'b': 'blue', 'g': 'green', 'r': 'red', 'c': 'cyan',
    'm': 'magenta', 'y': 'yellow', 'k': 'black', 'w': 'white'}


def mpl_color_to_tikz(color) -> str:
    if isinstance(color, str) and color[0] == 'C' and (color := int(color[1:])):
        pc = rcParams["axes.prop_cycle"].by_key()
        color = pc[color % len(pc)]
    if isinstance(color, str):
        try:
            color = format(100-100*float(color), rc["clr_gray_num_fmt"]).split(".")
            return f'black!{color[0]}.{color[1].rstrip("0")}'
        except ValueError as e:
            if color[0] != '#':
                return _mpl_color_shorthands.get(color, color)
    return 'rgb,1:red,{};green,{};blue,{}'.format(*(format(val, rc['clr_gray_num_fmt']) for val in to_rgb(color)))


_mpl_linestyles = {
    '-': 'solid', '--': 'dashed', '-.': 'dashdotdotted', ':': 'dotted'
}


def mpl_linestyle_to_tikz(ls) -> dict[str, Optional[str]]:
    if ls in ('None', ' ', ''):
        return {'draw': None}
    if isinstance(ls, str):
        return {_mpl_linestyles.get(ls, ls): None}
    elif isinstance(ls, tuple) and len(ls) == 2:
        return {'dash phase': format(ls[0], rc['ls_num_fmt'])+'pt',
                'dash pattern': ' '.join(f'{format(s, rc["ls_num_fmt"])}pt {onoff}' for s, onoff in zip(ls[1], cycle(('on', 'off'))))}


def segs_to_coords(segs, *args, **kwargs):
    return r'\par'.join(points_to_coords(s, *args, **kwargs) for s in segs)


def points_to_coords(points: Union[Iterable[Iterable[tuple[float, float]]], np.ndarray], num_fmt=None, coordsys=None):
    num_fmt = num_fmt or '.6g'
    coordsys = f'{coordsys}:' if coordsys else ''

    return r'\par'.join(
        ' '.join(f'({coordsys}{", ".join(format(_p, num_fmt) for _p in p)})' for p in ch)
        for ch in ([points] if np.array(spy(points)[0][0]).ndim == 1 else points)
    )


def points3d_to_metacoords(points: Union[Iterable[Iterable[tuple[float, float, float]]], Iterable[tuple[float, float, float]], np.ndarray], num_fmt=None, coordsys=None):
    num_fmt = num_fmt or '.6g'
    coordsys = f'{coordsys}:' if coordsys else ''
    points = np.array(points)
    return r' \par '.join(
        ' '.join(f'({coordsys}{fa[0]}, {fa[1]}) [{fa[2]}]' for a in row
                 for fa in [[format(_p, num_fmt) for _p in a]])
        for row in (points if points.ndim == 3 else [points])
    )

