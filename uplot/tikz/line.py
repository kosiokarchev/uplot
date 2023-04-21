import typing as tp

import numpy as np
from frozendict import frozendict
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from uplot.tikz.axes import PGFAbstractAxis, PGFAxis
from uplot.tikz.commands import AddplotCoordinates
from uplot.tikz.utils import mpl_color_to_tikz, mpl_linestyle_to_tikz


class TikzLine2D(AddplotCoordinates):
    defaults = {**AddplotCoordinates.defaults, **{
        'solid': None
    }}

    # TODO: drawstyles: 'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'
    # TODO: label
    # TODO: marker
    def __init__(self, line2d: Line2D, *args, **kwargs):
        points = np.transpose(tuple(map(np.array, line2d.get_data())))
        coordsys = 'axis description cs' if line2d.get_transform() == line2d.axes.transAxes else None

        options = {**{
            'color': mpl_color_to_tikz(line2d.get_color()),
            **mpl_linestyle_to_tikz(line2d.get_linestyle()),
            'line width': f'{line2d.get_linewidth()}pt',
            'opacity': (_ := line2d.get_alpha()) is not None and _ or 1
        }, **kwargs.pop('options', {})}
        kwargs.setdefault('zorder', line2d.get_zorder())
        super().__init__(points, *args, coordsys=coordsys, options=options, **kwargs)

        self._line2d = line2d


# TODO: Automatic style extraction
# TODO: Legend
def TikzAxes(ax: Axes, cls: tp.Type[PGFAbstractAxis] = PGFAxis, *args,
             line2d_kwargs=frozendict(),
             **kwargs):
    options = {**{
        'xlabel': ax.get_xlabel(), 'ylabel': ax.get_ylabel(),
        'xmode': ax.get_xscale(), 'ymode': ax.get_yscale()
    }, **kwargs.pop('options', {})}
    ret = cls(*args, options=options, **kwargs)
    ret.children += [TikzLine2D(line2d, options=line2d_kwargs) for line2d in ax.get_lines()]
    return ret
