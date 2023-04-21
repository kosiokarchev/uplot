from typing import Tuple

from frozendict import frozendict
from matplotlib import pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


__all__ = 'imshow_with_cbar', 'traffic', 'midtraffic'

traffic = LinearSegmentedColormap.from_list('traffic', ('forestgreen', 'gold', 'firebrick'))
midtraffic = LinearSegmentedColormap.from_list('midtraffic', ((0, 'gold'), (0.5, 'forestgreen'), (0.75, 'orange'), (1, 'firebrick')))
midtraffic2 = LinearSegmentedColormap.from_list('midtraffic2', ((0, 'blue'), (0.5, 'forestgreen'), (0.75, 'orange'), (1, 'firebrick')))


def imshow_with_cbar(img, position='right', size='5%', pad=0.05, cbar_kwargs=frozendict(),
                     ax: plt.Axes = None, aspect='equal',
                     **kwargs) -> Tuple[plt.Axes, Colorbar]:
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(img, **kwargs)
    ax.set_aspect(aspect)
    cax = plt.colorbar(
        im, cax=make_axes_locatable(ax).append_axes(position, size=size, pad=pad),
        orientation='vertical' if position in ('left', 'right') else 'horizontal',
        **cbar_kwargs
    )
    if position == 'top':
        cax.ax.xaxis.tick_top()
        cax.ax.xaxis.set_label_position('top')

    return im, cax
