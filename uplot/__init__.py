from frozendict import frozendict
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def imshow_with_cbar(ax, img, position='right', size='5%', pad=0.05, cbar_kwargs=frozendict(), **kwargs):
    im = ax.imshow(img, **kwargs)
    plt.colorbar(
        im, cax=make_axes_locatable(ax).append_axes(position, size=size, pad=pad),
        orientation='vertical' if position in ('left', 'right') else 'horizontal',
        **cbar_kwargs
    )
    return im
