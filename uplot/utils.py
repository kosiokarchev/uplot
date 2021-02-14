import typing as tp
from functools import partial
from itertools import islice, product
from operator import attrgetter, is_not
from typing import Iterable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axis import Ticker
from matplotlib.ticker import Formatter, Locator
from more_itertools import consume


def unshare(ax: plt.Axes, axis: tp.Literal['x', 'y', 'xy'] = 'xy',
            locator: tp.Union[tp.Type[Locator], Locator] = plt.AutoLocator,
            formatter: tp.Union[tp.Type[Formatter], Formatter] = plt.NullFormatter):
    for xy in axis:
        grouper = getattr(ax, f'get_shared_{xy}_axes')()
        axis = getattr(ax, f'{xy}axis')
        old_locator = axis.major.locator
        if old_locator.axis is axis:
            consume(map(partial(old_locator.__setattr__, 'axis'),
                        map(attrgetter(f'{xy}axis'),
                            islice(filter(partial(is_not, ax), grouper.get_siblings(ax)), 1))))
        grouper.remove(ax)

        axis.major = Ticker()
        axis.set_major_locator(locator() if isinstance(locator, type) else locator)
        axis.set_major_formatter(formatter() if isinstance(formatter, type) else formatter)


def _rowwise_combinations(iterable_or_length: tp.Union[tp.Iterable, int], iterable_2: tp.Iterable = None, take_from: tp.Iterable = None, replacement=False, return_indices=False):
    if not isinstance(iterable_or_length, Iterable):
        iterable_or_length = range(iterable_or_length)
        aux = lambda iterable: ((el, el) for el in iterable)
    else:
        aux = enumerate

    if iterable_2 is None:
        iterable_2 = iterable_or_length

    itake_from = iter(take_from) if take_from is not None else None
    for (i, item_i), (j, item_j) in product(aux(iterable_or_length), aux(iterable_2)):
        res = next(itake_from) if itake_from is not None else (item_i, item_j)
        if j > i or (j == i and not replacement):
            continue
        yield ((i, j), res) if return_indices else res


vfloat = tp.Union[float, np.ndarray]
vstr = tp.Union[str, np.ndarray]
_move_batch_vec = partial(np.moveaxis, source=-1, destination=0)
_move_batch_mat = partial(np.moveaxis, source=(-2, -1), destination=(0, 1))


def _to_nd_obj_array(obj: np.ndarray, ndim=-1) -> np.ndarray:
    out = np.empty(int(np.prod(obj.shape[:ndim])), dtype=object)
    out[:] = list(obj.reshape(-1, *obj.shape[ndim:]))
    return np.atleast_1d(out.reshape(obj.shape[:ndim]))
