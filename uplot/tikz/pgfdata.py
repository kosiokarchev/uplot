from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import chain

import numpy as np
import torch
from astropy.table import Table

from .bases import Printable


_non_iterables = (str, np.ndarray, torch.Tensor)


def nested_iterables(o, keys=()):
    yield from (chain(*(
        nested_iterables(v, keys + (k,))
        for k, v in (o.items() if isinstance(o, Mapping) else enumerate(o))
    )) if isinstance(o, Iterable) and not isinstance(o, _non_iterables) else ((keys, o),))


def record_to_dict(struct):
    return (dict(zip(struct.dtype.names, (record_to_dict(s) for s in struct)))
            if (hasattr(struct, 'dtype') and struct.dtype.kind == 'V') else struct)


@dataclass
class BasePGFData:
    value: str
    keytype: str = 'initial'


class PGFData(BasePGFData):
    def __init__(self, *args):
        if isinstance(args[0], BasePGFData):
            args = args[0].value, args[0].keytype
        super().__init__(*args)


def to_pgfdata(data, namespace=None):
    children = [f'/{namespace}/.cd'] if namespace else []
    children.extend(f'{"/".join(map(str, keys))}/.{val.keytype}={{{val.value}}}'
                    for keys, val in nested_iterables(data)
                    for val in [PGFData(val)])
    return Printable(children=children, _header=r'\pgfkeys{', _footer=r'}', _joiner=',\n')


def table_to_pgfdata(t: Table, index=None, namespace=None):
    rowiter = ({key: record_to_dict(row[key]) for key in row.keys() if key != index} for row in t)
    return to_pgfdata(dict(zip(t['i'], rowiter)) if index else rowiter, namespace=namespace)
