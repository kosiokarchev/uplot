import numpy as np


class MetaArray(np.ndarray):
    def __new__(cls, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, meta=None):
        obj = super().__new__(cls, shape, dtype, buffer, offset, strides, order)
        obj.meta = {} if meta is None else meta
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.metadata = getattr(obj, 'metadata', {})

    def __reduce__(self):
        reduced = super().__reduce__()
        return reduced[:2] + (reduced[2] + (self.meta,),)

    def __setstate__(self, state, **kwargs):
        self.meta = state[-1]  # Set the info attribute
        super().__setstate__(state[:-1])
