import numpy as np


def _to_numpy(arr):
    if hasattr(arr, "detach"):
        arr = arr.detach()
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    if hasattr(arr, "numpy"):
        return arr.numpy()
    return np.asarray(arr)
