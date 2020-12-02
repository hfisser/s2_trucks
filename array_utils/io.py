import rasterio as rio
import numpy as np


def rio_read_all_bands(file_path):
    with rio.open(file_path, "r") as src:
        meta = src.meta
        n_bands = src.count
        arr = np.zeros((src.count, src.height, src.width), dtype=np.float32)
        for i in range(n_bands):
            arr[i] = src.read(i+1).astype(np.float32)
    return arr, meta
