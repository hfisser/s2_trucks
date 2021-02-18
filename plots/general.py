import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def to_band_int(band_str):
    band_int = band_str[1:]
    its_band_8a = int(band_int[-1] == "a")
    band_int = int([band_int, band_int[:-1]][its_band_8a])
    return [band_int, band_int + 1][int(its_band_8a or int(band_int) > 8)]



