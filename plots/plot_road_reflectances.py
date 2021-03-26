import os
import numpy as np
import pandas as pd
import rasterio as rio
import seaborn as sns
import matplotlib.pyplot as plt

dir_plots = os.path.join("F:" + os.sep + "Masterarbeit", "THESIS", "general", "plots")
f = "F:\\Masterarbeit\\DLR\\project\\1_truck_detection\\data\\s2\\subsets\\S2B_MSIL2A_20200327T101629_N0214_R065_T32UNA_20200327T134849_y0_x0.tif"

with rio.open(f, "r") as src:
    data = np.zeros((src.count, src.height, src.width))
    for i in range(src.count):
        data[i] = src.read(i + 1)

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 5))
ax = sns.violinplot(y=data.flatten(),
                    x=np.float32([np.repeat(i, data.shape[1] * data.shape[2]) for i in range(data.shape[0])]).flatten())