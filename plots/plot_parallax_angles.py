import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plots.general import to_band_int

dir_main = "F:\\Masterarbeit\\THESIS\\general"
dir_plots = os.path.join(dir_main, "plots")
s2_offsets_pd = pd.read_csv(os.path.join(dir_main, "s2_time_offsets.csv"), sep=";")

bands_a, bands_b = s2_offsets_pd["Band_a"], s2_offsets_pd["Band_b"]
time_offset = np.float32(s2_offsets_pd["Time_offset"])
bands_a_int = [to_band_int(band) for band in bands_a]
arg_sorted = np.argsort(time_offset)
time_offset_sorted = time_offset[arg_sorted]
bands_a_sorted, bands_b_sorted = np.array(bands_a)[arg_sorted], np.array(bands_b)[arg_sorted]
bands = list(bands_a_sorted + "/" + bands_b_sorted)
plt.scatter(time_offset_sorted, bands, color="black")
plt.subplots_adjust(bottom=0.2)
plt.xlabel("Time offset")
plt.ylabel("Band combination")
plt.title("Sentinel-2 temporal offsets [s] between bands")
plt.axes().xaxis.set_tick_params(labelsize=8)
plt.axes().yaxis.set_tick_params(labelsize=8)
plt.savefig(os.path.join(dir_plots, "s2_temporal_offsets_bands.png"), dpi=200)
