import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from plots.general import to_band_int

dir_main = "F:\\Masterarbeit\\THESIS\\general"
dir_plots = os.path.join(dir_main, "plots")
s2_offsets_pd = pd.read_csv(os.path.join(dir_main, "s2_time_offsets.csv"), sep=";")
rcParams["font.serif"] = "Times New Roman"
rcParams["font.family"] = "serif"

# plot inter-band angles
bands_a, bands_b = s2_offsets_pd["Band_a"], s2_offsets_pd["Band_b"]
time_offset = np.float32(s2_offsets_pd["Time_offset"])
bands_a_int = [to_band_int(band) for band in bands_a]
arg_sorted = np.argsort(time_offset)
time_offset_sorted = time_offset[arg_sorted]
bands_a_sorted, bands_b_sorted = np.array(bands_a)[arg_sorted], np.array(bands_b)[arg_sorted]
bands = list(bands_a_sorted + "/" + bands_b_sorted)
fig, ax = plt.subplots(figsize=(5, 3))
plt.subplots_adjust(bottom=0.15)
# plot relationship between angle and time offset
angles = {"B02": 0.022, "B03": 0.030, "B04": 0.034, "B08": 0.026, "B05": 0.038, "B06": 0.042, "B07": 0.046,
          "B8a": 0.051, "B01": 0.055, "B09": 0.059, "B10": 0.03, "B11": 0.04, "B12": 0.05}
angles = {key: angles[key] - 0.022 for key in angles.keys()}
offset_from_b02 = {"B02": 0, "B03":  0.527, "B04":  1.005, "B08":  0.264, "B05": 1.269, "B06":  1.468, "B07":  1.790,
                   "B8a":  2.055, "B01":  2.314, "B09":  2.586, "B10": 0.851, "B11":  1.468, "B12":  2.085}
colors = np.hstack([["blue", "green", "red", "#5e0e0e"], np.repeat("#4d4d4d", len(angles.values()) - 4)])
ax.scatter(angles.values(), offset_from_b02.values(), c=colors.tolist())
ax.set_ylabel("Time offset [s] relative to B02")
ax.set_xlabel("Parallax angle relative to B02")
off = 0.001
positions = [[angles["B02"] + off, offset_from_b02["B02"] + off],
             [angles["B03"] + off, offset_from_b02["B03"] + off],
             [angles["B04"] + off, offset_from_b02["B04"] + off],
             [angles["B08"] + off, offset_from_b02["B08"] + off],
             [angles["B05"] + off, offset_from_b02["B05"] + off],
             [angles["B06"] + off, offset_from_b02["B06"] + off * 2],
             [angles["B07"] + off, offset_from_b02["B07"] + off],
             [angles["B8a"] + off, offset_from_b02["B8a"] + off],
             [angles["B01"] + off, offset_from_b02["B01"] + off],
             [angles["B09"] - off * 4, offset_from_b02["B09"] + off],
             [angles["B10"] + off, offset_from_b02["B10"] + off],
             [angles["B11"] - off * 4, offset_from_b02["B11"] + off * 3],
             [angles["B12"] - off * 4, offset_from_b02["B12"] + off * 3]]
for angle, time, band, pos in zip(angles.values(), offset_from_b02.values(), angles.keys(), positions):
    ax.text(pos[0], pos[1], band)
fig.savefig(os.path.join(dir_plots, "s2_temporal_offsets_bands_points.png"), dpi=500)
plt.close()
