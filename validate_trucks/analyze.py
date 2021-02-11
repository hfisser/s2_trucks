import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress

dir_validation = "F:\\Masterarbeit\\DLR\\project\\1_truck_detection\\validation"
validation_csv = os.path.join(dir_validation, "validation_run.csv")
validation = pd.read_csv(validation_csv)

s2_counts = np.float32(validation["s2_counts"])[:-1]
bast_truck = np.float32(validation["Lkw_R1"])[:-1] + np.float32(validation["Lkw_R2"])[:-1]
bast_lzg = np.float32(validation["Lzg_R1"])[:-1] + np.float32(validation["Lzg_R2"])[:-1]
bast_car = np.float32(validation["Pkw_R1"])[:-1] + np.float32(validation["Pkw_R2"])[:-1]

pearsons, regressions = [], []
bast = [bast_truck, bast_lzg, bast_car]
s2 = [s2_counts] * len(bast)

for idx in np.where(bast_lzg < 0)[0]:
    bast[1] = np.delete(bast[1], idx)
    s2[1] = np.delete(s2[1], idx)

for x, y in zip(s2, bast):
    pearsons.append(pearsonr(x, y))
    regressions.append(linregress(x, y))

x_labels = ["Sentinel-2 counts"] * 3
y_labels = ["BAST Lkw", "BAST Lzg", "BAST Pkw"]

for x_label, y_label, s2_count, bast_count in zip(x_labels, y_labels, s2, bast):
    plt.scatter(s2_counts, bast_truck)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("S2 detections vs. BAST")
    plt.savefig(os.path.join(dir_validation, "s2_vs_%s.png" % y_label.replace(" ", "_")))
