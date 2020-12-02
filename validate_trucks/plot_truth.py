import matplotlib.pyplot as plt
import os
import pandas as pd

dir_ancillary = os.path.join("F:" + os.sep + "Masterarbeit", "DLR", "project", "1_truck_detection", "truth")
RGB_VECTORS = pd.read_csv(os.path.join(dir_ancillary, "rgb_vector_clusters.csv"), index_col=0)

plot_names = ["b02", "b03", "b04"]
for low, up, name in zip([0, 3, 6], [3, 6, 9], plot_names):
    for i in range(len(RGB_VECTORS)):
        plt.plot(["B02", "B03", "B04"], RGB_VECTORS.iloc[i][low:up])
    plt.ylabel("Reflectance")
    plt.savefig(os.path.join(os.path.dirname(dir_ancillary), "project", "figures", "rgb_at_" + name + ".png"))
plt.close()
