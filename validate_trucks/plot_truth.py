import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

dir_ancillary = os.path.join("F:" + os.sep + "Masterarbeit", "DLR", "project", "1_truck_detection", "truth")
dir_figures = os.path.join(os.path.dirname(dir_ancillary), "figures")
RGB_VECTORS = pd.read_csv(os.path.join(dir_ancillary, "rgb_vector_clusters.csv"), index_col=0)


elements = ["boxes", "whiskers", "fliers", "means", "medians", "caps"]
# plot RGB clusters

# ratios
x_labels = [["B02-B03", " | B02-B04"], ["B03-B02", " | B03-B04"], ["B04-B02", " | B04-B03"]]
plt.figure(figsize=(6, 3))
line_color, fill_color = "#000000", "#8a8870"
for idx, position, x_label in zip([0, 2, 4], [0, 1.5, 3], x_labels):
    data = [[], []]
    for i in range(len(RGB_VECTORS)):
        data[0].append(np.array(RGB_VECTORS.iloc[i][idx]))
        data[1].append(np.array(RGB_VECTORS.iloc[i][idx + 1]))
        #plt.plot(["B02", "B03", "B04"], RGB_VECTORS.iloc[i][low:up])
    for data_idx, offset in zip(range(len(data)), [0, 0.6]):
        bp = plt.boxplot(np.array(data[data_idx]).flatten(), positions=[position + offset], labels=[x_label[data_idx]],
                         patch_artist=True)
        for element in elements:
            plt.setp(bp[element], color=line_color)
        for patch in bp["boxes"]:
            patch.set(facecolor=fill_color)
    plt.ylabel("Ratio value")
plt.tight_layout()
plt.savefig(os.path.join(dir_figures, "rgb_ratios.png"))
plt.close()

# reflectances
plt.figure(figsize=(6, 3))
x_labels = ["B02", "B03", "B04"]
colors = ["#0000ff", "#00ff00", "#ff0000"]
for idx, position in zip([[12, 16, 19], [13, 15, 20], [14, 17, 18]], [0, 1.5, 3]):
    data = [[], [], []]
    for i in range(len(RGB_VECTORS)):
        data[0].append(np.array(RGB_VECTORS.iloc[i][idx[0]]))
        data[1].append(np.array(RGB_VECTORS.iloc[i][idx[1]]))
        data[2].append(np.array(RGB_VECTORS.iloc[i][idx[2]]))
    off = 0.25
    for data_idx, offset in zip(range(len(data)), [0, off, off * 2]):
        bp = plt.boxplot(np.array(data[data_idx]).flatten(), positions=[position + offset], labels=[x_labels[data_idx]],
                         patch_artist=True)
        for element in elements:
            plt.setp(bp[element], color=line_color)
        for patch in bp["boxes"]:
            patch.set(facecolor=colors[data_idx])
plt.ylabel("Reflectance")
plt.tight_layout()
plt.savefig(os.path.join(dir_figures, "rgb_reflectances.png"))
plt.close()

# std
plt.figure(figsize=(6, 3))
x_labels = ["blue max", "green max", "red max"]
x_labels_prefix = ["rgb std\n", "spatial std\n"]
idx_offset = 3
colors = ["#0000ff", "#00ff00", "#ff0000"]
for i, idx, position in zip(range(len(x_labels)), [6, 7, 8], [0, 1.5, 3]):
    data = [[], []]
    for j in range(len(RGB_VECTORS)):
        data[0].append(np.array(RGB_VECTORS.iloc[j][idx]) / 10)  # was multiplied by 10 before
        data[1].append(np.array(RGB_VECTORS.iloc[j][idx + idx_offset]) / 10)
    off = 0.75
    for data_idx, offset in zip(range(len(data)), [0, off]):
        bp = plt.boxplot(np.array(data[data_idx]).flatten(), positions=[position + offset],
                         labels=[x_labels_prefix[data_idx] + x_labels[i]], patch_artist=True)
        for element in elements:
            plt.setp(bp[element], color=line_color)
        for patch in bp["boxes"]:
            patch.set(facecolor=colors[i])
plt.ylabel("Value")
plt.tight_layout()
plt.savefig(os.path.join(dir_figures, "rgb_std.png"))
plt.close()
