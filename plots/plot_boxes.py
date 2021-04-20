import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from glob import glob

rcParams["font.serif"] = "Times New Roman"
rcParams["font.family"] = "serif"

dirs = dict(main="F:\\Masterarbeit\\DLR\\project\\1_truck_detection")
dirs["labels"] = os.path.join(dirs["main"], "data", "labels")
dirs["plots"] = os.path.join("F:" + os.sep + "Masterarbeit", "THESIS", "general", "plots")
dirs["truth"] = os.path.join(dirs["main"], "truth")
training_validation_tiles_pd = pd.read_csv(os.path.join(dirs["main"], "training", "tiles.csv"), sep=";")
truth_path_training = os.path.join(dirs["truth"], "spectra_ml_training_tiles.csv")
truth_path_validation = os.path.join(dirs["truth"], "spectra_ml_validation_tiles.csv")


def plot_box_size(tiles_pd):
    color = "#425b69"
    all_boxes = {}
    for column, n in zip(["training_tiles", "validation_tiles"], [250, 35]):
        all_boxes[column] = []
        countries = list(tiles_pd[column.replace("_tiles", "_countries")])
        for country in np.unique(countries):
            idxs = np.where(np.array(countries) == country)[0]  # maybe several of same country that should combined
            boxes = []
            for idx in idxs:
                tile = tiles_pd.loc[idx, column]
                try:
                    boxes.append(gpd.read_file(glob(os.path.join(dirs["labels"], "*%s*.gpkg" % tile))[0]))
                except IndexError:
                    continue
            try:
                boxes = pd.concat(boxes)
            except ValueError:
                continue
            boxes_subset = []
            np.random.seed(99)
            for idx in np.random.choice(list(range(len(boxes))), n, replace=False):
                boxes_subset.append(boxes.iloc[idx].geometry)
            all_boxes[column].append(gpd.GeoDataFrame({"geometry": boxes_subset}, crs=boxes.crs))
    hw = {}
    all_areas, all_widths, all_heights = [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for key, value, idx in zip(all_boxes.keys(), all_boxes.values(), range(2)):
        areas, widths, heights = [], [], []
        for boxes in value:
            for row in boxes.iterrows():
                areas.append(row[1].geometry.area)
                bounds = row[1].geometry.bounds
                widths.append(bounds[2] - bounds[0])
                heights.append(bounds[3] - bounds[1])
        # get number of pixels
        areas = np.float32(areas) / 100
        hw[key] = {"w": np.float32(widths) / 10, "h": np.float32(heights) / 10, "area": areas}
        all_areas.append(areas)
        all_widths.append(widths)
        all_heights.append(heights)
    idx = 0
    all_areas = np.hstack(all_areas)
    parts = axes[idx].violinplot(all_areas, showmeans=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(1)
        pc.set_facecolor(color)
    quartile1, medians, quartile3 = np.percentile(all_areas, [25, 50, 75])
    inds = np.arange(1, 2)
    axes[idx].scatter(inds, medians, marker="o", color="white", s=30, zorder=3)
    axes[idx].vlines(inds, quartile1, quartile3, color="k", linestyle="-", lw=5)
    axes[idx].set_title("%s (n=%s)" % ("Number of pixels/box", len(all_areas)))
    axes[idx].set_ylabel("Number of pixels")
#    plt.savefig(os.path.join(dirs["plots"], "training_validation_box_dimensions_violinplot.png"), dpi=500)
 #   plt.close()
    means = {}
    countries = []
 #   fix, ax = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.25)
    for idx, key in enumerate(["training_tiles", "validation_tiles"]):
        means[key] = []
        for boxes in all_boxes[key]:
            area = []
            for box in list(boxes.geometry):
                area.append(box.area / 100)
            means[key].append(np.mean(area))
        countries.append(list(tiles_pd[key.replace("_tiles", "_countries")].dropna().drop_duplicates()))
    mean_values = np.array(list(means.values())).flatten()
    argsorted = np.argsort(mean_values)[::-1]
    countries = np.array(countries).flatten()
    axes[1].bar(list(range(len(countries))), np.float32(mean_values)[argsorted], color=color)
    axes[1].set_ylabel("Number of pixels")
    axes[1].set_xticks(list(range(len(countries))))
    axes[1].set_xticklabels(np.array(countries)[argsorted], rotation=90)
    axes[1].set_ylim(0, 7.5)
    axes[1].set_title("Number of pixels/box by country")
    plt.savefig(os.path.join(dirs["plots"], "box_areas_summary.png"), dpi=500)
    plt.close()


def plot_truck_rgb(truth_training_csv, truth_validation_csv):
    truth_training_pd = pd.read_csv(truth_training_csv)
    truth_validation_pd = pd.read_csv(truth_validation_csv)
    truth_pd = pd.concat([truth_training_pd, truth_validation_pd])
    # get consecutive non-background records with red, green, blue
    rgb_reflectances = []
    for i in range(len(truth_pd)):
        if truth_pd.iloc[i].label in ["background", "green", "blue"]:  # look only from red
            continue
        else:
            red = truth_pd.iloc[i]
            green = truth_pd.iloc[i + 1]
            blue = truth_pd.iloc[i + 2]
            if green.label != "green":
                print("Not green")
                print(i)
            elif blue.label != "blue":
                print("Not blue")
                print(i)
            else:
                pass
            rgb = np.zeros(3)
            for class_idx, part in enumerate([red, green, blue]):
                rgb[class_idx] = np.float32(part[part.label])
            rgb_reflectances.append(rgb)
    rgb_np = np.float32(rgb_reflectances).reshape(3, 50, 57)  # not really needed anymore
    argmax = np.argmax(rgb_np.swapaxes(0, 2).swapaxes(0, 1), 2).flatten()
    idxs = []
    for i in range(rgb_np.shape[0]):
     #   argsort = np.in16(np.argsort(rgb_np[i, :, :].flatten())[::-1])
        idxs.append(np.argwhere(argmax == i).flatten())
    idxs = np.hstack(idxs)
    sorted = np.float32([rgb_np[0].flatten(), rgb_np[1].flatten(), rgb_np[2].flatten()])[:, np.int16(idxs)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(np.clip(sorted * 5, 0, 1).reshape(3, 30, 95).swapaxes(0, 2).swapaxes(0, 1))

    #np.random.shuffle(rgb_np)


def plot_truck_rgb_spectra(truth_training_csv, truth_validation_csv):
    truth_training_pd = pd.read_csv(truth_training_csv)
    truth_validation_pd = pd.read_csv(truth_validation_csv)
    truth_pd = pd.concat([truth_training_pd, truth_validation_pd])
    fig, axes = plt.subplots(4, 1, figsize=(5, 5))
    shape = axes.shape
    labels = ["red", "green", "blue", "background"]
    bands = ["red", "green", "blue", "nir"]
    colors = ["#FF0000", "#00FF00", "#0000FF", "#760909"]
    for y in range(shape[0]):
        positions = [0, 0.6, 1.2, 1.8]
        all_values = [np.float32(truth_pd[truth_pd.label == labels[y]][band]) for band in bands]
        mean_values = [np.mean(values) for values in all_values]
        std_values = [np.std(values) for values in all_values]
        parts = axes[y].violinplot(all_values, positions,
                                   showmeans=False, showextrema=False, showmedians=False)
        axes[y].plot(positions, mean_values, color="#000000", alpha=0.75)
        axes[y].scatter(positions, mean_values, color="#FFFFFF", alpha=0.75)
        axes[y].set_ylim(0, 0.5)
        for pc, color, position, mean_stat, std_stat in zip(parts["bodies"], colors, positions,
                                                            mean_values, std_values):
            pc.set_alpha(1)
            pc.set_facecolor(color)
            for line_idx, stat, stat_name in zip([0.4, 0.32], [mean_stat, std_stat], ["mean=%s", "std=%s"]):
                axes[y].text(position - 0.34, line_idx, stat_name % np.round(stat, 2), fontsize=10)
        axes[y].set_xticks([])
        axes[y].set_title("'%s'" % labels[y])
    plt.tight_layout()


if __name__ == "__main__":
    plot_box_size(training_validation_tiles_pd)
    plot_truck_rgb(truth_path_training, truth_path_validation)
    plot_truck_rgb_spectra(truth_path_training, truth_path_validation)

