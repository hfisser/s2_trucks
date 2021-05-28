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
training_validation_tiles_pd = pd.read_csv(os.path.join(dirs["main"], "training", "tiles.csv"))
truth_path_training = os.path.join(dirs["truth"], "spectra_ml_training_tiles.csv")
truth_path_validation = os.path.join(dirs["truth"], "spectra_ml_validation_tiles.csv")


def plot_box_size(tiles_pd):
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
    fig, ax = plt.subplots(figsize=(2, 1.5))
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
#    idx = 0
    all_areas = np.hstack(all_areas)
    parts = ax.violinplot(all_areas, showmeans=False, showextrema=False)
    parts["bodies"][0].set_alpha(1)
    parts["bodies"][0].set_facecolor("#ffe201")
    parts["bodies"][0].set_edgecolor("#363636")
    ax.scatter([1], np.mean(all_areas), marker="o", color="white", s=30, zorder=3)
    ax.vlines([1], np.percentile(all_areas, [25])[0], np.percentile(all_areas, [75])[0],
              color="k", linestyle="-", lw=5)
    ax.vlines([1], np.min(all_areas), np.max(all_areas), color="k", linestyle="-", lw=1)
    ax.set_ylabel("Number of pixels")
    ax.set_xticks([])
    ax.set_xticklabels([])
#    plt.savefig(os.path.join(dirs["plots"], "training_validation_box_dimensions_violinplot.png"), dpi=500)
 #   plt.close()
    means, countries = {}, []
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
#    ax.bar(list(range(len(countries))), np.float32(mean_values)[argsorted], color=color)
 #   ax.set_ylabel("Number of pixels")
  #  ax.set_xticks(list(range(len(countries))))
   # ax.set_xticklabels(np.array(countries)[argsorted], rotation=90)
    #ax.set_ylim(0, 7.5)
    #ax.set_title("Number of pixels/box by country")
    plt.tight_layout()
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
    #fig, axes = plt.subplots(4, 1, figsize=(5, 5))
    fig, axes = plt.subplots(1, 4, figsize=(9, 2))
    labels = ["red", "green", "blue", "background"]
    colors = ["#FF0000", "#00FF00", "#0000FF", "#000000"]
    bands = ["B02", "B03", "B04", "B08"]
    vis_std = []
    vis_nir_std = []
    for ax, label, color in zip(axes.flatten(), labels, colors):
        values = np.float32([truth_pd[truth_pd.label == label][b] for b in ["blue", "green", "red", "nir"]])
        mean_values, std_values = values.mean(1), values.std(1)
        vis_std.append(mean_values[0:3].std())
        vis_nir_std.append(mean_values.std())
        ax.plot(bands, mean_values, color=color, linewidth=2)
        ax.fill_between(bands, mean_values - std_values, mean_values + std_values, color=color, alpha=0.3)
#        ax.plot(bands, mean_values - std_values, color=color, linewidth=0.5)
 #       ax.plot(bands, mean_values + std_values, color=color, linewidth=0.5)
        ax.scatter(bands, values.mean(1), color=color, alpha=0.5, s=20)
        ax.set_ylim(0, 0.3)
        ax.margins(x=0)
        ax.set_title("'%s'" % label)
        if ax == axes[0]:
            ax.set_ylabel("Reflectance")
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["plots"], "label_band_spectra.png"), dpi=500)
    plt.close()

    """
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
    """


if __name__ == "__main__":
    plot_box_size(training_validation_tiles_pd)
   # plot_truck_rgb(truth_path_training, truth_path_validation)
    plot_truck_rgb_spectra(truth_path_training, truth_path_validation)

