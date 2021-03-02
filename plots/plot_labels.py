import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dirs = dict(main="F:\\Masterarbeit\\DLR\\project\\1_truck_detection")
dirs["training"] = os.path.join(dirs["main"], "training")
dirs["plots"] = os.path.join("F:" + os.sep + "Masterarbeit", "THESIS", "general", "plots")

tiles_pd = pd.read_csv(os.path.join(dirs["training"], "tiles.csv"), sep=";")


def plot_labels(tiles, column1, column2, some_colors):
    tiles_no_na = tiles.dropna()
    fig, ax = plt.subplots()
    areas = np.array(tiles_no_na["Area"])
    keys = np.unique(areas)
    colors = [some_colors[np.where(keys == area)[0][0]] for area in areas]
    for country, n, color, label in zip(tiles_no_na[column1], tiles_no_na[column2], colors, areas):
        ax.bar(country, n, width=0.5, color=color, label=label)
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(right=0.75)
    plt.xticks(rotation=90)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    handles, labels = ax.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)
    ax.legend(handle_list, label_list, loc="center right", fontsize=8, bbox_to_anchor=(1.35, 0.5))
    name = country_column.split("_")[0]
    name = name[0].upper() + name[1:]
    plt.title(name + " regions", fontsize=12)
    plt.savefig(os.path.join(dirs["plots"], name + "_labels_by_area_barplot.png"), dpi=200)


if __name__ == "__main__":
    color_palette = ["#a4f4a6", "#50d4b6",  "#304674", "#56083b",  "#df3333", "#58a7c5"]
    np.random.shuffle(color_palette)
    tiles_pd.loc[tiles_pd["Area"] == "Australia", ["Area"]] = "Oceania"
    for country_column, count_column in zip(["training_countries", "validation_countries"],
                                            ["n_retain", "n_retain_validation"]):
        plot_labels(tiles_pd, country_column, count_column, color_palette)
