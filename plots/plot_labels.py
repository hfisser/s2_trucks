import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

dirs = dict(main="F:\\Masterarbeit\\DLR\\project\\1_truck_detection")
dirs["training"] = os.path.join(dirs["main"], "training")
dirs["plots"] = os.path.join("F:" + os.sep + "Masterarbeit", "THESIS", "general", "plots")
tiles_pd = pd.read_csv(os.path.join(dirs["training"], "tiles.csv"), sep=";")
truth_pd = pd.read_csv(os.path.join(dirs["main"], "truth", "spectra_ml.csv"))


COLORS = ["#2e2e2e", "#0000ff", "#00ff00", "#ff0000"]
LABEL_INTS, LABELS = [1, 2, 3, 4], ["background", "blue", "green", "red"]


def plot_labels_relative(tiles, area_columns):
    fig, ax = plt.subplots(figsize=(10, 3))
    pos, off, labels_sum = 0, 0.12, []
    colors = {"Europe": "#cbd5e8", "Africa": "#b3e2cd", "East Asia": "#fdcdac", "North America": "#fff2ae",
              "Oceania": "#e6f5c9", "South America": "#f4cae4"}
    continents_sorted = None
    for n_label_column, c, area_column in zip(["n_retain_validation", "n_retain"], ["Greens", "Purples"], area_columns):
        unique_areas = np.unique(tiles[area_column].dropna())
        n_labels = np.int16(tiles[n_label_column])
        n_sum = np.nansum(np.int16(tiles[n_label_column].dropna()))
        labels_sum.append(n_sum)
        shares = [np.sum(n_labels[np.where(tiles[area_column] == a)[0]]) / n_sum * 100 for a in unique_areas]
        continents = [np.array(tiles["training_areas"])[np.where(tiles[area_column] == a)[0]][0] for a in unique_areas]
        left = 0
        argsorted = np.argsort(continents)[::-1]
        continents_sorted = np.array(continents)[argsorted]
        for share, area, continent in zip(np.array(shares)[argsorted], unique_areas[argsorted], continents_sorted):
            share_rounded = share - (left + share - 100) if (left + share) > 100 else share
            plt.barh(pos, share_rounded, height=0.1, color=colors[continent], left=left, edgecolor="black",
                     label=continent)
            text = ax.text(left + share_rounded * 0.5, pos, area.replace(" ", "\n"), ha="center", va="center",
                           color="black", fontsize=8)
            left += share
        pos += off
    ax.set_yticks([0, off])
    ax.set_yticklabels(["Validation\nn=%s\nshare=%s%%" % (labels_sum[0],
                                                          np.round(labels_sum[0] / np.sum(labels_sum) * 100, 2)),
                        "Training\nn=%s\nshare=%s%%" % (labels_sum[1],
                                                        np.round(labels_sum[1] / np.sum(labels_sum) * 100, 2))])
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.legend(continents_sorted[::-1])
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.subplots_adjust(right=0.9)  # for legend
    plt.legend(by_label.values(), by_label.keys(), loc="center right", bbox_to_anchor=(1.2, 0.5),
               fontsize=8)
    #plt.legend(by_label.values(), by_label.keys(), loc="center right", bbox_to_anchor=(1.25, 0.5))
    plt.xlim(0, 100)
    plt.xlabel("Share of labels [%]", fontsize=8)
    plt.title("Labels by country", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["plots"], "training_validation_%s.png" % area_columns[0]), dpi=300)
    plt.close()


def plot_label_data(truth):
    red = truth[truth["label"] == "red"]
    green = truth[truth["label"] == "green"]
    blue = truth[truth["label"] == "blue"]
    background = truth[truth["label"] == "background"]
    plot_label_distribution(truth)


def plot_label_distribution(truth):
    fig, ax = plt.subplots(figsize=(8, 1))
    left = 0
    counts = [np.count_nonzero(truth["label_int"] == label_int) for label_int in LABEL_INTS]
    for count, label, label_int in zip(counts, LABELS, LABEL_INTS):
        plt.barh(0, count, height=0.1, color=COLORS[label_int - 1], left=left, edgecolor="black", label=label)
        text = ax.text(left + count * 0.5, 0, label + "\nn=" % count, ha="center", va="center",
                       color="w", fontsize=10)
        left += count
    ax.set_yticks([0])
    ax.set_yticklabels(["Number of\n labels\nn=%s" % sum(counts)])
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    plt.xlim(0, left)
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["plots"], "label_distribution_barplot.png"), dpi=300)
    plt.close()


def plot_label_stats(truth):
    sns.set_theme(style="whitegrid")
    variable_names = ["red_normalized", "green_normalized", "blue_normalized"]
    var_names_clean = ["Red", "Green", "Blue"]
    for label_int, label in zip(LABEL_INTS, LABELS):
        subset = truth[truth["label"] == label]
        subset.index = range(len(subset))
        variables = np.zeros((len(subset), len(variable_names)))
        i = 0
        for row in subset.iterrows():
            r = row[1]
            variables[i] = np.float32([r[column] for column in variable_names])
            i += 1
        indices = range(len(variable_names))
        data = pd.DataFrame({"label": np.array([np.repeat(var_names_clean[i], len(subset)) for i in indices]).flatten(),
                             "value": np.float32([variables[:, i] for i in range(variables.shape[1])]).flatten()})
        ax = sns.violinplot(y="value", x="label", data=data, palette=COLORS[1:][::-1])
        ax.set_ylim(0, np.nanquantile(data["value"], [0.999]))
        ax.set_ylabel("Normalized\nreflectance", fontsize=12)
        ax.set_xticklabels(var_names_clean, fontsize=12)
        ax.set_xlabel("")
        ax.set_title("Reflectances at '%s' label" % label, fontsize=16)
        plt.savefig(os.path.join(dirs["plots"], "reflectances_at_%s_label_violinplot.png" % label))

if __name__ == "__main__":
    area_column_names = [["validation_countries", "training_countries"], ["validation_areas", "training_areas"]]
    for these_area_column_names in area_column_names:
        plot_labels_relative(tiles_pd, these_area_column_names)
