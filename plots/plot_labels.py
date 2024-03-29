import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm, rcParams
from scipy.stats import pearsonr
from collections import OrderedDict

rcParams["font.serif"] = "Times New Roman"
rcParams["font.family"] = "serif"

dirs = dict(main="F:\\Masterarbeit\\DLR\\project\\1_truck_detection")
dirs["training"] = os.path.join(dirs["main"], "training")
dirs["plots"] = os.path.join("F:" + os.sep + "Masterarbeit", "THESIS", "general", "plots")
tiles_pd = pd.read_csv(os.path.join(dirs["training"], "tiles.csv"), sep=",")
truth_pd = pd.read_csv(os.path.join(dirs["main"], "truth", "spectra_ml_training_tiles.csv"))


COLORS = ["#2e2e2e", "#0000ff", "#00ff00", "#ff0000"]
LABEL_INTS, LABELS = [1, 2, 3, 4], ["background", "blue", "green", "red"]
FEATURES = ["blue_normalized", "green_normalized", "red_normalized", "nir_normalized",
            "green_blue_ratio", "red_blue_ratio", "reflectance_var"]

#truth_pd["red_blue_ratio"] = (truth_pd["red"] - truth_pd["blue"]) / (truth_pd["red"] + truth_pd["blue"])
#truth_pd["green_blue_ratio"] = (truth_pd["green"] - truth_pd["blue"]) / (truth_pd["green"] + truth_pd["blue"])
#truth_pd["reflectance_var"] = np.var(np.float32([truth_pd["red"], truth_pd["green"], truth_pd["blue"]]).swapaxes(0, 1), 1)

def plot_labels_relative(tiles, area_columns):
    fig, ax = plt.subplots(figsize=(8, 2))
    pos, off, labels_sum = 0, 0.12, []
    colors = {"Europe": "#cbd5e8", "Africa": "#b3e2cd", "East Asia": "#fdcdac", "North America": "#fff2ae",
              "Oceania": "#e6f5c9", "South America": "#f4cae4"}
    continents_sorted = None
    for n_label_column, c, area_column in zip(["n_retain_validation", "n_retain"], ["Greens", "Purples"], area_columns):
        unique_areas = np.unique(tiles[area_column].dropna())
        unique_tiles = np.unique(tiles[{"n_retain_validation": "validation_tiles",
                                        "n_retain": "training_tiles"}[n_label_column]].dropna())
        n_labels = np.int16(tiles[n_label_column])
        n_sum = np.nansum(np.int16(tiles[n_label_column].dropna()))
        labels_sum.append(n_sum)
        shares = [np.sum(n_labels[np.where(tiles[area_column] == a)[0]]) / n_sum * 100 for a in unique_areas]
        continents = [np.array(tiles["training_areas"])[np.where(tiles[area_column] == a)[0]][0] for a in unique_areas]
        left = 0
        argsorted = np.argsort(continents)[::-1]
        continents_sorted = np.array(continents)[argsorted]
        for share, area, tile, continent in zip(np.array(shares)[argsorted], unique_areas[argsorted],
                                                unique_tiles[argsorted], continents_sorted):
            share_rounded = share - (left + share - 100) if (left + share) > 100 else share
            plt.barh(pos, share_rounded, height=0.1, color=colors[continent], left=left, edgecolor="black",
                     label=continent)
            text = ax.text(left + share_rounded * 0.5, pos, area.replace(" ", "\n") + "\n%s" % tile,
                           ha="center", va="center", color="black", fontsize=8)
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
    plt.legend(by_label.values(), by_label.keys(), loc="center right", bbox_to_anchor=(1.25, 0.5),
               fontsize=8)
    #plt.legend(by_label.values(), by_label.keys(), loc="center right", bbox_to_anchor=(1.25, 0.5))
    plt.xlim(0, 100)
    plt.xlabel("Share of labels in training/validation dataset [%]", fontsize=8)
 #   plt.title("Labels by country", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["plots"], "training_validation_%s.png" % area_columns[0]), dpi=600)
    plt.close()


def plot_label_distribution(truth):
    fig, ax = plt.subplots(figsize=(8, 1))
    left = 0
    counts = [np.count_nonzero(truth["label_int"] == label_int) for label_int in LABEL_INTS]
    for count, label, label_int in zip(counts, LABELS, LABEL_INTS):
        plt.barh(0, count, height=0.1, color=COLORS[label_int - 1], left=left, edgecolor="black", label=label)
        text = ax.text(left + count * 0.5, 0, label + "\nn=%s" % count, ha="center", va="center",
                       color="w", fontsize=10)
        left += count
    ax.set_yticks([0])
    ax.set_yticklabels(["Number of\n labels\nn=%s" % sum(counts)])
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    plt.xlim(0, left)
    plt.tight_layout()
    ax.set_yticklabels("")
    plt.savefig(os.path.join(dirs["plots"], "label_distribution_barplot.png"), dpi=300)
    plt.close()

"""
def plot_label_stats(truth):
    variable_names = ["red_normalized", "green_normalized", "blue_normalized", "reflectance_var", "red_blue_ratio",
                      "green_blue_ratio"]
    rgb_variables = ["Red", "Green", "Blue"]
    variances = []
    for label_int, label in zip(LABEL_INTS, LABELS):
        subset = truth[truth["label"] == label]
        subset.index = range(len(subset))
        variables = np.zeros((len(subset), len(variable_names)))
        i = 0
        for row in subset.iterrows():
            r = row[1]
            variables[i] = np.float32([r[column] for column in variable_names])
            i += 1
        indices = range(len(rgb_variables))
        data = pd.DataFrame({"label": np.hstack([np.repeat(rgb_variables[i], len(subset)) for i in indices]),
                             "value": np.hstack([variables[:, i] for i in indices]).flatten()})
        ax = sns.violinplot(y="value", x="label", data=data, palette=COLORS[1:][::-1])
        ax.set_ylim(0, 3.5)
        ax.set_ylabel("Normalized\nreflectance", fontsize=12)
        ax.set_xticklabels(rgb_variables, fontsize=12)
        ax.set_xlabel("")
        ax.set_title("Normalized reflectances at '%s' label" % label)
        plt.savefig(os.path.join(dirs["plots"], "reflectances_at_%s_label_violinplot.png" % label))
        plt.close()
        variances.append(variables[:, 3])  # for plotting all together
        # plot ratios
        indices = range(-2, 0)
        data = pd.DataFrame({"label": np.hstack([np.repeat(variable_names[i], len(subset)) for i in indices]),
                             "value": np.hstack([variables[:, i] for i in indices])})
        ax = sns.violinplot(y="value", x="label", data=data, palette=["#dc4ff0", "#39e7ad"])
        ax.set_ylim(-0.7, 0.7)
        ax.set_ylabel("Ratio", fontsize=12)
        ax.set_xticklabels(variable_names[-2:], fontsize=12)
        ax.set_xlabel("")
        ax.set_title("Reflectance ratios at '%s' label" % label, fontsize=12)
        plt.savefig(os.path.join(dirs["plots"], "reflectance_ratios_at_%s_label_violinplot.png" % label))
        plt.close()
    # plot variance
    data = pd.DataFrame({
        "label": np.hstack([np.repeat(label, len(variances[i])) for i, label in enumerate(LABELS)]),
        "value": np.hstack(variances)})
    ax = sns.violinplot(y="value", x="label", data=data, palette=["#757575"])
    ax.set_ylim(-0.1, np.nanquantile(data["value"], [0.999]))
    ax.set_ylabel("Variance", fontsize=12)
    labels_clean = ["'%s' label" % (the_label[0].upper() + the_label[1:]) for the_label in LABELS]
    ax.set_xticklabels(labels_clean, fontsize=12)
    ax.set_xlabel("")
    ax.set_title("Variance of normalized RGB reflectances by label", fontsize=12)
    plt.savefig(os.path.join(dirs["plots"], "rgb_variances_all_labels_violinplot.png"), dpi=300)
    plt.close()
"""


def plot_label_stats(truth):
    fig, axes = plt.subplots(4, 3, figsize=(11, 6), gridspec_kw={"width_ratios": [3, 2, 1]})
    ax_features = [FEATURES[:4], FEATURES[4:6], [FEATURES[-1]]]
    ax_features_clear = [["B02_centered", "B03_centered", "B04_centered", "B08_centered"],
                         ["B03_B02_ratio", "B04_B02_ratio"], ["reflectance_variance"]]
    colors = [["#0000ff", "#00ff00", "#ff0000", "#7c0912"], ["#39e7ad", "#dc4ff0"], ["#757575"]]
    for label_idx, label, row_axes in zip(range(len(axes)), ["blue", "green", "red", "background"], axes):
        row_axes = row_axes.flatten()
        for ax_idx, ax in enumerate(row_axes):
            ax_values = []
            for feature_idx, feature in enumerate(ax_features[ax_idx]):
                ax_values.append(np.float32(truth[truth["label"] == label][feature]))
            xticks = np.arange(0, len(ax_values) * 0.6, 0.6)
            parts = ax.violinplot(ax_values, positions=xticks,
                                  showmedians=False, showmeans=True, showextrema=False)
            parts["cmeans"].set_edgecolor("black")
            for pc, color in zip(parts["bodies"], colors[ax_idx]):
                pc.set_facecolor(color)
                pc.set_edgecolor("#363636")
                pc.set_alpha(1)
            ax.scatter(xticks, np.mean(np.float32(ax_values), 1), marker="o", color="white", s=15, zorder=3)
            ax.vlines(xticks, np.percentile(ax_values, [25], 1), np.percentile(ax_values, [75])[0],
                      color="k", linestyle="-", lw=5)
            ax.vlines(xticks, np.min(ax_values, 1), np.max(ax_values, 1), color="k", linestyle="-", lw=1)
            if ax_idx == 0:
                ax.set_ylim(-0.25, 0.25)
                ax.text(-1.7, -0.02, "'%s'" % label, fontsize=18)  # place class label
            elif ax_idx == 1:
                ax.set_ylim(-0.7, 0.7)
            elif ax_idx == 2:
                ax.set_ylim(0, 0.004)
            if label_idx == 3:
                ax.set_xticklabels(ax_features_clear[ax_idx], fontsize=18, rotation=45)
                ax.set_xticks(xticks)
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])
            plt.rc("ytick", labelsize=15)
    for title, ax in zip(["(a)", "(b)", "(c)"], axes[0]):
        ax.set_title(title, fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["plots"], "featuere_statistics_violinplot.png"), dpi=600)
    plt.close()


def plot_feature_correlation(truth):
    labels = ["B02_centered", "B03_centered", "B04_centered", "B08_centered", "B03_B02_ratio",
              "B04_B02_ratio", "reflectance_variance"]
    correlation_matrix = np.zeros((len(FEATURES), len(FEATURES)))
    for i, y in enumerate(FEATURES):
        for j, x in enumerate(FEATURES):
            correlation_matrix[i, j] = pearsonr(truth[x], truth[y])[0]
    fig, ax = plt.subplots(figsize=(5, 5))
    cmap = cm.YlGn.__copy__()
    im = plt.imshow(correlation_matrix, cmap=cmap)
    shape = correlation_matrix.shape
    ax.set_xticks(np.arange(shape[1]))
    ax.set_yticks(np.arange(shape[0]))
    ax.set_yticklabels(labels)
    ax.set_xticklabels(labels)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.subplots_adjust(bottom=0.25, left=0.4, right=1)
    plt.colorbar(im, fraction=0.02, pad=0.02)
    # add numeric labels inside plot
    for i in range(shape[0]):
        for j in range(shape[1]):
            value = str(np.round(correlation_matrix[i, j], 2))
            value = value if len(value) > 3 else value + "0"
            plt.text(i - 0.2, j + 0.1, value, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["plots"], "feature_correlations_heatmap.png"), dpi=500)
    plt.close()


if __name__ == "__main__":
    area_column_names = [["validation_countries", "training_countries"], ["validation_areas", "training_areas"]]
    for these_area_column_names in area_column_names:
        plot_labels_relative(tiles_pd, these_area_column_names)
    plot_label_stats(truth_pd)
   # plot_label_distribution(truth_pd)
   # plot_feature_correlation(truth_pd)
