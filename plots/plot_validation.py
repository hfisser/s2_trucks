import os
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from datetime import date
from matplotlib import rcParams
from scipy.stats import linregress

TR1, TR2 = "Lzg_R1", "Lzg_R2"
S21, S22 = "s2_direction1", "s2_direction2"
S2_COLOR = "#611840"

rcParams["font.serif"] = "Times New Roman"
rcParams["font.family"] = "serif"

dirs = dict(main="F:\\Masterarbeit\\DLR\\project\\1_truck_detection")
dirs["plots"] = os.path.join("F:" + os.sep + "Masterarbeit", "THESIS", "general", "plots")
dirs["validation"] = os.path.join(dirs["main"], "validation")

#box_validation_pd = pd.read_csv(os.path.join(dirs["validation"], "boxes_validation.csv"))
#bast_validation_pd = pd.read_csv(os.path.join(dirs["validation"], "validation_run.csv"))
#series_pd = pd.read_csv(os.path.join(dirs["validation"], "series_comparison.csv"))
countries = list(pd.read_csv(os.path.join(dirs["main"], "training", "tiles.csv"), sep=",")["validation_countries"].dropna())


def plot_box_validation(box_validation, country_names):
    for column in ["producer_percentage", "user_percentage"]:
        column_clean = column.split("_")
        column_clean = column_clean[0][0].upper() + column_clean[0][1:] + " %s [%%]" % "accuracy"
        argsort = np.argsort(np.float32(box_validation[column]))[::-1]
        data = pd.DataFrame({"Country": np.array(country_names)[argsort],
                             column: np.float32(box_validation[column])[argsort]})
        ax = sns.barplot(y=column, x="Country", data=data, color="#0e420a")
        ax.set_xticklabels(country_names, fontsize=10)
        plt.ylim(0, 100)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.subplots_adjust(bottom=0.2)
        plt.xlabel("")
        plt.ylabel(column_clean, size=10)
        plt.title("Box validation " + column_clean.split("[")[0])
        plt.savefig(os.path.join(dirs["plots"], column_clean + "box_validation_barplot.png"), dpi=300)
        plt.close()


def plot_bast_validation(bast_validation):
    s2_direction1 = np.float32(bast_validation["s2_direction1"])
    s2_direction2 = np.float32(bast_validation["s2_direction2"])
    lzg_direction1 = np.float32(bast_validation["Lzg_R1"])
    lzg_direction2 = np.float32(bast_validation["Lzg_R2"])
    valid = (lzg_direction1 >= 0) * (lzg_direction2 >= 0)
    s2_direction1, s2_direction2 = s2_direction1[valid], s2_direction2[valid]
    lzg_direction1, lzg_direction2 = lzg_direction1[valid], lzg_direction2[valid]
    positions = [[105, 130], [105, 100]]
    positions = [[130, 180], [130, 140]]
    colors = ["#3e8e53", "#c5ca2c"]
    for s2, bast, direction in zip([s2_direction1, s2_direction2], [lzg_direction1, lzg_direction2], ["1", "2"]):
        c = colors[int(direction) - 1]
        ax = sns.scatterplot(x=s2, y=bast, color=c)
        ax = sns.regplot(x=s2, y=bast, color=c)
        regress = linregress(x=s2, y=bast)
        position = positions[int(direction) - 1]
        plt.text(position[0], position[1],
                 "Direction%s\nLin. regression\nr-value=%s\nslope=%s" % (direction, np.round(regress.rvalue, 2),
                                                                          np.round(regress.slope, 2)), fontsize=8)
    plt.ylabel("BAST Lzg")
    plt.xlabel("Sentinel-2 count")
    plt.title("Sentinel-2 trucks vs. BAST Lzg trucks", fontsize=12)
    plt.subplots_adjust(right=0.85)
    plt.legend(["Direction 1", "Direction 2"], loc="upper right", bbox_to_anchor=(1.3, 0.5), fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["plots"], "s2_vs_bast_all_stations.png"), dpi=300)


def plot_bast_validation_rvalues():
    primary_road = ["Crailsheim", "Herzhausen", "Sprakensehl", "Winklarn"]
    files = glob(os.path.join(dirs["validation"], "validation*.csv"))
    rvalues, slopes, names, colors = np.zeros(len(files) + 2), np.zeros(len(files) + 2), [], []
    s2_detections, station_detections = np.zeros_like(rvalues), np.zeros_like(rvalues)
    all_s2_counts, all_station_counts, all_dates, all_weekdays, speeds = [], [], [], [], []
    for idx, file in enumerate(files):
        names.append(file.split("_run_")[-1].replace("_", " ").replace(".csv", ""))
        c = "#af2c74" if any([n in file for n in primary_road]) else S2_COLOR
        colors.append(c)
        v = pd.read_csv(file)
        dates_unique = np.unique(v["date"])
        v = v[np.where(v["date"] == dates_unique[0])[0][-1]:]
        station_sum, s2_sum = np.float32(v[TR1] + v[TR2]), np.float32(v[S21] + v[S22])
        regression = linregress(s2_sum, station_sum)
        rvalues[idx] = np.round(regression.rvalue, 2)
        slopes[idx] = np.round(regression.slope, 2)
        station_detections[idx] = np.mean(station_sum)
        s2_detections[idx] = np.mean(s2_sum)
        all_s2_counts.append(s2_sum)
        all_station_counts.append(station_sum)
        all_dates.append(list(v["date"]))
        all_weekdays.append([date.fromisoformat(d).weekday() for d in v["date"]])
    all_dates, all_weekdays = np.hstack(all_dates), np.hstack(all_weekdays)
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    rvalues_weekdays, slopes_weekdays = np.zeros(len(weekday_names)), np.zeros(len(weekday_names))
    all_s2_counts, all_station_counts = np.hstack(all_s2_counts), np.hstack(all_station_counts)
    s2_means_weekdays, station_means_weekdays = np.zeros(7), np.zeros(7)
    for w in np.arange(7):
        weekday_s2, weekday_station = all_s2_counts[all_weekdays == w], all_station_counts[all_weekdays == w]
        regression = linregress(weekday_s2, weekday_station)
        rvalues_weekdays[w] = np.round(regression.rvalue, 2)
        slopes_weekdays[w] = np.round(regression.slope, 2)
        station_means_weekdays[w] = np.mean(weekday_station)
        s2_means_weekdays[w] = np.mean(weekday_s2)
    # plot rvalue by weekday
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    # plot mean absolute values by weekday
    for ax in axes:
        ax.set_xticklabels(weekday_names, fontsize=16, rotation=45)
    ax = axes[0]
    ax.plot(weekday_names, s2_means_weekdays, S2_COLOR, linewidth=2.5)
    ax.plot(weekday_names, station_means_weekdays, "#3e6118", linewidth=2.5)
    ax.set_ylabel("Truck count", fontsize=16)
    ax.set_title("(a)", fontsize=16)
    ax.set_xlim(-0.1, 6.1)
    ax.text(1, 50, "Sentinel-2 trucks", fontsize=14)
    ax.text(3.9, 82, "Station trucks", fontsize=14)
    ax = axes[1]
    ax.bar(x=weekday_names, height=rvalues_weekdays, color=S2_COLOR)
    for idx, rvalue in enumerate(rvalues_weekdays):
        ax.text(idx - 0.4, rvalue + 0.005, str(rvalue), fontsize=14)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pearson r-value", fontsize=16)
    ax.set_title("(b)", fontsize=16)
    ax.set_xlim(-0.5, 6.5)
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["validation"], "plots", "comparison_by_weekday_lineplot_bar.png"), dpi=500)
    plt.close()
    # plot rvalue by low detection and high detection percentile
    low25, high75 = np.percentile(all_s2_counts, [25]), np.percentile(all_s2_counts, [75])
    regression = linregress(all_s2_counts[all_s2_counts < low25[0]], all_station_counts[all_s2_counts < low25[0]])
    print(low25)
    print(regression.rvalue)
    regression = linregress(all_s2_counts[all_s2_counts > high75[0]], all_station_counts[all_s2_counts > high75[0]])
    print(high75)
    print(regression.rvalue)
    for values, values_name in zip([all_s2_counts, all_station_counts], ["S2", "Station"]):
        print("-" * 20)
        print(values_name)
        print("Mean: %s" % np.round(values.mean(), 2))
        print("Std: %s" % np.round(values.std(), 2))
        print("Max: %s" % np.round(values.max(), 2))
        print("Min: %s" % np.round(values.min(), 2))
    # plot rvalue by station
    station_detections[-1] = np.mean(station_detections[:-1])
    s2_detections[-1] = np.mean(s2_detections[:-1])
    colors.append("black")  # median
    colors.append("black")  # mean
    rvalues[-1] = np.round(np.median(rvalues[:-1]), 2)
    rvalues[-2] = np.round(np.mean(rvalues[:-1]), 2)
    slopes[-1] = np.round(np.median(slopes[:-1]), 2)
    slopes[-2] = np.round(np.mean(slopes[:-1]), 2)
    names.append(r"$\bf{""Mean""}$")
    names.append(r"$\bf{""Median""}$")
    rvalues_argsort = np.argsort(rvalues)
    names_sorted, colors_sorted = np.array(names)[rvalues_argsort], np.array(colors)[rvalues_argsort]
    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    axes[0].barh(y=names_sorted, width=rvalues[rvalues_argsort], color=colors_sorted)
    for idx, rvalue in enumerate(rvalues[rvalues_argsort]):
        axes[0].text(rvalue - 0.09, idx - 0.28, str(rvalue), fontsize=16, color="w")
    axes[0].set_xlabel("Pearson r-value", fontsize=18)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in np.unique(colors_sorted)[:-1]]
    axes[0].legend(handles, ["Motorway (A)", "Primary/Trunk (B)"], fontsize=14)
    axes[0].set_yticklabels(names_sorted, fontsize=18)
    axes[0].set_ylim(-0.5, len(names_sorted) - 0.5)
    axes[0].set_xlim(0, 1.03)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), fontsize=14)
   # axes[0].set_title("Pearson r-value", fontsize=16)
    # plot slopes
    axes[1].barh(y=names_sorted, width=slopes[rvalues_argsort], color=colors_sorted)
 #   axes[1].set_ylabel(names_sorted, fontsize=12)
    axes[1].set_yticks([])
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Lin. regression slope", fontsize=18)
    axes[1].set_ylim(-0.5, len(names_sorted) - 0.5)
    axes[1].set_xlim(0, 2.52)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), fontsize=14)
    for idx, slope in enumerate(slopes[rvalues_argsort]):
        axes[1].text(slope - 0.2, idx - 0.28, str(slope), fontsize=16, color="w")
   # axes[1].set_title("Slope")
    fig.tight_layout()
    fig.savefig(os.path.join(dirs["validation"], "plots", "station_validation_rvalues_by_station_barh.pdf"), dpi=500)
    plt.close(fig)


def plot_detection_attributes():
    files = glob(os.path.join(dirs["validation"], "detections", "s2_detections*.gpkg"))
    speeds, headings, scores = [], [], []
    for file in files:
        if "Steinebrück" in file:
            continue
        else:
            detections = gpd.read_file(file)
            detections = detections[detections["score"] > 1.2]
            speeds.append(detections["speed"])
            headings.append(detections["direction_description"])
            scores.append(detections["score"])
    speeds, headings, scores = np.hstack(speeds), np.hstack(headings), np.hstack(scores)
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))
    c = "#611840"
    for ax, values, title in zip(axes.flatten(), [speeds, headings, scores], ["Speeds", "Headings", "Detection scores"]):
        print(title)
        if title == "Headings":
            directions = ["N", "NW", "W", "SW", "S", "SE", "E", "NE"]
            heading_counts = [np.count_nonzero(values == h) for h in directions]
            ax.bar(x=directions, height=heading_counts, color=c)
            ax.set_xlabel("Compass direction")
        else:
            parts = ax.violinplot(values, showmeans=False, showextrema=False)
            parts["bodies"][0].set_alpha(1)
            parts["bodies"][0].set_facecolor(c)
            parts["bodies"][0].set_edgecolor("#363636")
        #    parts["bodies"][0].set_edgecolor("black")
            ax.scatter([1], np.mean(values), marker="o", color="white", s=30, zorder=3)
            ax.vlines([1], np.percentile(values, [25])[0], np.percentile(values, [75])[0],
                      color="k", linestyle="-", lw=5)
            ax.vlines([1], np.min(values), np.max(values), color="k", linestyle="-", lw=1)
            ax.set_xticklabels([])
            ax.set_xticks([])
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["validation"], "plots", "speed_heading_score_violinplot_barplot.png"), dpi=500)
    plt.close()


if __name__ == "__main__":
    plot_bast_validation_rvalues()
    plot_detection_attributes()
