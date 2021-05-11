import os
import numpy as np
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

box_validation_pd = pd.read_csv(os.path.join(dirs["validation"], "boxes_validation.csv"))
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
    all_s2_counts, all_station_counts, all_dates, all_weekdays = [], [], [], []
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    # plot mean absolute values by weekday
    ax = axes[0]
    ax.plot(weekday_names, s2_means_weekdays, S2_COLOR, linewidth=2.5)
    ax.plot(weekday_names, station_means_weekdays, "#3e6118", linewidth=2.5)
    ax.set_ylabel("Truck count")
    ax.set_title("Truck count by weekday")
    ax.set_xlim(-0.1, 6.1)
    ax = axes[1]
    ax.bar(x=weekday_names, height=rvalues_weekdays, color=S2_COLOR)
    for idx, rvalue in enumerate(rvalues_weekdays):
        ax.text(idx - 0.15, rvalue + 0.005, str(rvalue), fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("pearson r-value")
    ax.set_title("Pearson r-values by weekday")
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
    names.append(r"$\bf{""Mean""}$")
    names.append(r"$\bf{""Median""}$")
    rvalues_argsort = np.argsort(rvalues)
    names_sorted, colors_sorted = np.array(names)[rvalues_argsort], np.array(colors)[rvalues_argsort]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(y=names_sorted, width=rvalues[rvalues_argsort], color=colors_sorted)
    for idx, rvalue in enumerate(rvalues[rvalues_argsort]):
        ax.text(rvalue + 0.001, idx - 0.3, str(rvalue), fontsize=10)
    ax.set_xlabel("pearson r-value")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in np.unique(colors_sorted)[:-1]]
    ax.legend(handles, ["Motorway (A)", "Primary/Trunk (B)"])
    plt.ylim(-0.5, len(names_sorted) - 0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["validation"], "plots", "station_validation_rvalues_by_station_barh.png"), dpi=500)
    plt.close()


if __name__ == "__main__":
    plot_bast_validation_rvalues()
