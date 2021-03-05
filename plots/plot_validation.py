import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

dirs = dict(main="F:\\Masterarbeit\\DLR\\project\\1_truck_detection")
dirs["plots"] = os.path.join("F:" + os.sep + "Masterarbeit", "THESIS", "general", "plots")
dir_validation = os.path.join(dirs["main"], "validation")

sns.set_theme(style="whitegrid")

box_validation_pd = pd.read_csv(os.path.join(dir_validation, "boxes_validation.csv"))
bast_validation_pd = pd.read_csv(os.path.join(dir_validation, "validation_run.csv"))
series_pd = pd.read_csv(os.path.join(dir_validation, "series_comparison.csv"))
countries = list(pd.read_csv(os.path.join(dirs["main"], "training", "tiles.csv"), sep=";")["validation_countries"].dropna())


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
                 "Direction%s\nLin. regression\nrsquared=%s\nslope=%s" % (direction, np.round(regress.rvalue, 2),
                                                                          np.round(regress.slope, 2)), fontsize=8)
    plt.ylabel("BAST Lzg")
    plt.xlabel("Sentinel-2 count")
    plt.title("Sentinel-2 trucks vs. BAST Lzg trucks", fontsize=12)
    plt.subplots_adjust(right=0.85)
    plt.legend(["Direction 1", "Direction 2"], loc="upper right", bbox_to_anchor=(1.3, 0.5), fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["plots"], "s2_vs_bast_all_stations.png"), dpi=300)
