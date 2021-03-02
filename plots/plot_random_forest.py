import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import metrics
from sklearn import tree

dirs = dict(main="F:\\Masterarbeit\\DLR\\project\\1_truck_detection")
dirs["plots"] = os.path.join("F:" + os.sep + "Masterarbeit", "THESIS", "general", "plots")
dirs["truth"] = os.path.join(dirs["main"], "truth")

rf_file = os.path.join(dirs["main"], "code", "detect_trucks", "rf_model.pickle")
rf = pickle.load(open(rf_file, "rb"))
# read test variables and labels in order to calculate metrics
variables_list = pickle.load(open(os.path.join(dirs["truth"], "validation_variables.pickle"), "rb"))
labels_list = pickle.load(open(os.path.join(dirs["truth"], "validation_labels.pickle"), "rb"))

test_variables = variables
test_labels = labels


def plot_random_forest(rf_model, test_variables, test_labels):
    test_pred = rf.predict(test_variables)
    accuracy = metrics.accuracy_score(test_labels, test_pred)
    report = metrics.classification_report(test_labels, test_pred)
    labels = np.unique(test_labels)
    summary = np.zeros((len(labels) + 3, 4), dtype=np.float16)
    for i, label in enumerate(labels):
        for j, fun in enumerate([metrics.precision_score, metrics.recall_score, metrics.f1_score]):
            summary[i, j] = fun(test_labels, test_pred, average="micro", labels=[label])
            summary[-3, j] = fun(test_labels, test_pred, average="macro")
            summary[-2, j] = fun(test_labels, test_pred, average="weighted")
        summary[i, 3] = np.count_nonzero(np.int8(test_labels) == label)
    summary[-3, 3] = len(test_labels)
    summary[-2, 3] = len(test_labels)
    summary[-1, 3] = len(test_labels)
    summary[-1, 2] = metrics.accuracy_score(test_labels, test_pred)
    columns = ["Precision", "Recall", "F1-score", "N labels"]
    shape = summary.shape
    fig, ax = plt.subplots()
    summary_altered = summary.copy()  # copy in order to set n label column to 0 for imshow
    summary_altered[:, -1] = 0 # np.min(summary[0:-1, 0:3]) - 0.1
    summary_altered[summary_altered == 0] = np.nan
    cmap = cm.Greens.__copy__()
    #cmap.set_under("w", alpha=0)
    im = ax.imshow(summary_altered.astype(np.float32), cmap=cmap, aspect=0.3)
   # im = ax.imshow(summary_altered[:, 0:3].astype(np.float32), cmap=cmap, aspect=0.3)
    ax.set_xticks(np.arange(shape[1]))
    ax.set_yticks(np.arange(shape[0]))
    # ... and label them with the respective list entries
    ax.set_yticklabels(["Background", "Blue", "Green", "Red", "Macro avg.", "Weighted avg.", "Accuracy"])
    ax.set_xticklabels(columns)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.subplots_adjust(bottom=0.2)
    for i in range(shape[0]):
        for j in range(shape[1]):
            value = summary[i, j]
            value = np.round(value, 2) if value <= 1 else np.int32(value)
            if value != 0:
                text = ax.text(j, i, value, ha="center", va="center", color="black", fontsize=10)
    fig.tight_layout()
    plt.savefig(os.path.join(dirs["plots"], "rf_classification_summary_heatmap.png"), dpi=300)


if __name__ == "__main__":
    plot_random_forest()