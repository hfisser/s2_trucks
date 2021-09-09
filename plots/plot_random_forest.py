import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib import rcParams
from sklearn import metrics
from sklearn import tree

rcParams["font.serif"] = "Times New Roman"
rcParams["font.family"] = "serif"

dirs = dict(main="F:\\Masterarbeit\\DLR\\project\\1_truck_detection")
dirs["plots"] = os.path.join("F:" + os.sep + "Masterarbeit", "THESIS", "general", "plots")
dirs["truth"] = os.path.join(dirs["main"], "truth")

rf_file = os.path.join(dirs["main"], "code", "detect_trucks", "rf_model.pickle")
rf = pickle.load(open(rf_file, "rb"))
# read test variables and labels in order to calculate metrics
variables_list = pickle.load(open(os.path.join(dirs["truth"], "validation_variables.pickle"), "rb"))
labels_list = pickle.load(open(os.path.join(dirs["truth"], "validation_labels.pickle"), "rb"))


def plot_random_forest(rf_model, test_variables, test_labels):
    test_pred = rf.predict(test_variables)
    plot_confusion_matrix(metrics.confusion_matrix(test_labels, test_pred, labels=[2, 3, 4, 1]))
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
    columns = ["Precision", "Recall", "F1-score", "Support"]
    shape = summary.shape
    fig, ax = plt.subplots()
    summary_altered = summary.copy()  # copy in order to set n label column to 0 for imshow
    summary_altered[:, -1] = 0  # np.min(summary[0:-1, 0:3]) - 0.1
    summary_altered[summary_altered == 0] = np.nan
    cmap = cm.Greens.__copy__()
    im = ax.imshow(summary_altered.astype(np.float32), cmap=cmap, aspect=0.3)
    ax.set_xticks(np.arange(shape[1]))
    ax.set_yticks(np.arange(shape[0]))
    ax.set_yticklabels(["Background", "Blue", "Green", "Red", "Macro avg.", "Weighted avg.", "Accuracy"])
    ax.set_xticklabels(columns)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.subplots_adjust(bottom=0.2)
    for i in range(shape[0]):
        for j in range(shape[1]):
            value = summary[i, j]
            value = np.round(value, 2) if value <= 1 else np.int32(value)
            if value != 0:
                text = ax.text(j, i, value, ha="center", va="center", color="black")
    fig.tight_layout()
    plt.savefig(os.path.join(dirs["plots"], "rf_classification_summary_heatmap.png"), dpi=500)


def plot_confusion_matrix(conf_matrix):
    labels = ["blue", "green", "red", "background"]
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    cmap = cm.YlGn.__copy__()
    im = plt.imshow(conf_matrix, cmap=cmap)
    shape = conf_matrix.shape
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(shape[1]))
    ax.set_yticks(np.arange(shape[0]))
    ax.set_yticklabels(labels)
    ax.set_xticklabels(labels)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    plt.subplots_adjust(bottom=0.25, left=0.25)
    # add numeric labels inside plot
    for i in range(shape[0]):
        for j in range(shape[1]):
            value = str(conf_matrix[i, j])
            if len(value) == 2:
                value = " %s" % value
            elif len(value) == 1:
                value = "  %s" % value
            plt.text(i - 0.2, j + 0.11, value, fontsize=11)
    plt.text(1.2, -1.2, "True", fontsize=12, fontweight="bold")
    plt.text(-2.8, 2, "Predicted", fontsize=12, fontweight="bold", rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["plots"], "confusion_matrix.png"), dpi=500)
    plt.close()


def plot_feature_importance(rf_model):
    fig, ax = plt.subplots(figsize=(10, 1))
    left = 0
    feature_importances = np.round(rf_model.feature_importances_, 2)
    argsort = np.argsort(feature_importances)[::-1]
    labels = np.array(["reflectance_variance", "B04_B02_ratio", "B03_B02_ratio", "B04_centered", "B03_centered",
                       "B02_centered", "B08_centered"])[argsort]
    colors = np.array(["#757575", "#dc4ff0", "#39e7ad", "#ff0000", "#00ff00", "#0000ff", "#7c0912"])[argsort]
    feature_importances = feature_importances[argsort]
    offsets = [0.18, 0.12, 0, -0.1, -0.1, -0.5, -0.3]
    for c, importance, label, idx in zip(colors, feature_importances, labels, range(len(labels))):
        ax.barh(0, importance, height=0.2, color=c, left=left, edgecolor="black", label="label")
        text = ax.text(left + importance * 0.5, -0.01, "%s" % importance, ha="center",
                       va="center", color="w", weight="bold", fontsize=16)
        text = ax.text(left + importance * offsets[idx], [-0.24, 0.16][int(int(idx / 2) == idx / 2)], label, fontsize=16)
        left += importance
    text = ax.text(-0.015, -0.05, "0", fontsize=16)
    text = ax.text(1.005, -0.05, "1", fontsize=16)
    ax.set_xlabel("")
    plt.ylabel("")
    plt.subplots_adjust(bottom=0.8)
    plt.subplots_adjust(top=0.9)
    plt.xlim(0, left)
    positions = feature_importances.copy()
    for i in range(len(feature_importances)):
        positions[i] = np.sum(feature_importances[:i])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_yticklabels("")
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.3)
    plt.savefig(os.path.join(dirs["plots"], "rf_feature_importances_barplot.png"), dpi=500)
    plt.close()


if __name__ == "__main__":
    #plot_random_forest(rf, variables_list, labels_list)
    plot_feature_importance(rf)
