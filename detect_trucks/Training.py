import os
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio as rio
from datetime import datetime
from glob import glob

from array_utils.points import points_from_np
from detect_trucks.TruckDetector import Detector

# files
dir_data = "F:\\Masterarbeit\\DLR\\project\\1_truck_detection\\data"
dir_s2_subsets = os.path.join(dir_data, "s2", "subsets")
dir_labels = os.path.join(dir_data, "labels")
dir_ancillary = os.path.join(os.path.dirname(dir_data), "truth")
dir_training = os.path.join(os.path.dirname(dir_data), "training")
if not os.path.exists(dir_training):
    os.mkdir(dir_training)
files = glob(dir_s2_subsets + os.sep + "*.tif")
for f in files:
    if "_x0" in f:
        files.remove(f)

# constants
THRESHOLDS = pd.read_csv(os.path.join(dir_ancillary, "thresholds.csv"), index_col=0)
COLUMN_KEYS = ["validation_percentage", "detection_percentage",
               "validation_intersection_percentage", "detection_intersection_percentage"]

# parameters
subset_box = {"ymin": 0, "xmin": 0, "ymax": 0, "xmax": 0}
n_epochs, n_runs = 2, 5
min_r_squared, max_r_squared = 0.5, 0.9
min_score, max_score = 1., 1.8
step_ratios = 0.005
min_ratios = -step_ratios
max_ratios = min_ratios + step_ratios * 10


class TrainingLoop:
    def __init__(self, epochs, runs):
        self.epochs = epochs
        self.runs = runs
        self.training_tracker = None
        self.r_squared_thresholds = None
        self.score_thresholds = None

    def loop_rsquared(self, band_stack_files, init_min_rsquared, init_max_rsquared, init_min_score, init_max_score,
                      min_ratio_thresholds, sub_box):
        t0 = datetime.now()
        #step_r_squared = (init_max_rsquared - init_min_rsquared) / self.runs
        step_score = (init_max_score - init_min_score) / self.runs
        #self.r_squared_thresholds = np.arange(init_min_rsquared, init_max_rsquared + step_r_squared, step_r_squared)
        self.score_thresholds = np.arange(init_min_score, init_max_score + step_score, step_score)
        for epoch in range(self.epochs):
            self.r_squared_thresholds = np.repeat(0.6, len(self.score_thresholds))
            print("Epoch: " + str(epoch) + "\n" + "*" * 50)
            accuracy_files = []
            for file_idx, file in enumerate(band_stack_files):
                #file = "C:\\Users\\Lenovo\\Downloads\\test42.tif"
                print("=" * 50)
                print("Dataset: %s" % file_idx)
                print(os.path.basename(file))
                file_str = os.path.basename(file).split(".tif")[0].split("_y0")[0]
                validation_filename = file_str + "_validation.gpkg"
                accuracy_file = os.path.join(dir_training, file_str + "_accuracy_epoch%s_dataset%s.csv" % (epoch,
                                                                                                           file_idx))
                if os.path.exists(accuracy_file):  # has already been processed
                    accuracy_files.append(accuracy_file)
                    continue
                with rio.open(file, "r") as src:
                    rio_meta = src.meta
                    band_stack = np.zeros((src.count, src.height, src.width), dtype=np.float32)
                    for i in range(src.count):
                        band_stack[i] = src.read(i + 1)
                sub_box["ymax"] = band_stack.shape[1] if sub_box["ymax"] == 0 else sub_box["ymax"]
                sub_box["xmax"] = band_stack.shape[2] if sub_box["xmax"] == 0 else sub_box["xmax"]
                accuracy_pd = pd.DataFrame()
                detector = Detector()
                detector.min_blue_red = min_ratio_thresholds["br"]
                detector.min_blue_green = min_ratio_thresholds["bg"]
                band_stack_np = detector.pre_process({"B08": band_stack[3], "B04": band_stack[0],
                                                      "B03": band_stack[1], "B02": band_stack[2]},
                                                     rio_meta, sub_box)
                for run, curr_score in enumerate(self.score_thresholds):
                    print("-" * 50)
                    curr_rsquared = self.r_squared_thresholds[run]
                    print("\rRun: %s/%s | rsquared threshold: %s | score threshold: %s" % (run + 1,
                                                                                           len(self.r_squared_thresholds),
                                                                                           curr_rsquared, curr_score),
                          end=" ")
                    detector.min_r_squared = curr_rsquared
                    detector.min_score = curr_score
                    detections = detector.detect_trucks(band_stack_np)
                    try:
                        detections.to_file(os.path.join(dir_training, file_str + "_detections_%s.gpkg" % curr_rsquared),
                                           driver="GPKG")
                    except ValueError:
                        continue
                    accuracy_dict = detector.calc_primary_accuracy(detections,
                                                                   gpd.read_file(os.path.join(dir_labels,
                                                                                              validation_filename)))
                    print("Validation percentage: %s | Detection percentage: %s"
                          % (accuracy_dict["validation_percentage"], accuracy_dict["detection_percentage"]))
                    idx = len(accuracy_pd)
                    accuracy_pd.loc[idx, "rsquared"] = curr_rsquared
                    accuracy_pd.loc[idx, "score"] = curr_score
                    for key in COLUMN_KEYS:
                        accuracy_pd.loc[idx, key] = accuracy_dict[key]
                accuracy_pd.to_csv(accuracy_file)
                accuracy_files.append(accuracy_file)
            self._evaluate(accuracy_files)
        t1 = datetime.now()
        print("Training loop finished\nTime elapsed: %s" % str(np.round((t1 - t0).total_seconds() / 60, 1)))

    def _evaluate(self, accuracy_files):
        if self.training_tracker is None:
            self.training_tracker = pd.DataFrame()
        column_keys = COLUMN_KEYS
        mean_accuracies = np.zeros((len(column_keys), len(self.score_thresholds)))
        for k, key in enumerate(column_keys):
            accuracies = []
            for i, file in enumerate(accuracy_files):
                accuracies.append(pd.read_csv(file))
            all_tiles_accuracy = np.zeros((len(accuracies), len(self.score_thresholds)))
            for idx, accuracy_pd in enumerate(accuracies):
                for row_idx in range(len(accuracy_pd)):
                    value = float(accuracy_pd.loc[row_idx, key])
                    value = 100 - (value - 100) if value > 100 else value  # if value above 100 % treat as if below
                    all_tiles_accuracy[idx, row_idx] = value
            mean_accuracies[k] = np.nanmean(all_tiles_accuracy, 0)  # mean accuracy of all tiles
        # write combined scores per rsquared threshold
        for idx in range(mean_accuracies.shape[1]):
            tracker_idx = len(self.training_tracker)
            for k, key in enumerate(column_keys):
                self.training_tracker.loc[tracker_idx, key] = mean_accuracies[k, idx]
                self.training_tracker.loc[tracker_idx, "score"] = self.score_thresholds[idx]
        mean_accuracies = mean_accuracies[0:2]
        scores = np.sum(mean_accuracies, 0)  # + np.std(mean_accuracies, 0)  # score of all accuracies and all tiles
        max_score_idx = np.where(scores == np.nanmax(scores))[0][0]
        # generate new rsquared thresholds for next epoch
        lower, upper = max_score_idx - 1, max_score_idx + 1
        lower, upper = 0 if lower < 0 else lower, 0 if upper >= len(self.r_squared_thresholds) else upper
        #r_squared_lower = self.r_squared_thresholds[lower].copy()
        #r_squared_higher = self.r_squared_thresholds[upper].copy()
        score_lower = self.score_thresholds[lower].copy()
        score_higher = self.score_thresholds[upper].copy()
        #step_r_squared = (r_squared_higher - r_squared_lower) / self.runs
        step_score = (score_higher - score_lower) / self.runs
        #self.r_squared_thresholds = np.arange(r_squared_lower, r_squared_higher, step_r_squared)
        self.score_thresholds = np.arange(score_lower, score_higher, step_score)

    @staticmethod
    def loop_min_ratios(band_stack_files, init_min_ratios, overall_max_ratios, step, sub_box, ratio_thresholds_file):
        relation = THRESHOLDS["br_low"][0] / THRESHOLDS["bg_low"][0]
        min_thresholds = np.arange(init_min_ratios, overall_max_ratios, step)
        try:
            ratio_thresholds_pd = pd.read_csv(ratio_thresholds_file)
        except FileNotFoundError:
            ratio_thresholds_pd = pd.DataFrame()
            columns = ["min_blue_green_ratio", "min_blue_red_ratio", "perc_intersects", "perc_intersects_validation"]
            for file in band_stack_files:
                with rio.open(file, "r") as src:
                    rio_meta = src.meta
                    band_stack = np.zeros((src.count, src.height, src.width), dtype=np.float32)
                    for i in range(src.count):
                        band_stack[i] = src.read(i + 1)
                sub_box["ymax"] = band_stack.shape[1] if sub_box["ymax"] == 0 else sub_box["ymax"]
                sub_box["xmax"] = band_stack.shape[2] if sub_box["xmax"] == 0 else sub_box["xmax"]
                detector = Detector()
                band_stack_np = detector.pre_process({"B08": band_stack[3], "B04": band_stack[0],
                                                      "B03": band_stack[1], "B02": band_stack[2]},
                                                     rio_meta, sub_box)
                file_str = os.path.basename(file).split(".tif")[0].split("_y0")[0]
                validation_filename = file_str + "_validation.gpkg"
                validation_boxes = gpd.read_file(os.path.join(dir_labels, validation_filename))
                for idx, min_blue_green, min_blue_red in zip(range(len(min_thresholds)), min_thresholds, min_thresholds):
                    detector.band_stack_np = band_stack_np
                    detector.min_blue_green = min_blue_green
                    detector.min_blue_red = min_blue_red * relation
                    detector._detect()  # call internal method to run only ratio-based detection
                    detection_points = points_from_np(detector.trucks_np, 1, {"lon": detector.lon, "lat": detector.lat}, #
                                                      crs=rio_meta["crs"])
                    n_intersects = np.int16()
                    try:
                        geoms = detection_points.geometry
                    except AttributeError:
                        continue
                    for point in geoms:
                        n_intersects += np.int16(any(validation_boxes.buffer(20).intersects(point)))
                    tile_id = os.path.basename(file).split("_")[5]
                    for column, value in zip(columns, [min_blue_green, min_blue_red,
                                                       n_intersects / len(detection_points) * 100,
                                                       n_intersects / len(validation_boxes) * 100]):
                        ratio_thresholds_pd.loc[idx, tile_id + "_" + column] = value
                    ratio_thresholds_pd.loc[idx, tile_id] = len(detection_points)
            ratio_thresholds_pd["min_bg"] = min_thresholds
            ratio_thresholds_pd["min_br"] = min_thresholds * relation
            ratio_thresholds_pd.to_csv(ratio_thresholds_file)
        best_ratio_idx = np.zeros(len(band_stack_files))
        for i, file in enumerate(band_stack_files):
            tile_id = os.path.basename(file).split("_")[5]
            deviation = np.abs(200 - ratio_thresholds_pd[tile_id + "_perc_intersects_validation"])
            try:
                best_ratio_idx[i] = np.where(deviation == np.nanmin(deviation))[0][-1]
            except IndexError:
                continue
        min_ratio = min_thresholds[np.int16(np.min(best_ratio_idx))]
        return {"bg": min_ratio, "br": min_ratio * relation}

    def finalize(self, tracker_file):
        if self.training_tracker is not None:
            self.training_tracker.to_csv(tracker_file)
        final_rsquared = (self.r_squared_thresholds[-1] + self.r_squared_thresholds[0]) / 2  # return middle of value
        final_score = (self.score_thresholds[-1] + self.score_thresholds[0]) / 2
        print("Final rsquared: %s | Final score: %s" % (final_rsquared, final_score))
        return final_rsquared, final_score


if __name__ == "__main__":
    ratios_file = os.path.join(dir_training, "ratios.csv")
    tracker_filename = "training_tracker"
    training_trackers = glob(dir_training + os.sep + tracker_filename + "*.csv")
    training_tracker_file = os.path.join(dir_training, tracker_filename + str(len(training_trackers)) + ".csv")
    training_loop = TrainingLoop(n_epochs, n_runs)
    min_ratio_t = training_loop.loop_min_ratios(files, min_ratios, max_ratios, step_ratios, subset_box, ratios_file)
    training_loop.loop_rsquared(files, min_r_squared, max_r_squared, min_score, max_score, min_ratio_t, subset_box)
    r_squared, score = training_loop.finalize(training_tracker_file)
    print(training_loop.r_squared_thresholds)
    print(training_loop.score_thresholds)
