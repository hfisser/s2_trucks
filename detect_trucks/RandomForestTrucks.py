import os
import warnings
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio as rio
import xarray as xr
from datetime import datetime
from glob import glob
from shutil import rmtree
from rasterio.transform import Affine
from shapely.geometry import Polygon, box
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from array_utils.math import rescale, normalized_ratio
from array_utils.geocoding import lat_from_meta, lon_from_meta, metadata_to_bbox_epsg4326
from osm_utils.utils import get_roads, rasterize_osm

warnings.filterwarnings("ignore")

dirs = dict(main="F:\\Masterarbeit\\DLR\\project\\1_truck_detection")
dirs["truth"] = os.path.join(dirs["main"], "truth")
dirs["s2_data"] = os.path.join(dirs["main"], "validation", "data", "s2", "archive")
dirs["osm"] = os.path.join(dirs["main"], "code", "detect_trucks", "AUXILIARY", "osm")
dirs["imgs"] = os.path.join(dirs["main"], "data", "s2", "subsets")
s2_file = os.path.join(dirs["s2_data"], "s2_bands_Salzbergen_2018-06-07_2018-06-07_merged.tiff")
#s2_file = os.path.join(dirs["s2_data"], "s2_bands_Theeßen_2018-11-28_2018-11-28_merged.tiff")
s2_file = os.path.join(dirs["s2_data"], "s2_bands_Nieder_Seifersdorf_2018-10-31_2018-10-31_merged.tiff")
#s2_file = os.path.join(dirs["s2_data"], "s2_bands_AS_Dierdorf_VQ_Nord_2018-05-08_2018-05-08_merged.tiff")#
s2_file = os.path.join(dirs["s2_data"], "s2_bands_Schuby_2018-05-05_2018-05-05_merged.tiff")
#s2_file = os.path.join(dirs["s2_data"], "s2_bands_Gospersgrün_2018-10-14_2018-10-14_merged.tiff")
#s2_file = os.path.join(dirs["s2_data"], "s2_bands_Offenburg_2018-09-27_2018-09-27_merged.tiff")
#s2_file = os.path.join(dirs["s2_data"], "s2_bands_Hagenow_2018-11-16_2018-11-16_merged.tiff")
#s2_file = os.path.join(dirs["s2_data"], "s2_bands_Bockel_2018-11-16_2018-11-16_merged.tiff")
#s2_file = os.path.join(dirs["s2_data"], "s2_bands_Schwandorf-Mitte_2018-07-03_2018-07-03_merged.tiff")
#s2_file = os.path.join(dirs["s2_data"], "s2_bands_Wurmberg_2018-09-27_2018-09-27_merged.tiff")
#s2_file = os.path.join(dirs["s2_data"], "s2_bands_Zimmern_ob_Rottweil_2018-09-27_2018-09-27_merged.tiff")
#s2_file = os.path.join(dirs["s2_data"], "s2_bands_Röstebachtalbrücke_2018-04-10_2018-04-10_merged.tiff")
#s2_file = os.path.join(dirs["main"], "data", "s2", "subsets", "S2A_MSIL2A_20200831T073621_N0214_R092_T37MCT_20200831T101156_y0_x0.tif")
#s2_file = os.path.join(dirs["main"], "data", "s2", "subsets", "S2A_MSIL2A_20200824T074621_N0214_R135_T35JPM_20200824T113239.tif")
#s2_file = os.path.join(dirs["main"], "data", "s2", "subsets", "S2B_MSIL2A_20200914T095029_N0214_R079_T34UDC_20200914T121343_y0_x0.tif")
#s2_file = os.path.join(dirs["imgs"], "S2B_MSIL2A_20200327T101629_N0214_R065_T32UNA_20200327T134849_y0_x0.tif")
tiles_pd = pd.read_csv(os.path.join(dirs["main"], "training", "tiles.csv"), sep=";")

do_tuning = False
truth_path = os.path.join(dirs["truth"], "spectra_ml.csv")
t = pd.read_csv(truth_path)

OSM_BUFFER = 30

# RF hyper parameters from hyper parameter tuning
N_ESTIMATORS = 800
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 1
MAX_FEATURES = "sqrt"
MAX_DEPTH = 90
BOOTSTRAP = False

SECONDS_OFFSET_B02_B04 = 1.01  # sensing offset between B02 and B04


class RFTruckDetector:
    def __init__(self):
        self.rf_spectral, self.rf_box = None, None
        self.lat, self.lon, self.meta = None, None, None
        self.variables = None
        self.truth_variables, self.truth_labels = None, None
        self.truth_variables_box, self.truth_labels_box = None, None
        self.truth_path_tmp = None
        self.background_mask, self.high_std_mask, self.low_ndvi_mask, self.max_mask = None, None, None, None
        self.std_mask_blue, self.var_mask_green, self.var_mask_red = None, None, None
        self.var_mask_blue, self.var_mask_green, self.var_mask_red = None, None, None
        self.blue_ratio_mask = None
        self.low_reflectance_mask = None
        self.probabilities = None

    def train(self):
        rf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                    min_samples_split=MIN_SAMPLES_SPLIT,
                                    min_samples_leaf=MIN_SAMPLES_LEAF,
                                    max_features=MAX_FEATURES,
                                    max_depth=MAX_DEPTH,
                                    bootstrap=BOOTSTRAP)
        self._prepare_truth(False)
        self._split_train_test()
        rf.fit(self.truth_variables["train"], self.truth_labels["train"])
        test_pred = rf.predict(self.truth_variables["test"])
        accuracy = metrics.accuracy_score(self.truth_labels["test"], test_pred)
        print("RF accuracy: %s" % accuracy)
        print("Feature importance: ")
        print(rf.feature_importances_)
    #    self.variables = self._build_variables(band_stack)
        try:
            os.remove(self.truth_path_tmp)
        except FileNotFoundError:
            pass
        self.rf_spectral = rf

    def tune(self, tuning_path):
        self._split_train_test()
        tuning_pd = pd.DataFrame()
        n_trees = list(np.int16(np.linspace(start=200, stop=2000, num=10)))
        max_features = ["auto", "sqrt"]
        max_depth = list(np.int16(np.linspace(10, 110, num=11)))
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        hyper_hyper = {"n_estimators": n_trees,
                       "max_features": max_features,
                       "max_depth": max_depth,
                       "min_samples_split": min_samples_split,
                       "min_samples_leaf": min_samples_leaf,
                       "bootstrap": bootstrap}
        rf = RandomForestClassifier()
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=hyper_hyper, n_iter=100, cv=3, verbose=2,
                                       random_state=42, n_jobs=-1)
        rf_random.fit(self.truth_variables["train"], self.truth_labels["train"])
        for key, value in rf_random.best_params_.items():
            tuning_pd[key] = [value]
        tuning_pd.to_csv(tuning_path)

    def preprocess_bands(self, band_stack, subset_box=None):
        bands_rescaled = band_stack[0:4].copy()
        bands_rescaled[np.isnan(bands_rescaled)] = 0
        bands_rescaled = rescale(bands_rescaled, 0, 1)
        bands_rescaled[bands_rescaled == 0] = np.nan
        band_stack = None
        self.lat, self.lon = lat_from_meta(self.meta), lon_from_meta(self.meta)
        if subset_box is not None:
            ymin, ymax, xmin, xmax = subset_box["ymin"], subset_box["ymax"], subset_box["xmin"], subset_box["xmax"]
            bands_rescaled = bands_rescaled[:, ymin:ymax, xmin:xmax]
            self.lat, self.lon = self.lat[ymin:ymax], self.lon[xmin:xmax]
            self.meta["height"] = bands_rescaled.shape[1]
            self.meta["width"] = bands_rescaled.shape[2]
        bbox_epsg4326 = list(np.flip(metadata_to_bbox_epsg4326(self.meta)))
        osm_mask = self._get_osm_mask(bbox_epsg4326, self.meta["crs"], bands_rescaled[0], {"lat": self.lat,
                                                                                           "lon": self.lon},
                                      dirs["osm"])
        bands_rescaled *= osm_mask
        bands_rescaled[bands_rescaled == 0] = np.nan
        osm_mask = None
        self._build_variables(bands_rescaled)

    def predict(self):
        t0 = datetime.now()
        self.variables[:, self.low_ndvi_mask == 0] = np.nan
        vars_reshaped = []
        for band_idx in range(self.variables.shape[0]):
            vars_reshaped.append(self.variables[band_idx].flatten())
        vars_reshaped = np.array(vars_reshaped).swapaxes(0, 1)  # (n observations, n variables)
        nan_mask = np.zeros_like(vars_reshaped)
        for var_idx in range(vars_reshaped.shape[1]):
            nan_mask[:, var_idx] = ~np.isnan(vars_reshaped[:, var_idx])  # exclude nans
        not_nan = np.nanmin(nan_mask, 1).astype(np.bool)
        predictions_flat = self.rf_spectral.predict_proba(vars_reshaped[not_nan])
     #   probabilities_shaped = vars_reshaped[:, 0:4].copy()
        probabilities_shaped = vars_reshaped[:, 1:4].copy()
        for idx in range(predictions_flat.shape[1]):
            probabilities_shaped[not_nan, idx] = predictions_flat[:, idx]
        probabilities_shaped = np.swapaxes(probabilities_shaped, 0, 1)
        probabilities_shaped = probabilities_shaped.reshape((probabilities_shaped.shape[0], self.variables.shape[1],
                                                             self.variables.shape[2]))
        probabilities_shaped[np.isnan(probabilities_shaped)] = 0
        meta = self.meta.copy()
        meta["count"] = probabilities_shaped.shape[0]
        meta["dtype"] = np.float32
        with rio.open(os.path.join(dirs["main"], "probs.tiff"), "w", **meta) as tgt:
            for i in range(probabilities_shaped.shape[0]):
                tgt.write(probabilities_shaped[i].astype(np.float32), i + 1)
        self.probabilities = probabilities_shaped
        classification = self._postprocess_prediction()
        return classification.astype(np.int8)

    def extract_objects(self, predictions_arr):
 #       with rio.open("C:\\Users\\Lenovo\\Downloads\\test94.tif", "r") as src:
  #          predictions_arr = src.read(1)
   #     with rio.open("C:\\Users\\Lenovo\\Downloads\\test95.tif", "r") as src:
    #        self.probabilities = src.read(1)
        self._build_box_model()
        t0 = datetime.now()
        predictions_copy = predictions_arr.copy()  # copy because will be modified
        probabilities_copy = self.probabilities.copy()
        blue_ys, blue_xs = np.where(predictions_copy == 2)
        out_gpd = gpd.GeoDataFrame(crs=self.meta["crs"])
        detection_boxes, directions, direction_descriptions, speeds, mean_probabilities, sub_size = [], [], [], [], [], 9
        for y_blue, x_blue in zip(blue_ys, blue_xs):
            if predictions_copy[y_blue, x_blue] == 0:
                continue
            subset_15 = self._get_arr_subset(predictions_copy, y_blue, x_blue, 15).copy()
      #      subset_15_copy = subset_15.copy()
       #     for value in [2, 3, 4]:
        #        ys, xs = np.where(subset_15_copy == value)
         ##       if len(ys) > 10:
           ###         for y, x in zip(ys, xs):
              #          clusters = subset_15_copy.copy()
               #         clusters, yet_seen = self._find_cluster(clusters, [y, x], value)
                ###        matches_y, matches_x = np.where(clusters == 100)
                   #     if len(matches_y) > 10:
                    #        for match_y, match_x in zip(matches_y, matches_x):
                     #           subset_15_copy[match_y, match_x] = 0
            #subset_15 = subset_15_copy
#            subset_9 = self._get_arr_subset(subset_15, half_idx_y, half_idx_x, 9)
 #           subset_3 = self._get_arr_subset(subset_15, half_idx_y, half_idx_x, 3)
            subset_9 = self._get_arr_subset(predictions_copy, y_blue, x_blue, sub_size).copy()
            subset_3 = self._get_arr_subset(predictions_copy, y_blue, x_blue, 3).copy()
            subset_9_probs = self._get_arr_subset(probabilities_copy, y_blue, x_blue, sub_size).copy()
            half_idx_y = y_blue if subset_9.shape[0] < sub_size else int(subset_9.shape[0] * 0.5)
            half_idx_x = x_blue if subset_9.shape[1] < sub_size else int(subset_9.shape[1] * 0.5)
  #          try:
   #             subset_9_probs[:] *= np.int8(subset_9 != 0)
    #        except ValueError:
     #           print(subset_9.shape)
      #          print(subset_9_probs.shape)
       #         continue
            try:
                current_value = subset_9[half_idx_y, half_idx_x]
            except IndexError:  # upper array edge
                half_idx_y, half_idx_x = int(sub_size / 2), int(sub_size / 2)  # index from lower edge is ok
                current_value = subset_9[half_idx_y, half_idx_x]
            new_value = 100
            if not all([value in subset_9 for value in [2, 3, 4]]):
                continue
            # eliminate free greens that do not belong to potential object most likely
            green_ys, green_xs = np.where(subset_9 == 3)
            for gy, gx in zip(green_ys, green_xs):
                red_ys, red_xs = np.where(subset_9 == 4)
                subset_3_tmp = self._get_arr_subset(subset_9, gy, gx, 3)
                if np.count_nonzero(subset_3_tmp > 1) < 2:
                    between = [False]
                else:
                    between = []
                    for ry, rx in zip(red_ys, red_xs):
                        between_ys = half_idx_y <= gy <= ry  or ry <= gy <= half_idx_y
                        between_xs = half_idx_x <= gx <= rx or rx <= gx <= half_idx_x
                        between.append(all([between_ys, between_xs]))
                if not any(between):
                    subset_9[gy, gx] = 0

            result_tuple = self._cluster_array(arr=subset_9,
                                               probs=subset_9_probs,
                                               point=[half_idx_y, half_idx_x],
                                               new_value=new_value,
                                               current_value=current_value,
                                               yet_seen_indices=[],
                                               yet_seen_values=[],
                                               joker_played=False)
            cluster = result_tuple[0]
            if np.count_nonzero(cluster == new_value) < 3:
                continue
            else:
                out_gpd, predictions_copy = self._postprocess_cluster(cluster, predictions_copy,
                                                                      subset_3, y_blue, x_blue,
                                                                      half_idx_y, half_idx_x, new_value, out_gpd)
        self._elapsed(t0)
        return out_gpd

    def _cluster_array(self, arr, probs, point, new_value, current_value, yet_seen_indices, yet_seen_values,
                       joker_played):
        """
        looks for non zeros in 3x3 window around point in array and assigns a new value to these non-zeros
        :param arr: np array
        :param point: list of int y, x indices
        :param new_value: int value to assign
        :param current_value: value from np array
        :param yet_seen_indices: list of lists, each list is a point with int y, x indices that has been seen before
        :param yet_seen_values: list of values, each value is a value at the yet_seen_indices
        :return: tuple of np array and list
        """
        joker_played = True
        if len(yet_seen_indices) == 0:
            yet_seen_indices.append(point)
            yet_seen_values.append(current_value)
        arr_modified = arr.copy()
        arr_modified[point[0], point[1]] = 0
        window_3x3 = self._get_arr_subset(arr_modified.copy(), point[0], point[1], 3)
        y, x, ys, xs, window_idx, offset_y, offset_x = point[0], point[1], [], [], 0, 0, 0
        window_5x5_no_corners = self._eliminate_array_corners(self._get_arr_subset(arr_modified.copy(), y, x, 5), 1)
        window_3x3_probs = self._get_arr_subset(probs, y, x, 3)
        window_5x5_probs_no_corners = self._eliminate_array_corners(self._get_arr_subset(probs, y, x, 5), 1)
        # first look for values on horizontal and vertical, if none given try corners
        windows, windows_probs = [window_3x3, window_5x5_no_corners], [window_3x3_probs, window_5x5_probs_no_corners]
        windows = windows[0:1] if current_value == 4 or joker_played else windows
        while len(ys) == 0 and window_idx < len(windows):
            window = windows[window_idx]
            window_probs = windows_probs[window_idx]
            offset_y, offset_x = int(window.shape[0] / 2), int(window.shape[1] / 2)  # offset for window ymin and xmin
            go_next = (current_value + 1) in window or current_value == 2
            target_value = current_value + 1 if go_next else current_value
            match = window == target_value
            ys, xs = np.where(match)
            if len(ys) > 1:  # look for match with highest probability
                window_probs_target = window_probs[target_value - 2] * match
                max_prob = (window_probs_target == np.max(window_probs_target))
                ys, xs = np.where(max_prob)
            window_idx += 1
        ymin, xmin = int(np.clip(point[0] - offset_y, 0, np.inf)), int(np.clip(point[1] - offset_x, 0, np.inf))
        for y_local, x_local in zip(ys, xs):
            y, x = ymin + y_local, xmin + x_local
            if [y, x] not in yet_seen_indices or len(yet_seen_indices) == 0:
                try:
                    current_value = arr[y, x]
                except IndexError:
                    continue
                if 4 in yet_seen_values and current_value <= 3:  # red yet seen but this is green or blue
                    continue
                arr_modified[y, x] = new_value
                yet_seen_indices.append([y, x])
                yet_seen_values.append(current_value)
                # avoid picking many more reds than blues and greens
                n_picks = [np.count_nonzero(np.array(yet_seen_values) == value) for value in [2, 3, 4]]
                if n_picks[2] > n_picks[0] and n_picks[2] > n_picks[1]:
                    break  # finish clustering in order to avoid picking many reds at the edge of object
                if any([n > 5 for n in n_picks]):
                    return np.zeros_like(arr_modified), yet_seen_indices, yet_seen_values, joker_played
                arr_modified, yet_seen_indices, yet_seen_values, joker_played = self._cluster_array(arr_modified,
                                                                                                    probs,
                                                                                                    [y, x],
                                                                                                    new_value,
                                                                                                    current_value,
                                                                                                    yet_seen_indices,
                                                                                                    yet_seen_values,
                                                                                                    joker_played)
        arr_modified[point[0], point[1]] = new_value
        return arr_modified, yet_seen_indices, yet_seen_values, joker_played

    def _postprocess_cluster(self, cluster, preds_copy, prediction_subset_3, y_blue, x_blue, half_idx_y, half_idx_x,
                             new_value, out_gpd):
        # add neighboring blue in 3x3 window around blue
        ys_blue_additional, xs_blue_additional = np.where(prediction_subset_3 == 2)
        ys_blue_additional += half_idx_y - 1  # get index in subset
        xs_blue_additional += half_idx_x - 1
        for y_blue_add, x_blue_add in zip(ys_blue_additional, xs_blue_additional):
            cluster[int(np.clip(y_blue_add, 0, np.inf)), int(np.clip(x_blue_add, 0, np.inf))] = new_value
        cluster[cluster != new_value] = 0
        cluster_ys, cluster_xs = np.where(cluster == new_value)
        # corner of 9x9 subset
        ymin_subset, xmin_subset = np.clip(y_blue - half_idx_y, 0, np.inf), np.clip(x_blue - half_idx_x, 0, np.inf)
        cluster_ys += ymin_subset.astype(cluster_ys.dtype)
        cluster_xs += xmin_subset.astype(cluster_xs.dtype)
        ymin, xmin = np.min(cluster_ys), np.min(cluster_xs)
        # +1 on index because Polygon has to extent up to upper bound of pixel (array coords at upper left corner)
        ymax, xmax = np.max(cluster_ys) + 1, np.max(cluster_xs) + 1
        # check if blue, green and red are given in box and box is large enough, otherwise drop
        box_preds = preds_copy[ymin:ymax, xmin:xmax].copy()
        box_probs = self.probabilities[:, ymin:ymax, xmin:xmax].copy()
        #box_probs = self.probabilities[1:, ymin:ymax, xmin:xmax].copy()
        max_prob_blue = np.nanmax(box_probs[0] * (box_preds == 2))
        max_prob_green = np.nanmax(box_probs[1] * (box_preds == 3))
        max_prob_red = np.nanmax(box_probs[2] * (box_preds == 4))
        max_probs = [max_prob_blue, max_prob_green, max_prob_red]
        spectral_probability = np.nanmean(max_probs)
        all_given = all([value in box_preds for value in [2, 3, 4]])
        large_enough = box_preds.shape[0] > 2 or box_preds.shape[1] > 2
        too_large = box_preds.shape[0] > 5 or box_preds.shape[1] > 5
        too_large += box_preds.shape[0] > 4 and box_preds.shape[1] > 4
        if too_large > 0 or not all_given or not large_enough:
            return out_gpd, preds_copy
        # predict probability of whole box spectrum
        box_vars = self.variables[:, ymin:ymax, xmin:xmax].copy()
        marker_box_variables = self._build_box_marker_variables(box_preds, box_probs, box_vars)
    #    box_probability = self.rf_box.predict_proba(marker_box_variables)[0][1]  # prediction probability of whole box
        box_probability = 1
        # calculate direction
        blue_y, blue_x = np.where(box_preds == 2)
        ry, rx = np.where(box_preds == 4)
        # simply use first index
        blue_indices = np.int8([blue_y[0], blue_x[0]])
        red_indices = np.int8([ry[0], rx[0]])
        blue_red_vector = red_indices - blue_indices
        direction = self.calc_vector_direction_in_degree(blue_red_vector)
        diameter = np.max(box_preds.shape) * 10 / 2  # 10 m resolution
        speed = (diameter * (3600 / SECONDS_OFFSET_B02_B04)) * 0.001
        # create output box
        lon_min, lat_min = self.lon[xmin], self.lat[ymin]
        try:
            lon_max = self.lon[xmax]
        except IndexError:  # may happen at edge of array
            # calculate next coordinate beyond array bound -> this is just the upper boundary of the box
            lon_max = self.lon[-1] + (self.lon[-1] - self.lon[-2])
        try:
            lat_max = self.lat[ymax]
        except IndexError:
            lat_max = self.lat[-1] + (self.lat[-1] - self.lat[-2])
        cluster_box = Polygon(box(lon_min, lat_min, lon_max, lat_max))
        above_50 = np.sum([p > 0.5 for p in max_probs]) > 2
        above_65 = np.sum([p > 0.65 for p in max_probs]) > 1
        above_80 = np.sum([p > 0.8 for p in max_probs]) > 0
        #combined_probability = [spectral_probability, box_probability]
        #both_high, combined_high = all([p > 0.6 for p in combined_probability]), np.mean(combined_probability) > 0.8
        #(both_high and combined_high) or any([p > 0.98 for p in combined_probability])
        if spectral_probability > 0.4: #and all([above_50, above_65, above_80]):
            # set box cells to zero value in predictions array
            preds_copy[ymin:ymax, xmin:xmax] *= np.zeros_like(box_preds)
            blue_indices = np.where(box_preds == 2)
            for yb, xb in zip(blue_indices[0], blue_indices[1]):  # 3x3 around cell blues to 0
                ymin, ymax = np.clip(yb - 1, 0, preds_copy.shape[0]), np.clip(yb + 2, 0, preds_copy.shape[0])
                xmin, xmax = np.clip(xb - 1, 0, preds_copy.shape[1]), np.clip(xb + 2, 0, preds_copy.shape[1])
                preds_copy[ymin:ymax, xmin:xmax] *= np.int8(preds_copy[ymin:ymax, xmin:xmax] != 2)
            rox_idx = len(out_gpd)
            for key, value in zip(["geometry", "id", "box_probability_true",
                                  "mean_spectral_probability", "max_blue_probability",
                                   "max_green_probability", "max_red_probability", "direction_degree",
                                   "direction_description", "speed"],
                                  [cluster_box, rox_idx, box_probability,
                                   spectral_probability, max_prob_blue, max_prob_green, max_prob_red, direction,
                                   self.direction_degree_to_description(direction), speed]):
                out_gpd.loc[rox_idx, key] = value
        return out_gpd, preds_copy

    def read_bands(self, file_path):
        try:
            with rio.open(file_path, "r") as src:
                self.meta = src.meta
                if src.count < 4:
                    raise TypeError("Need 4 bands but %s given" % src.count)
                band_stack = np.zeros((src.count, src.height, src.width), dtype=np.float32)
                for band_idx in range(src.count):
                    band_stack[band_idx] = src.read(band_idx + 1)
        except rio.errors.RasterioIOError as e:
            print("Could not read from %s" % file_path)
            raise e
        else:
            print("Read %s bands from %s" % (self.meta["count"], file_path))
        return band_stack

    def prediction_raster_to_gtiff(self, prediction_raster, file_path):
        if not any([file_path.endswith(suffix) for suffix in [".tif", ".tiff"]]):
            file_path += ".tiff"
        meta_copy = self.meta.copy()
        meta_copy["count"] = 1
        meta_copy["dtype"] = prediction_raster.dtype
        try:
            with rio.open(file_path, "w", **meta_copy) as tgt:
                tgt.write(prediction_raster, 1)
        except rio.errors.RasterioIOError as e:
            print("Could not write to %s" % file_path)
            raise e
        else:
            print("Wrote to: %s" % file_path)

    def prediction_boxes_to_gpkg(self, prediction_boxes, file_path):
        self._write_boxes(file_path, prediction_boxes, ".gpkg")

    def prediction_boxes_to_geojson(self, prediction_boxes, file_path):
        self._write_boxes(file_path, prediction_boxes, ".geojson")

    def _prepare_truth(self, add_background):
        truth_data = pd.read_csv(truth_path, index_col=0)
        truth_data.index, label = list(range(len(truth_data))), "background"
        b = np.where(truth_data["label"] == "background")[0]
        truth_data.drop(b, inplace=True)
    #    for row_idx in np.random.choice(np.where(truth_data["label"] == label)[0],
     #                                   int(np.count_nonzero(truth_data["label"] == "background") * 0.97), replace=False):
      #      truth_data.drop(row_idx, inplace=True)
        truth_data.index = list(range(len(truth_data)))
        self.truth_path_tmp = os.path.join(os.path.dirname(truth_path), "tmp.csv")
        try:
            truth_data.to_csv(self.truth_path_tmp)
        except AttributeError:
            self.truth_path_tmp = None

    def _split_train_test(self):
        truth_data = pd.read_csv(self.truth_path_tmp, index_col=0)
        labels, variables = truth_data["label_int"], []
        for key in ["reflectance_std", "reflectance_var", "red_blue_ratio", "green_blue_ratio", "red", "green",
                    "blue", "nir"]:
            variables.append(truth_data[key])
        variables = np.float32(variables).swapaxes(0, 1)
        variables_box = []
        for idx in np.where(labels == 4)[0]:
            variables_box.append(np.float32([variables[idx], variables[idx + 1],
                                             variables[idx + 2]]).flatten())  # rgb box marker spectra
        n_truth = len(variables_box)
#        background_indices = np.where(labels == 1)[0]
 #       for indices in [[0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 2, 1]]:
  #          for a, b, c in zip(background_indices[indices[0:]][::3],
   #                            background_indices[indices[1]:][::3], background_indices[indices[2]:][::3]):
    #            variables_box.append(np.float32([variables[a], variables[b], variables[c]]).flatten())
 #       n_background = int(n_truth * 10)
  #      for background_idx in range(n_background):
   #         variables_box.append(np.random.random(len(variables_box[0])))  # artificial background spectra
    #    variables_box = np.float32(variables_box)
        labels_box = [1] * n_truth + [0] * (len(variables_box) - n_truth)
        vars_train, vars_test, labels_train, labels_test = train_test_split(variables, list(labels), test_size=0.15)
        vars_train_box, vars_test_box, labels_train_box, labels_test_box = train_test_split(variables_box, labels_box,
                                                                                            test_size=0.15)
        self.truth_variables = dict(train=vars_train, test=vars_test)
        self.truth_labels = dict(train=labels_train, test=labels_test)
        self.truth_variables_box = dict(train=vars_train_box, test=vars_test_box)
        self.truth_labels_box = dict(train=labels_train_box, test=labels_test_box)

    def _build_variables(self, band_stack):
        rgb_var = np.nanvar(band_stack[0:3], 0, dtype=np.float16)
        self.var_mask_blue = np.int8(rgb_var > np.nanquantile(rgb_var, [0.25]))
        self.var_mask_green = np.int8(rgb_var > np.nanquantile(rgb_var, [0.75]))
        self.var_mask_red = np.int8(rgb_var > np.nanquantile(rgb_var, [0.65]))
        red_blue_ratio = normalized_ratio(band_stack[0], band_stack[2])
        green_blue_ratio = normalized_ratio(band_stack[1], band_stack[2])
        green_blue_mask = np.int8(green_blue_ratio < np.nanquantile(green_blue_ratio, [0.1]))
        red_blue_mask = np.int8(red_blue_ratio < np.nanquantile(red_blue_ratio, [0.1]))
        self.blue_ratio_mask = green_blue_mask * red_blue_mask
        self.low_reflectance_mask = np.zeros_like(band_stack[0])
        self.low_reflectance_mask += np.int8(band_stack[0] > np.nanquantile(band_stack[0], [0.25]))
        self.low_reflectance_mask += np.int8(band_stack[1] > np.nanquantile(band_stack[1], [0.25]))
        self.low_reflectance_mask += np.int8(band_stack[2] > np.nanquantile(band_stack[2], [0.25]))
        self.low_reflectance_mask[self.low_reflectance_mask >= 1] = 1
        self.low_ndvi_mask = np.int8(normalized_ratio(band_stack[3], band_stack[0]) < 0.7)
        variables = np.zeros((8, band_stack.shape[1], band_stack.shape[2]), dtype=np.float16)
        variables[0] = np.nanstd(band_stack[0:3], 0, dtype=np.float16)
        variables[1] = np.nanvar(band_stack[0:3], 0, dtype=np.float16)
        variables[2] = normalized_ratio(band_stack[0], band_stack[2]).astype(np.float16)  # red/blue
        variables[3] = normalized_ratio(band_stack[1], band_stack[2]).astype(np.float16)  # green/blue
        variables[4] = band_stack[0]
        variables[5] = band_stack[1]
        variables[6] = band_stack[2]
        variables[7] = band_stack[3]
        meta = self.meta.copy()
        meta["count"] = variables.shape[0]
        meta["dtype"] = np.float32
        with rio.open(os.path.join(dirs["main"], "test1.tiff"), "w", **meta) as tgt:
            for idx in range(variables.shape[0]):
                tgt.write(variables[idx].astype(np.float32), idx + 1)
        self.variables = variables

    def _build_box_marker_variables(self, box_predictions, box_probabilities, box_variables):
        n_variables = self.variables.shape[0]
        variables = np.zeros((1, n_variables * 3))
        for i, value in enumerate([2, 3, 4]):
            value_probs = box_probabilities.copy().astype(np.float32)
            value_probs[:, box_predictions != value] = np.nan
            z, y, x = np.where(value_probs == np.nanmax(value_probs, 0))
            variables[0, i * n_variables:(i + 1) * n_variables] = box_variables[:, y[0], x[0]].flatten()
        return variables

    def _build_box_model(self):
        rf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                    min_samples_split=MIN_SAMPLES_SPLIT,
                                    min_samples_leaf=MIN_SAMPLES_LEAF,
                                    max_features=MAX_FEATURES,
                                    max_depth=MAX_DEPTH,
                                    bootstrap=BOOTSTRAP)
        rf.fit(self.truth_variables_box["train"], self.truth_labels_box["train"])
        test_pred = rf.predict(self.truth_variables_box["test"])
        accuracy = metrics.accuracy_score(self.truth_labels_box["test"], test_pred)
        self.rf_box = rf

    def _measure_cluster(self, arr, value, ys, xs, n):
        for next_y, next_x in zip(ys, xs):
            n += 1
            sub_3x3 = self._get_arr_subset(arr, next_y, next_x, 3).copy()
            sub_3x3[1, 1] = 0
            arr_copy = np.zeros_like(arr)
            ymin, ymax = np.clip(next_y - 1, 0, arr.shape[0]), np.clip(next_y + 2, 0, arr.shape[0])
            xmin, xmax = np.clip(next_x - 1, 0, arr.shape[1]), np.clip(next_x + 2, 0, arr.shape[1])
            arr_copy[ymin:ymax, xmin:xmax] = sub_3x3
            arr[next_y, next_x] = 0
            ys, xs = np.where(arr_copy == value)
            n = self._measure_cluster(arr, value, ys, xs, n)
        return n

    def _eliminate_clusters(self, arr):
        for value in [2, 3, 4]:  # eliminate large clusters of pixels
            arr_copy = arr.copy()
            ys, xs = np.where(arr_copy == value)
            for y, x in zip(ys, xs):
                subset_3 = self._get_arr_subset(arr, y, x, 3)
                n_matches = np.count_nonzero(subset_3 == value)
                single_blue = value == 2 and n_matches == 1 and np.count_nonzero(~np.isnan(subset_3)) == 0
                if n_matches > (subset_3.shape[0] * subset_3.shape[1]) * 0.5 or single_blue:
                    arr[y, x] = 0
        return arr

    def _find_cluster(self, a, point, value, yet_seen=None):
        yet_seen = [point] if yet_seen is None else yet_seen
        if a[point[0], point[1]] != value:
            return a, yet_seen
        a_copy = a.copy()
        mask = np.zeros_like(a_copy)
        ymin, ymax = np.clip(point[0] - 1, 0, a_copy.shape[0]), np.clip(point[0] + 2, 0, a_copy.shape[0])
        xmin, xmax = np.clip(point[1] - 1, 0, a_copy.shape[1]), np.clip(point[1] + 2, 0, a_copy.shape[1])
        mask[ymin:ymax, xmin:xmax] = np.ones((ymax - ymin, xmax - xmin))
        #  mask[point[0], point[1]] = 0
        a_copy *= mask
        ys, xs = np.where(a_copy == value)
        a_copy = a.copy()
        a_copy[point[0], point[1]] = 100
        for y, x in zip(ys, xs):
            new_point = [y, x]
            if new_point in yet_seen:
                continue
            else:
                yet_seen.append(new_point)
                a_copy, yet_seen = self._find_cluster(a_copy, new_point, value, yet_seen)
                a_copy[y, x] = 100
        return a_copy, yet_seen

    def _postprocess_prediction(self):
        classification = np.argmax(self.probabilities, 0) + 2
        classification[(self.probabilities[0] < 0.5) * (classification == 2)] = 0
        classification[(self.probabilities[1] < 0.5) * (classification == 3)] = 0
        classification[(self.probabilities[2] < 0.5) * (classification == 4)] = 0
        classification[self.low_ndvi_mask == 0] = 0
        classification[(np.int8(self.blue_ratio_mask == 0) * np.int8(classification == 2)) == 1] = 0
        classification[self.low_reflectance_mask == 0] = 0
        classification[(np.int8(self.var_mask_blue == 0) * np.int8(classification == 2)) == 1] = 0
        classification[(np.int8(self.var_mask_green == 0) * np.int8(classification == 3)) == 1] = 0
        classification[(np.int8(self.var_mask_red == 0) * np.int8(classification == 4)) == 1] = 0
        classification = self._eliminate_clusters(classification)
        #classification_copy = classification.copy()
       # classification = classification_copy.copy().astype(np.int8)
        classification_copy = None
        return classification

    @staticmethod
    def _get_arr_subset(arr, y, x, size):
        pseudo_max = 1e+30
        size_low = int(size / 2)
        size_up = int(size / 2)
        size_up = size_up + 1 if (size_low + size_up) < size else size_up
        ymin, ymax = int(np.clip(y - size_low, 0, pseudo_max)), int(np.clip(y + size_up, 0, pseudo_max))
        xmin, xmax = int(np.clip(x - size_low, 0, pseudo_max)), int(np.clip(x + size_up, 0, pseudo_max))
        n = len(arr.shape)
        if n == 2:
            subset = arr[ymin:ymax, xmin:xmax]
        elif n == 3:
            subset = arr[:, int(np.clip(y - size_low, 0, pseudo_max)):int(np.clip(y + size_up, 0, pseudo_max)),
                     int(np.clip(x - size_low, 0, pseudo_max)):int(np.clip(x + size_up, 0, pseudo_max))]
        else:
            subset = arr
        return subset

    @staticmethod
    def _eliminate_array_corners(arr, assign_value):
        y_shape_idx = 1 if len(arr.shape) == 3 else 0
        y_max, x_max = arr.shape[y_shape_idx], arr.shape[y_shape_idx + 1]
        ys, xs = [0, 0, y_max, y_max], [0, x_max, 0, x_max]
        if arr.shape[y_shape_idx] >= 5:
            ys += [0, 1, y_max, y_max - 1, 0, 1, y_max, y_max - 1]  # eliminate 4x3 corner positions
            xs += [1, 0, 1, 0, x_max - 1, x_max, x_max - 1, x_max]
        for y_idx, x_idx in zip(ys, xs):
            try:
                arr[y_idx, x_idx] = assign_value  # pseudo background
            except IndexError:  # edge
                continue
        return arr

    @staticmethod
    def _get_osm_mask(bbox, crs, reference_arr, lat_lon_dict, dir_out):
        osm_file = get_roads(bbox, ["motorway", "trunk", "primary"], OSM_BUFFER,
                             dir_out, str(bbox).replace(", ", "_").replace("-", "minus")[1:-1] + "_osm_roads", str(crs),
                             reference_arr)
        osm_vec = gpd.read_file(osm_file)
        ref_xr = xr.DataArray(data=reference_arr, coords=lat_lon_dict, dims=["lat", "lon"])
        osm_raster = rasterize_osm(osm_vec, ref_xr).astype(np.float16)
        osm_raster[osm_raster != 0] = 1
        osm_raster[osm_raster == 0] = np.nan
        return osm_raster

    @staticmethod
    def _elapsed(start_time):
        print("Elapsed: %s s" % (datetime.now() - start_time).total_seconds())

    @staticmethod
    def _write_boxes(file_path, prediction_boxes, suffix):
        if not file_path.endswith(suffix):
            file_path += suffix
        drivers = {".geojson": "GeoJSON", ".gpkg": "GPKG"}
        try:
            prediction_boxes.to_file(file_path, driver=drivers[suffix])
        except TypeError as e:
            raise e
        else:
            print("Wrote to: %s" % file_path)

    @staticmethod
    def _add_background(out_pd, vars, n_background):
        # pick random indices from non nans
 #       not_nan_reflectances = np.int8(~np.isnan(reflectances[0:4]))
  #      not_nan_ratios = np.int8(~np.isnan(ratios))
        not_nan = np.int8(~np.isnan(vars[0]))
        for i in range(vars.shape[0]):
            not_nan *= np.int8(~np.isnan(vars[i]))
        not_nan_y, not_nan_x = np.where(not_nan == 1)
        try:
            random_indices = np.random.randint(0, len(not_nan_y), n_background)
        except ValueError:
            return
        row_idx = len(out_pd)
        for random_idx in zip(random_indices):
            y_arr_idx, x_arr_idx = not_nan_y[random_idx], not_nan_x[random_idx]
     #       rgb = reflectances[0:3, y_arr_idx, x_arr_idx]
            out_pd.loc[row_idx, "label_int"] = 1
            out_pd.loc[row_idx, "label"] = "background"
      #      out_pd.loc[row_idx, "red"] = reflectances[0, y_arr_idx, x_arr_idx]
       #     out_pd.loc[row_idx, "green"] = reflectances[1, y_arr_idx, x_arr_idx]
        #    out_pd.loc[row_idx, "blue"] = reflectances[2, y_arr_idx, x_arr_idx]
         #   out_pd.loc[row_idx, "nir"] = reflectances[3, y_arr_idx, x_arr_idx]
            out_pd.loc[row_idx, "reflectance_var"] = vars[0, y_arr_idx, x_arr_idx]
            out_pd.loc[row_idx, "red_blue_ratio"] = vars[1, y_arr_idx, x_arr_idx]
            out_pd.loc[row_idx, "green_blue_ratio"] = vars[2, y_arr_idx, x_arr_idx]
            out_pd.loc[row_idx, "red_difference"] = vars[3, y_arr_idx, x_arr_idx]
            out_pd.loc[row_idx, "green_difference"] = vars[4, y_arr_idx, x_arr_idx]
            out_pd.loc[row_idx, "blue_difference"] = vars[5, y_arr_idx, x_arr_idx]
            out_pd.loc[row_idx, "reflectance_std"] = vars[6, y_arr_idx, x_arr_idx]
            row_idx += 1
        return out_pd

    @staticmethod
    def calc_vector_direction_in_degree(vector):
        # [1,1] -> 45°; [-1,1] -> 135°; [-1,-1] -> 225°; [1,-1] -> 315°
        y_offset = 90 if vector[0] > 0 else 0
        x_offset = 90 if vector[1] < 0 else 0
        offset = 180 if y_offset == 0 and x_offset == 90 else 0
        if vector[0] == 0:
            direction = 0.
        else:
            direction = np.degrees(np.arctan(np.abs(vector[1]) / np.abs(vector[0])))
        direction += offset + y_offset + x_offset
        return direction

    @staticmethod
    def direction_degree_to_description(direction_degree):
        step = 22.5
        bins = np.arange(0, 359, step, dtype=np.float32)
        descriptions = np.array(["N", "NNE", "NE", "ENE",
                                 "E", "ESE", "SE", "SEE",
                                 "S", "SSW", "SW", "WSW",
                                 "W", "WNW", "NW", "NNW"])
        i, b = 0, -1
        while b < direction_degree and i < len(bins):
            b = bins[i]
            i += 1
        return descriptions[i - 1]

    @staticmethod
    def expose_anomalous_pixels(band_stack_np):
        w = 50
        y_bound, x_bound = band_stack_np.shape[1], band_stack_np.shape[2]
        roads = np.zeros((3, band_stack_np.shape[1], band_stack_np.shape[2]), dtype=np.float32)
        for y in range(int(np.round(y_bound / w))):
            for x in range(int(np.round(x_bound / w))):
                y_idx, x_idx = np.clip((y + 1) * w, 0, y_bound), np.clip((x + 1) * w, 0, x_bound)
                y_low, x_low = int(np.clip(y_idx - w, 0, 1e+30)), int(np.clip(x_idx - w, 0, 1e+30))
                y_up, x_up = np.clip(y_idx + w + 1, 0, y_bound), np.clip(x_idx + w + 1, 0, x_bound)
                y_size, x_size = (y_up - y_low), (x_up - x_low)
                n = y_size * x_size
                subset = band_stack_np[:, y_low:y_up, x_low:x_up]
                roads[0, y_low:y_up, x_low:x_up] = np.repeat(np.nanmedian(subset[0]), n).reshape(y_size, x_size)
                roads[1, y_low:y_up, x_low:x_up] = np.repeat(np.nanmedian(subset[1]), n).reshape(y_size, x_size)
                roads[2, y_low:y_up, x_low:x_up] = np.repeat(np.nanmedian(subset[2]), n).reshape(y_size, x_size)
        diff_red = (band_stack_np[0] - (roads[0] / 2)) / (band_stack_np[0] + (roads[0] / 2))
        diff_green = (band_stack_np[1] - (roads[1] / 2)) / (band_stack_np[1] + (roads[1] / 2))
        diff_blue = (band_stack_np[2] - (roads[2] / 2)) / (band_stack_np[2] + (roads[2] / 2))
        diff_stack = np.array([diff_red, diff_green, diff_blue])
        mask = np.zeros_like(diff_stack[0])
        for i in range(diff_stack.shape[0]):
            mask += np.int8(diff_stack[i] > np.nanquantile(diff_stack[i], [0.90]))
        mask[mask != 0] = 1
        mask[mask == 0] = np.nan
        return np.float32(mask), diff_stack


if __name__ == "__main__":
    rf_td = RFTruckDetector()
    bands = rf_td.read_bands(s2_file)
    s = {"xmin": 0, "ymin": 3000, "xmax": 3000, "ymax": 6000}
    s = None
    rf_td.preprocess_bands(bands, s)
    rf_td.train()
    predictions = rf_td.predict()
    boxes = rf_td.extract_objects(predictions)
    print(len(boxes))
    try:
        name = os.path.basename(s2_file).split("bands_")[1].split("_merged")[0]
    except IndexError:
        name = os.path.basename(s2_file).split(".")[0]
    rf_td.prediction_raster_to_gtiff(predictions, os.path.join(dirs["main"], name + "_raster.tiff"))
    rf_td.prediction_boxes_to_gpkg(boxes, os.path.join(dirs["main"], name + "_boxes.gpkg"))
