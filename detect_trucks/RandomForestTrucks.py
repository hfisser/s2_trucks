import os
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio as rio
import xarray as xr
from datetime import datetime
from glob import glob
from shutil import rmtree
from rasterio.transform import Affine
from rasterio.merge import merge
from shapely.geometry import Polygon, box
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from array_utils.math import rescale, normalized_ratio
from array_utils.geocoding import lat_from_meta, lon_from_meta, metadata_to_bbox_epsg4326
from osm_utils.utils import get_roads, rasterize_osm

dirs = dict(main="F:\\Masterarbeit\\DLR\\project\\1_truck_detection")
dirs["truth"] = os.path.join(dirs["main"], "truth")
dirs["s2_data"] = os.path.join(dirs["main"], "validation", "data", "s2", "archive")
dirs["osm"] = os.path.join(dirs["main"], "code", "detect_trucks", "AUXILIARY", "osm")
dirs["imgs"] = os.path.join(dirs["main"], "data", "s2", "subsets")
s2_file = os.path.join(dirs["s2_data"], "s2_bands_Salzbergen_2018-06-07_2018-06-07_merged.tiff")
#s2_file = os.path.join(dirs["s2_data"], "s2_bands_Theeßen_2018-11-28_2018-11-28_merged.tiff")
s2_file = os.path.join(dirs["s2_data"], "s2_bands_Nieder Seifersdorf_2018-10-31_2018-10-31_merged.tiff")

#s2_file = os.path.join(dirs["main"], "data", "s2", "subsets", "S2A_MSIL2A_20200831T073621_N0214_R092_T37MCT_20200831T101156.tif")
#s2_file = os.path.join(dirs["main"], "data", "s2", "subsets", "S2A_MSIL2A_20200824T074621_N0214_R135_T35JPM_20200824T113239.tif")
tiles_pd = pd.read_csv(os.path.join(dirs["main"], "training", "tiles.csv"), sep=";")

do_tuning = False
truth_csv = os.path.join(dirs["truth"], "spectra_ml.csv")

OSM_BUFFER = 40

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
        self.rf_model = None
        self.lat, self.lon, self.meta = None, None, None
        self.variables = None
        self.truth_path_tmp = None

    def run(self, band_stack, truth_path):
        tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        try:
            os.mkdir(tmp_dir)
        except FileExistsError:
            pass
        sub_size = 2000
        n_y_subsets = int(band_stack.shape[1] / sub_size + 1)
        n_x_subsets = int(band_stack.shape[2] / sub_size + 1)
        all_detections, tmp_files = [], []
        for y in range(n_y_subsets):
            for x in range(n_x_subsets):
                ymin, ymax = y * sub_size, (y + 1) * sub_size
                xmin, xmax = x * sub_size, (x + 1) * sub_size
                band_stack_subset = band_stack[:, ymin:ymax, xmin:xmax]
                lat_copy, lon_copy = self.lat.copy(), self.lon.copy()
                self.lat = self.lat[ymin:ymax]
                self.lon = self.lon[xmin:xmax]
                try:
                    self.train(band_stack_subset, truth_path)
                except ValueError:
                    prediction_arr = np.zeros((len(self.lat), len(self.lon)))
                    prediction_boxes = []
                else:
                    prediction_arr = self.predict()
                    prediction_boxes = self.extract_objects(prediction_arr)
                meta = self.meta.copy()
                t = meta["transform"]
                meta["transform"] = Affine(t[0], t[1], self.lon[0], t[3], t[4], self.lat[0])
                meta["height"] = prediction_arr.shape[0]
                meta["width"] = prediction_arr.shape[1]
                meta["dtype"] = prediction_arr.dtype
                meta["count"] = 1
                tmp_file = os.path.join(tmp_dir, "subset_tmp_%s_%s.tiff" % (y, x))
                with rio.open(tmp_file, "w", **meta) as tgt:
                    tgt.write(prediction_arr, 1)
                tmp_files.append(tmp_file)
                if len(prediction_boxes) > 0:
                    all_detections.append(prediction_boxes)
                self.lat, self.lon = lat_copy, lon_copy
        # merge prediction arrays
        merged, transform = merge(tmp_files)
        merged = merged[0]
        if merged.shape[0] > self.meta["height"]:
            merged = merged[0:self.meta["height"], :]
        if merged.shape[1] > self.meta["width"]:
            merged = merged[:, 0:self.meta["width"]]
        rmtree(tmp_dir)
        return merged, pd.concat(all_detections)

    def train(self, band_stack, truth_path, coeff):
        variables = self._build_variables(band_stack)
        self.variables = variables
        #n = np.count_nonzero(~np.isnan(band_stack)) / 25
        self._prepare_truth(truth_path, coeff)
        try:
            self._split_train_test()
        except ValueError as e:
            raise e
        rf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                    min_samples_split=MIN_SAMPLES_SPLIT,
                                    min_samples_leaf=MIN_SAMPLES_LEAF,
                                    max_features=MAX_FEATURES,
                                    max_depth=MAX_DEPTH,
                                    bootstrap=BOOTSTRAP)
        rf.fit(self.vars["train"], self.labels["train"])
        test_pred = rf.predict(self.vars["test"])
        accuracy = metrics.accuracy_score(self.labels["test"], test_pred)
        print("RF accuracy: %s" % accuracy)
        try:
            os.remove(self.truth_path_tmp)
        except FileNotFoundError:
            pass
        self.rf_model = rf

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
        rf_random.fit(self.vars["train"], self.labels["train"])
        for key, value in rf_random.best_params_.items():
            tuning_pd[key] = [value]
        tuning_pd.to_csv(tuning_path)

    def extract_background_for_tuning(self, truth_path, band_stack):
        variables = self._build_variables(band_stack)
        self.variables = variables
        self._prepare_truth(truth_path)
        return pd.read_csv(self.truth_path_tmp, index_col=0)

    def preprocess_bands(self, band_stack, dir_osm, subset_box=None):
        band_stack = band_stack[0:4]
        band_stack[np.isnan(band_stack)] = 0
        bands_rescaled = rescale(band_stack.copy(), 0, 1)
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
                                      dir_osm)
        bands_rescaled *= osm_mask
        bands_rescaled[bands_rescaled == 0] = np.nan
        osm_mask = None
        return bands_rescaled

    def predict(self):
        t0 = datetime.now()
        vars_reshaped = []
        for band_idx in range(self.variables.shape[0]):
            vars_reshaped.append(self.variables[band_idx].flatten())
        vars_reshaped = np.array(vars_reshaped).swapaxes(0, 1)  # (n observations, n variables)
        nan_mask = np.zeros_like(vars_reshaped)
        for idx in range(vars_reshaped.shape[1]):
            nan_mask[:, idx] = ~np.isnan(vars_reshaped[:, idx])  # exclude nans. there should not be any but anyways
        not_nan = np.nanmin(nan_mask, 1).astype(np.bool)
        predictions_flat = self.rf_model.predict(vars_reshaped[not_nan])
        predictions_shaped = vars_reshaped[:, 0].copy()
        predictions_shaped[not_nan] = predictions_flat
        predictions_shaped = predictions_shaped.reshape((self.variables.shape[1], self.variables.shape[2]))
        predictions_shaped[predictions_shaped == 0] = np.nan
        predictions_shaped[predictions_shaped > 4] = 1
        self._elapsed(t0)
        return predictions_shaped.astype(np.int8)

    def extract_objects(self, predictions_arr):
        t0 = datetime.now()
        preds = predictions_arr.copy()  # copy because will be modified
        blue_ys, blue_xs = np.where(preds == 2)
        detection_boxes, directions, direction_descriptions, speeds, sub_size = [], [], [], [], 15
        for y_blue, x_blue in zip(blue_ys, blue_xs):
            if preds[y_blue, x_blue] == 0:
                continue
            subset_15 = self._get_arr_subset(preds, y_blue, x_blue, sub_size).copy()
            subset_3 = self._get_arr_subset(preds, y_blue, x_blue, 3).copy()
            if np.count_nonzero(subset_3 > 1) <= 1:  # not background
                continue
            half_idx_y = y_blue if subset_15.shape[0] < sub_size else int(subset_15.shape[0] * 0.5)
            half_idx_x = x_blue if subset_15.shape[1] < sub_size else int(subset_15.shape[1] * 0.5)
            try:
                current_value = subset_15[half_idx_y, half_idx_x]
            except IndexError:  # upper array edge
                half_idx_y, half_idx_x = int(sub_size / 2), int(sub_size / 2)  # index from lower edge is ok
                current_value = subset_15[half_idx_y, half_idx_x]
            new_value = 100
            # eliminate reds directly neighboring blue (only in preds copy), not 3x3 window -> only '+'
            for y_off, x_off in zip([-1, 0, 0, 1], [0, -1, 1, 0]):
                this_y = np.clip(half_idx_y + y_off, 0, subset_15.shape[0] - 1)
                this_x = np.clip(half_idx_x + x_off, 0, subset_15.shape[1] - 1)
                if subset_15[this_y, this_x] == 4:
                    subset_15[this_y, this_x] = 0
            if not all([value in subset_15 for value in [2, 3, 4]]):
                continue
            cluster, yet_seen_indices, yet_seen_values = self._cluster_array(arr=subset_15,
                                                                             point=[half_idx_y, half_idx_x],
                                                                             new_value=new_value,
                                                                             current_value=current_value,
                                                                             yet_seen_indices=[],
                                                                             yet_seen_values=[])
            if np.count_nonzero(cluster == new_value) < 3:
                continue
            # add neighboring blue in 3x3 window around blue
            ys_blue_additional, xs_blue_additional = np.where(subset_3 == 2)
            ys_blue_additional += half_idx_y - 1  # get index in subset
            xs_blue_additional += half_idx_x - 1
            for y_blue_add, x_blue_add in zip(ys_blue_additional, xs_blue_additional):
                cluster[int(np.clip(y_blue_add, 0, np.inf)), int(np.clip(x_blue_add, 0, np.inf))] = new_value
            cluster[cluster != new_value] = 0
            cluster_ys, cluster_xs = np.where(cluster == new_value)
            # corner of 15x15 subset
            ymin_subset, xmin_subset = np.clip(y_blue - half_idx_y, 0, np.inf), np.clip(x_blue - half_idx_x, 0, np.inf)
            cluster_ys += ymin_subset.astype(cluster_ys.dtype)
            cluster_xs += xmin_subset.astype(cluster_xs.dtype)
            ymin, xmin = np.min(cluster_ys), np.min(cluster_xs)
            # +1 on index because Polygon has to extent up to upper bound of pixel (array coords at upper left corner)
            ymax, xmax = np.max(cluster_ys) + 1, np.max(cluster_xs) + 1
            # check if blue, green and red are given in box and box is large enough, otherwise drop
            box_preds = predictions_arr[ymin:ymax, xmin:xmax]
            all_given = all([value in box_preds for value in [2, 3, 4]])
            large_enough = box_preds.shape[0] > 2 or box_preds.shape[1] > 2
            too_large = box_preds.shape[0] > 5 or box_preds.shape[1] > 5
            too_large *= box_preds.shape[0] > 4 and box_preds.shape[1] > 4
            if too_large or not all_given or not large_enough:
                continue
            # calculate direction
            blue_y, blue_x = np.where(box_preds == 2)
            red_y, red_x = np.where(box_preds == 4)
            # simply use first index
            red_indices, blue_indices = np.int8([red_y[0], red_x[0]]), np.int8([blue_y[0], blue_x[0]])
            blue_red_vector = red_indices - blue_indices
            direction = self.calc_vector_direction_in_degree(blue_red_vector)
            diameter = np.max(box_preds.shape) * 10 / 2  # 10 m resolution
            speed = (diameter * (3600 / SECONDS_OFFSET_B02_B04)) * 0.001
            # set cells from cluster and all blue cells to zero value in predictions array
            preds[ymin:ymax, xmin:xmax] = np.int8(box_preds != 2)  # set blue cells in preds within box to zero
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
            detection_boxes.append(cluster_box)
            directions.append(direction)
            direction_descriptions.append(self.direction_degree_to_description(direction))
            speeds.append(speed)
        out_gpd = gpd.GeoDataFrame({"id": list(range(1, len(detection_boxes) + 1)),
                                    "direction_degree": directions,
                                    "direction_description": direction_descriptions,
                                    "speed": speeds},
                                   geometry=detection_boxes,
                                   crs=self.meta["crs"])
        self._elapsed(t0)
        return out_gpd

    def _cluster_array(self, arr, point, new_value, current_value, yet_seen_indices, yet_seen_values):
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
        if len(yet_seen_indices) == 0:
            yet_seen_indices.append(point)
            yet_seen_values.append(current_value)
        arr_modified = arr.copy()
        arr_modified[point[0], point[1]] = 0
        window_3x3 = self._get_arr_subset(arr_modified.copy(), point[0], point[1], 3)
        # first look for values on horizontal and vertical, if none given try corners
        window_3x3_without_corners = self._eliminate_array_corners(window_3x3.copy(), 1)
        # try matches in 3x3 window, if none given in 3x3 without corner
        ys, xs, window_idx = [], [], 0
        windows = [window_3x3_without_corners, window_3x3]
        offset_y, offset_x = 0, 0
        while len(ys) == 0 and window_idx < len(windows):
            window = windows[window_idx]
            offset_y, offset_x = int(window.shape[0] / 2), int(window.shape[1] / 2)  # offset for window ymin and xmin
            if (current_value + 1) in window:
                ys, xs = np.where((window - current_value) == np.ones_like(window))  # one value higher
            else:
                ys, xs = np.where(window == current_value)  # equal value
            window_idx += 1
        ymin, xmin = int(np.clip(point[0] - offset_y, 0, np.inf)), int(np.clip(point[1] - offset_x, 0, np.inf))
        for y_local, x_local in zip(ys, xs):
            y, x = ymin + y_local, xmin + x_local
            if [y, x] not in yet_seen_indices or len(yet_seen_indices) == 0:
                current_value = arr[y, x]
                if 4 in yet_seen_values and current_value <= 3:  # red yet seen but this is green or blue
                    continue
                arr_modified[y, x] = new_value
                yet_seen_indices.append([y, x])
                yet_seen_values.append(current_value)
                # avoid picking many more reds than blues and greens
                n_picks = [np.count_nonzero(np.array(yet_seen_values) == value) for value in [2, 3, 4]]
                if n_picks[2] > n_picks[0] and n_picks[2] > n_picks[1]:
                    break  # finish clustering in order to avoid picking many reds at the edge of object
                if any([n > 4 for n in n_picks]):
                    return np.zeros_like(arr_modified), yet_seen_indices, yet_seen_values
                arr_modified, yet_seen_indices, yet_seen_values = self._cluster_array(arr_modified, [y, x],
                                                                                      new_value,
                                                                                      current_value,
                                                                                      yet_seen_indices,
                                                                                      yet_seen_values)
        arr_modified[point[0], point[1]] = new_value
        return arr_modified, yet_seen_indices, yet_seen_values

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

    def _prepare_truth(self, truth_path, coefficient):
        truth_data = pd.read_csv(truth_path, index_col=0)
        background_indices = np.where(truth_data["label"] == "background")[0]
        n = int(len(background_indices))
        for idx in np.random.choice(background_indices, n, replace=False):
            truth_data.drop(idx, inplace=True)
        truth_data = self._add_background(truth_data, self.variables[0:4], self.variables[-4:], self.variables[-5],
                                          len(truth_data[truth_data["label"] != "background"]))
   #     background = truth_data[truth_data["label"] == "background"]
    #    from sklearn.cluster import KMeans
     #   columns = background.columns
      #  value_idx = np.where(columns == "red")[0][0]
       # vectors = np.vstack([background[col] for col in columns[value_idx:]]).swapaxes(0, 1)
      #  k_means = KMeans(n_clusters=2).fit(vectors)
      #  labels = np.int8(k_means.labels_)
      #  labels += 1
      #  labels *= -1  # background classes have negative index
      #  for i, idx in enumerate(truth_data.index[np.where(truth_data["label"] == "background")[0]]):
      #      truth_data.loc[idx, "label_int"] = labels[i]
        self.truth_path_tmp = os.path.join(os.path.dirname(truth_path), "tmp.csv")
        try:
            truth_data.to_csv(self.truth_path_tmp)
        except AttributeError:
            self.truth_path_tmp = None

    def _split_train_test(self):
        truth_data = pd.read_csv(self.truth_path_tmp, index_col=0)
        labels = truth_data["label_int"]
        variables = [truth_data["red"], truth_data["green"], truth_data["blue"], truth_data["nir"],
                     truth_data["reflectance_std"],
                     truth_data["ndvi"],
                     truth_data["red_blue_ratio"],
                     truth_data["green_red_ratio"],
                     truth_data["blue_red_ratio"],
                     truth_data["green_blue_ratio"]]
        variables = np.float32(variables).swapaxes(0, 1)
        vars_train, vars_test, labels_train, labels_test = train_test_split(variables, list(labels), test_size=0.2)
        self.vars = dict(train=vars_train, test=vars_test)
        self.labels = dict(train=labels_train, test=labels_test)

    @staticmethod
    def _build_variables(band_stack):
        shape = band_stack.shape
        variables = np.zeros((shape[0] + 6, shape[1], shape[2]), dtype=np.float16)
        for band_idx in range(shape[0]):
            variables[band_idx] = band_stack[band_idx]
        variables[-6] = np.nanstd(band_stack[0:3], 0, dtype=np.float16)  # reflectance std
        variables[-5] = normalized_ratio(band_stack[3], band_stack[0]).astype(np.float16)  # NDVI
        variables[-4] = normalized_ratio(band_stack[0], band_stack[2]).astype(np.float16)  # red/blue
        variables[-3] = normalized_ratio(band_stack[1], band_stack[0]).astype(np.float16)  # green/red
        variables[-2] = normalized_ratio(band_stack[2], band_stack[0]).astype(np.float16)  # blue/red
        variables[-1] = normalized_ratio(band_stack[1], band_stack[2]).astype(np.float16)  # green/blue
        return variables

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
        for y_idx, x_idx in zip([0, 0, y_max, y_max], [0, x_max, 0, x_max]):
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
    def _add_background(out_pd, reflectances, ratios, ndvi, n_truth):
        # pick random indices from non nans
        not_nan_reflectances = np.int8(~np.isnan(reflectances[0:4]))
        not_nan_ndvi = np.int8(~np.isnan(ndvi))
        not_nan_ratios = np.int8(~np.isnan(ratios))
        not_nan = np.min(not_nan_reflectances, 0) * not_nan_ndvi * np.min(not_nan_ratios, 0)
        min_background, max_background = int(n_truth * 0.1), n_truth
        not_nan_y, not_nan_x = np.where(not_nan == 1)
        n_background = int(np.clip(np.count_nonzero(not_nan == 1) * 0.04,
                                   min_background,
                                   max_background))
        try:
            random_indices = np.random.randint(0, len(not_nan_y), n_background)
        except ValueError:
            return
        row_idx = len(out_pd)
        for random_idx in zip(random_indices):
            y_arr_idx, x_arr_idx = not_nan_y[random_idx], not_nan_x[random_idx]
            out_pd.loc[row_idx, "label_int"] = 1
            out_pd.loc[row_idx, "label"] = "background"
            out_pd.loc[row_idx, "red"] = reflectances[0, y_arr_idx, x_arr_idx]
            out_pd.loc[row_idx, "green"] = reflectances[1, y_arr_idx, x_arr_idx]
            out_pd.loc[row_idx, "blue"] = reflectances[2, y_arr_idx, x_arr_idx]
            out_pd.loc[row_idx, "nir"] = reflectances[3, y_arr_idx, x_arr_idx]
            out_pd.loc[row_idx, "reflectance_std"] = np.nanstd(reflectances[0:3, y_arr_idx, x_arr_idx], 0)
            out_pd.loc[row_idx, "ndvi"] = ndvi[y_arr_idx, x_arr_idx]
            out_pd.loc[row_idx, "red_blue_ratio"] = ratios[0, y_arr_idx, x_arr_idx]
            out_pd.loc[row_idx, "green_red_ratio"] = ratios[1, y_arr_idx, x_arr_idx]
            out_pd.loc[row_idx, "blue_red_ratio"] = ratios[2, y_arr_idx, x_arr_idx]
            out_pd.loc[row_idx, "green_blue_ratio"] = ratios[3, y_arr_idx, x_arr_idx]
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


if __name__ == "__main__":
    if do_tuning:
        n_detections = pd.DataFrame()
        coeffs = np.arange(50, 600, 100)
        n_detections["coefficients"] = coeffs
        for tile in tiles_pd["training_tiles"]:
            rf_td = RFTruckDetector()
            imgs = np.array(glob(dirs["imgs"] + os.sep + "*" + tile + "*.tif"))
            lens = np.int32([len(x) for x in imgs])
            bands = rf_td.read_bands(imgs[np.where(lens == lens.min())[0]][0])
            bands_preprocessed = rf_td.preprocess_bands(bands, dirs["osm"])
            for idx, c in enumerate(coeffs):
                rf_td.train(bands_preprocessed, truth_csv, c)
                boxes = rf_td.extract_objects(rf_td.predict())
                n_detections.loc[idx, tile] = len(boxes)
        n_detections.to_csv(os.path.join(dirs["validation"], "tuning.csv"))
    else:
        rf_td = RFTruckDetector()
        bands = rf_td.read_bands(s2_file)
        bands_preprocessed = rf_td.preprocess_bands(bands, dirs["osm"])
        rf_td.train(bands_preprocessed, truth_csv, 10)
        predictions = rf_td.predict()
        boxes = rf_td.extract_objects(predictions)
        print(len(boxes))
        try:
            name = os.path.basename(s2_file).split("bands_")[1].split("_merged")[0]
        except IndexError:
            name = os.path.basename(s2_file).split(".")[0]
        rf_td.prediction_raster_to_gtiff(predictions, os.path.join(dirs["main"], name + "_raster.tiff"))
        rf_td.prediction_boxes_to_gpkg(boxes, os.path.join(dirs["main"], name + "_boxes.gpkg"))
