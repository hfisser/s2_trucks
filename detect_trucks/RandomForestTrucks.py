import os
import warnings
import pickle
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio as rio
import xarray as xr
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
from fiona import errors
from shapely.geometry import Polygon, box
from rasterio.transform import Affine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from detect_trucks.ObjectExtractor import ObjectExtractor
from detect_trucks.IO import IO
from array_utils.math import rescale, normalized_ratio
from array_utils.geocoding import lat_from_meta, lon_from_meta, metadata_to_bbox_epsg4326
from osm_utils.utils import get_roads, rasterize_osm

warnings.filterwarnings("ignore")

dirs = dict(main="F:\\Masterarbeit\\DLR\\project\\1_truck_detection")
dirs["truth"] = os.path.join(dirs["main"], "truth")
dirs["s2_data"] = os.path.join(dirs["main"], "validation", "data", "s2", "archive")
dirs["osm"] = os.path.join(dirs["main"], "code", "detect_trucks", "AUXILIARY", "osm")
dirs["imgs"] = os.path.join(dirs["main"], "data", "s2", "subsets")
tiles_pd = pd.read_csv(os.path.join(dirs["main"], "training", "tiles.csv"), sep=";")

s2_files = [os.path.join(dirs["main"], "data", "s2", "subsets", f) for f in
            ["S2A_MSIL2A_20200922T100031_N0214_R122_T33UWP_20200924T114821_y0_x0.tif",
             "S2B_MSIL2A_20200828T100029_N0214_R122_T33UWP_20200828T120923_y0_x0.tif",
             "S2B_MSIL2A_20201106T100219_N0214_R122_T33UWP_20201106T121510_y0_x0.tif"]]

do_tuning = False
truth_path_training = os.path.join(dirs["truth"], "spectra_ml_training_tiles.csv")
truth_path_validation = os.path.join(dirs["truth"], "spectra_ml_validation_tiles.csv")

OSM_BUFFER = 20
SECONDS_OFFSET_B02_B04 = 1.01  # sensing offset between B02 and B04

# RF hyper parameters from hyper parameter tuning
N_ESTIMATORS = 1600
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 1
MAX_FEATURES = "sqrt"
MAX_DEPTH = 20
BOOTSTRAP = True

MODEL_PATH = os.path.join(os.path.dirname(__file__), "rf_model.pickle")


class RFTruckDetector:
    def __init__(self):
        self.t0 = datetime.now()
        self.rf = None
        self.io = IO(self)
        self.lat, self.lon, self.meta = None, None, None
        self.variables = None
        self.truth_variables, self.truth_labels = None, None
        self.truth_path_tmp = None
        self.var_mask_blue, self.var_mask_green, self.var_mask_red = None, None, None
        self.probabilities = None
        self.train()

    def train(self):
        try:
            rf = self.io.read_model(MODEL_PATH)
        except FileNotFoundError:
            rf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                        min_samples_split=MIN_SAMPLES_SPLIT,
                                        min_samples_leaf=MIN_SAMPLES_LEAF,
                                        max_features=MAX_FEATURES,
                                        max_depth=MAX_DEPTH,
                                        bootstrap=BOOTSTRAP,
                                        oob_score=True)
            self._load_truth()
            rf.fit(self.truth_variables["train"], self.truth_labels["train"])
            self.io.write_model(rf, MODEL_PATH)
        self.rf = rf

    def tune(self, tuning_path):
        self._load_truth()
        tuning_pd = pd.DataFrame()
        n_trees = list(np.int16(np.linspace(start=200, stop=2000, num=20)))
        max_features = ["auto", "sqrt"]
        max_depth = list(np.int16(np.linspace(10, 150, num=15)))
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
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=hyper_hyper, n_iter=200, cv=3, verbose=2,
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
            self.meta["height"], self.meta["width"] = bands_rescaled.shape[1], bands_rescaled.shape[2]
            t = list(self.meta["transform"])
            t[2], t[5] = self.lon[0], self.lat[0]
            self.meta["transform"] = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
        bbox_epsg4326 = list(np.flip(metadata_to_bbox_epsg4326(self.meta)))
        osm_mask = self._get_osm_mask(bbox_epsg4326, self.meta["crs"], bands_rescaled[0], {"lat": self.lat,
                                                                                           "lon": self.lon},
                                      dirs["osm"])
        if np.count_nonzero(osm_mask) == 0:
            raise ValueError("No OSM roads of requested road types in aoi")
        bands_rescaled *= osm_mask
        bands_rescaled[bands_rescaled == 0] = np.nan
        osm_mask = None
        self._build_variables(bands_rescaled)

    def mask_clouds(self, cloud_mask):
        cloud_mask = cloud_mask.astype(np.float32)
        cloud_mask[cloud_mask == 0] = np.nan
        self.variables[:] *= cloud_mask

    def predict(self):
        if self.rf is None:
            self.rf = self.io.read_model(MODEL_PATH)
        print("-" * 20 + "\nPredicting")
        t0 = datetime.now()
        vars_reshaped = []
        for band_idx in range(self.variables.shape[0]):
            vars_reshaped.append(self.variables[band_idx].flatten())
        vars_reshaped = np.array(vars_reshaped).swapaxes(0, 1)  # (n observations, n variables)
        nan_mask = np.zeros_like(vars_reshaped)
        for var_idx in range(vars_reshaped.shape[1]):
            nan_mask[:, var_idx] = ~np.isnan(vars_reshaped[:, var_idx])  # exclude nans
        not_nan = np.nanmin(nan_mask, 1).astype(np.bool) * np.min(np.isfinite(vars_reshaped), 1).astype(np.bool)
        predictions_flat = self.rf.predict_proba(vars_reshaped[not_nan])
        probabilities_shaped = vars_reshaped[:, 0:4].copy()
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
        self._elapsed(t0)
        return classification

    def extract_objects(self, predictions_arr):
        print("Extracting objects")
        extractor = ObjectExtractor(self)
        out_gpd = extractor.extract(predictions_arr)
        self._elapsed(self.t0)
        print("-" * 20)
        return out_gpd

    def _load_truth(self):
        truth_training = pd.read_csv(truth_path_training)
        indices = np.where(truth_training["label_int"] == 1)[0]
        for idx in np.random.choice(indices, int(len(indices) * 0.66), replace=False):
            truth_training.drop(idx, inplace=True)
        truth_validation = pd.read_csv(truth_path_validation)
        labels_training, variables_training = truth_training["label_int"], []
        labels_validation, variables_validation = truth_validation["label_int"], []
        for key in ["reflectance_var", "red_blue_ratio", "green_blue_ratio",
                    "red_normalized", "green_normalized", "blue_normalized", "nir_normalized"]:
            variables_training.append(truth_training[key])
            variables_validation.append(truth_validation[key])
        self.truth_variables = dict(train=np.float32(variables_training).swapaxes(0, 1),
                                    test=np.float32(variables_validation).swapaxes(0, 1))
        self.truth_labels = dict(train=labels_training, test=labels_validation)

    def _build_variables(self, band_stack):
        shape = band_stack.shape
        for band_idx in range(shape[0]):
            band_stack[band_idx] -= np.nanmean(band_stack[band_idx])
        variables = np.zeros((7, shape[1], shape[2]), dtype=np.float16)
        variables[0] = np.nanvar(band_stack[0:3], 0, dtype=np.float16)
        variables[1] = normalized_ratio(band_stack[0], band_stack[2]).astype(np.float16)  # red/blue
        variables[2] = normalized_ratio(band_stack[1], band_stack[2]).astype(np.float16)  # green/blue
        variables[3] = band_stack[0]
        variables[4] = band_stack[1]
        variables[5] = band_stack[2]
        variables[6] = band_stack[3]
        self.variables = variables.astype(np.float16)

    def _eliminate_clusters(self, arr):
        for value in [2, 3, 4]:  # eliminate large clusters of pixels
            arr_copy = arr.copy()
            ys, xs = np.where(arr_copy == value)
            for y, x in zip(ys, xs):
                subset_3 = self.get_arr_subset(arr, y, x, 3)
                n_matches = np.count_nonzero(subset_3 == value)
                single_blue = value == 2 and n_matches == 1 and np.count_nonzero(~np.isnan(subset_3)) == 0
                if n_matches > (subset_3.shape[0] * subset_3.shape[1]) * 0.5 or single_blue:
                    arr[y, x] = 0
        return arr

    def _postprocess_prediction(self):
  #      for idx, threshold in zip(range(self.probabilities.shape[0]), (0.6, 0.3, 0.3, 0.3)):
   #         self.probabilities[idx, self.probabilities[idx] < threshold] = 0
        classification = np.argmax(self.probabilities, 0) + 1
        return classification.astype(np.int8)

    def read_bands(self, path):
        # bands at path must be: Sentinel-2 0=B04, 1=B03, 2=B02, 3=B08
        return self.io.read_bands(path)

    def prediction_raster_to_gtiff(self, the_raster, path):
        self.io.prediction_raster_to_gtiff(the_raster, path)

    def prediction_boxes_to_gpkg(self, the_gpd, path):
        self.io.prediction_boxes_to_gpkg(the_gpd, path)

    def prediction_boxes_to_geojson(self, the_gpd, path):
        self.io.prediction_boxes_to_geojson(the_gpd, path)

    @staticmethod
    def get_arr_subset(arr, y, x, size):
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
    def eliminate_array_corners(arr, assign_value):
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
    def calc_speed(box_shape):
        diameter = np.max(box_shape) * 10 - 10  # 10 m resolution; -10 for center of pixel
        return diameter/SECONDS_OFFSET_B02_B04 * 3.6

    @staticmethod
    def calc_vector_direction_in_degree(vector):
        """
        :param vector: array-like y, x
        :return:
        """
        # [1,1] -> 45°; [-1,1] -> 135°; [-1,-1] -> 225°; [1,-1] -> 315°
        return np.degrees(np.arctan2(vector[1], vector[0])) % 360

    @staticmethod
    def direction_degree_to_description(direction_degree):
        step = 45
        bins = np.arange(0, 359, step, dtype=np.float32)
        descriptions = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
        diff = np.abs(bins - direction_degree)
        return descriptions[np.where(diff == np.min(diff))[0][0]]

    @staticmethod
    def _get_osm_mask(bbox, crs, reference_arr, lat_lon_dict, dir_out):
        osm_file = get_roads(bbox, ["motorway", "trunk", "primary"], OSM_BUFFER,
                             dir_out, str(bbox).replace(", ", "_").replace("-", "minus")[1:-1] + "_osm_roads", str(crs),
                             reference_arr)
        try:
            osm_vec = gpd.read_file(osm_file)
        except errors.DriverError:
            return np.zeros_like(reference_arr, dtype=np.float16)
        else:
            ref_xr = xr.DataArray(data=reference_arr, coords=lat_lon_dict, dims=["lat", "lon"])
            osm_raster = rasterize_osm(osm_vec, ref_xr).astype(np.float16)
            osm_raster[osm_raster != 0] = 1
            osm_raster[osm_raster == 0] = np.nan
            return osm_raster

    @staticmethod
    def _elapsed(start_time):
        print("Elapsed: %s s" % (datetime.now() - start_time).total_seconds())


if __name__ == "__main__":
    for s2_file in s2_files:
        rf_td = RFTruckDetector()
        bands = rf_td.read_bands(s2_file)
        if do_tuning:
            rf_td.tune(os.path.join(dirs["main"], "training", "hyper_parameter_tuning.csv"))
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
