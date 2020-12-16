import os
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio as rio
import xarray as xr
from array_utils.math import rescale, normalized_ratio
from array_utils.geocoding import lat_from_meta, lon_from_meta, metadata_to_bbox_epsg4326
from osm_utils.utils import get_roads, rasterize_osm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

dirs = dict(main="F:\\Masterarbeit\\DLR\\project\\1_truck_detection")
dirs["truth"] = os.path.join(dirs["main"], "truth")
dirs["s2_data"] = os.path.join(dirs["main"], "validation", "data", "s2", "archive")
dirs["osm"] = os.path.join(dirs["main"], "code", "detect_trucks", "AUXILIARY", "osm")
s2_file = os.path.join(dirs["s2_data"], "s2_bands_Salzbergen_2018-06-07_2018-06-07_merged.tiff")

truth_csv = os.path.join(dirs["truth"], "spectra_ml.csv")
number_trees = 500

OSM_BUFFER = 25


class RandomForestDetector:
    def __init__(self, truth_path):
        self.truth_data = pd.read_csv(truth_path, index_col=0)
        self.rf_model = None

    def train(self, n_trees):
        rf = RandomForestClassifier(n_estimators=n_trees, oob_score=True)
        self._prepare_truth()
        rf.fit(self.vars["train"], self.labels["train"])
        test_pred = rf.predict(self.vars["test"])
        accuracy = metrics.accuracy_score(self.labels["test"], test_pred)
        print(accuracy)
        self.rf_model = rf

    def preprocess_bands(self, band_stack, rio_meta, dir_osm):
        band_stack = band_stack[0:4]
        bands_rescaled = rescale(band_stack.copy(), 0, 1)
        band_stack = None
        lat, lon = lat_from_meta(rio_meta), lon_from_meta(rio_meta)
        bbox_epsg4326 = list(np.flip(metadata_to_bbox_epsg4326(rio_meta)))
        osm_mask = self.get_osm_mask(bbox_epsg4326, rio_meta["crs"], bands_rescaled[0], {"lat": lat, "lon": lon},
                                     dir_osm)
        bands_rescaled *= osm_mask
        bands_rescaled[bands_rescaled == 0] = np.nan
        return bands_rescaled

    def predict(self, band_stack):
        shape = band_stack.shape
        variables = np.zeros((shape[0] + 4, shape[1], shape[2]))
        for band_idx in range(shape[0]):
            variables[band_idx] = band_stack[band_idx]
        variables[-5] = np.nanstd(band_stack[0:3], 0)
        variables[-4] = normalized_ratio(band_stack[3], band_stack[0])
        variables[-3] = normalized_ratio(band_stack[0], band_stack[2])
        variables[-2] = normalized_ratio(band_stack[1], band_stack[0])
        variables[-1] = normalized_ratio(band_stack[2], band_stack[0])
        vars_reshaped = []
        for band_idx in range(variables.shape[0]):
            vars_reshaped.append(variables[band_idx].flatten())
        vars_reshaped = np.array(vars_reshaped).swapaxes(0, 1)  # (n observations, n variables)
        nan_mask = np.zeros_like(vars_reshaped)
        for idx in range(vars_reshaped.shape[1]):
            nan_mask[:, idx] = ~np.isnan(vars_reshaped[:, idx])  # exclude nans
        not_nan = np.nanmin(nan_mask, 1).astype(np.bool)
        prediction = self.rf_model.predict(vars_reshaped[not_nan])
        prediction_with_nans = vars_reshaped[:, 0].copy()
        prediction_with_nans[not_nan] = prediction
        prediction_shaped = prediction_with_nans.reshape((variables.shape[1], variables.shape[2]))
        return prediction_shaped

    def _prepare_truth(self):
        labels = self.truth_data["label_int"]
        variables = [self.truth_data["red"], self.truth_data["green"], self.truth_data["blue"],
                     self.truth_data["rgb_std"],
                     self.truth_data["ndvi"],
                     self.truth_data["red_blue_ratio"],
                     self.truth_data["green_red_ratio"],
                     self.truth_data["blue_red_ratio"]]
        variables = np.float64(variables).swapaxes(0, 1)
        vars_train, vars_test, labels_train, labels_test = train_test_split(variables, list(labels), test_size=0.15)
        self.vars = dict(train=vars_train, test=vars_test)
        self.labels = dict(train=labels_train, test=labels_test)

    @staticmethod
    def get_osm_mask(bbox, crs, reference_arr, lat_lon_dict, dir_out):
        osm_file = get_roads(bbox, ["motorway", "trunk", "primary"], OSM_BUFFER,
                             dir_out, str(bbox).replace(", ", "_")[1:-1] + "_osm_roads", str(crs),
                             reference_arr)
        osm_vec = gpd.read_file(osm_file)
        ref_xr = xr.DataArray(data=reference_arr, coords=lat_lon_dict, dims=["lat", "lon"])
        osm_raster = rasterize_osm(osm_vec, ref_xr).astype(np.float32)
        osm_raster[osm_raster != 0] = 1
        osm_raster[osm_raster == 0] = np.nan
        return osm_raster


if __name__ == "__main__":
    random_forest = RandomForestDetector(truth_csv)
    random_forest.train(number_trees)
    with rio.open(s2_file, "r") as src:
        meta = src.meta
        bands = np.zeros((src.count, src.height, src.width))
        for i in range(src.count):
            bands[i] = src.read(i + 1)
    bands_preprocessed = random_forest.preprocess_bands(bands, meta, dirs["osm"])
    predictions = random_forest.predict(bands_preprocessed)
    meta["count"] = 1
    meta["dtype"] = predictions.dtype
    with rio.open(os.path.join(dirs["main"], "test5.tiff"), "w", **meta) as tgt:
        tgt.write(predictions, 1)
