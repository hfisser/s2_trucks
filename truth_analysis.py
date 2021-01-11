import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import random
import rasterio as rio
from sklearn.cluster import KMeans
from scipy.stats import linregress

from osm_utils.utils import get_roads, rasterize_osm
from array_utils.io import rio_read_all_bands
from array_utils.math import rescale, normalized_ratio
from array_utils.geocoding import lat_from_meta, lon_from_meta, metadata_to_bbox_epsg4326

dir_main = "F:\\Masterarbeit\\DLR\\project\\1_truck_detection"
dir_imgs = os.path.join(dir_main, "data", "s2", "subsets")
dir_truth = os.path.join(dir_main, "truth")
dir_truth_labels = os.path.join(dir_main, "data", "labels")
dir_osm = os.path.join(dir_main, "code", "detect_trucks", "AUXILIARY")

tiles_pd = pd.read_csv(os.path.join(dir_main, "training", "tiles.csv"), sep=";")
tiles = list(tiles_pd["training_tiles"])
n_clusters = 50  # number of RGB vector clusters

overwrite_truth_csv = True
training_percentage = 80

OSM_BUFFER = 40


def extract_statistics(img_file, boxes_gpd, n_retain, truth_csv, spectra_csv, spectra_ml_csv):
    truth = pd.read_csv(truth_csv, index_col=0)
    spectra = pd.read_csv(spectra_csv, index_col=0)
    spectra_ml = pd.read_csv(spectra_ml_csv, index_col=0)
    arr, meta = rio_read_all_bands(img_file)
    osm_file = os.path.join(dir_osm, "osm%s" % os.path.basename(img_file))
    if os.path.exists(osm_file):
        with rio.open(osm_file, "r") as src:
            osm_mask = src.read(1)
    else:
        lat, lon = lat_from_meta(meta), lon_from_meta(meta)
        bbox_epsg4326 = list(np.flip(metadata_to_bbox_epsg4326(meta)))
        osm_mask = get_osm_mask(bbox_epsg4326, meta["crs"], arr[0], {"lat": lat, "lon": lon},
                                dir_osm)
        meta["count"] = 1
        meta["dtype"] = osm_mask.dtype
        with rio.open(osm_file, "w", **meta) as tgt:
            tgt.write(osm_mask, 1)
    arr *= osm_mask
    arr[np.isnan(arr)] = 0.
    arr = rescale(arr, 0, 1)
    arr[arr == 0] = np.nan
    ndvi = normalized_ratio(arr[3], arr[0])
    n_bands = 3
    ratios = np.zeros((n_bands + 1, arr.shape[1], arr.shape[2]))
    ratio_counterparts = [2, 0, 0]
    for band_idx in range(n_bands):
        ratios[band_idx] = normalized_ratio(arr[band_idx], arr[ratio_counterparts[band_idx]])
    ratios[3] = normalized_ratio(arr[1], arr[2])  # add green vs. blue
    ratios_diffs = expose_anomalous_pixels(ratios[0:3])
    lat, lon = lat_from_meta(meta), lon_from_meta(meta)
    # shift lat lon to pixel center
    lat_shifted, lon_shifted = shift_lat(lat, 0.5), shift_lon(lon, 0.5)
    for row_idx in np.random.choice(boxes_gpd.index, int(np.clip(len(boxes_gpd) - n_retain, 0, 1e+10)), replace=False):
        boxes_gpd.drop(row_idx, inplace=True)
    boxes_gpd.index = range(len(boxes_gpd))
    print(len(boxes_gpd))
    n = len(boxes_gpd)
    for i in range(n):
        box = boxes_gpd.geometry[i].bounds
        x0, x1 = get_smallest_deviation(lon_shifted, box[0]), get_smallest_deviation(lon_shifted, box[2])
        y1, y0 = get_smallest_deviation(lat_shifted, box[1]), get_smallest_deviation(lat_shifted, box[3])
        sub_arr = arr[0:4, y0:y1 + 1, x0:x1 + 1].copy()
        sub_ratios = ratios[:, y0:y1 + 1, x0:x1 + 1].copy()
        spectra_ml = extract_rgb_spectra(spectra_ml, sub_arr, sub_ratios, ndvi[y0:y1 + 1, x0:x1 + 1])
        arr[:, y0:y1 + 1, x0:x1 + 1] = np.nan  # mask out box reflectances in order to avoid using them as background
        ratios[:, y0:y1 + 1, x0:x1 + 1] = np.nan
    spectra_ml = add_background(spectra_ml, arr, ratios, ndvi, len(boxes_gpd))
    #print("Number of truth features in csv: %s" % (str(len(truth))))
   # truth.to_csv(truth_csv)
   # spectra.to_csv(spectra_csv)
    spectra_ml.to_csv(spectra_ml_csv)


def analyze_statistics(truth_csv, spectra_csv):
    truth = pd.read_csv(truth_csv, index_col=0)
    bg_thresholds = calc_ratio_thresholds(np.float32(truth.max_bg_ratio))
    br_thresholds = calc_ratio_thresholds(np.float32(truth.max_br_ratio))
    rb_thresholds = calc_ratio_thresholds1(np.float32(truth.min_rb_ratio), np.float32(truth.max_rb_ratio))
    gb_thresholds = calc_ratio_thresholds1(np.float32(truth.min_gb_ratio), np.float32(truth.max_gb_ratio))
    ndvi_thresholds = calc_ndvi_thresholds(np.float32(truth.max_ndvi))
    max_distance_green = truth.max_dist_green
    max_distance_red = truth.max_dist_red
    max_max_distance_green = max_distance_green.max()
    max_max_distance_red = max_distance_red.max()
    min_max_distance_green = max_distance_green.min()
    min_max_distance_red = max_distance_red.min()
    mean_max_distance_green = max_distance_green.mean()
    mean_max_distance_red = max_distance_red.mean()
    thresholds = pd.DataFrame({"box_mean_red_high": [float(np.nanquantile(truth.mean_red, [0.98]))],
                               "box_mean_green_high": [float(np.nanquantile(truth.mean_green, [0.98]))],
                               "box_mean_blue_high": [float(np.nanquantile(truth.mean_blue, [0.98]))],
                               "blue_low": [float(truth.mean_blue.min())],
                               "green_low": [float(truth.mean_green.min())],
                               "red_low": [float(truth.mean_red.min())],
                               "blue_high": [float(np.quantile(truth.max_blue, q=[0.98]))],
                               "green_high": [float(np.quantile(truth.max_green, q=[0.98]))],
                               "red_high": [float(np.quantile(truth.max_red, q=[0.98]))],
                               "bg_low": [float(bg_thresholds["ratio_low"])],
                               "bg_high": [float(bg_thresholds["ratio_high"])],
                               "bg_mean": [np.nanmean(truth.mean_bg_ratio)],
                               "bg_std_mean": [np.nanstd(truth.mean_bg_ratio)],
                               "br_low": [float(br_thresholds["ratio_low"])],
                               "br_high": [float(br_thresholds["ratio_high"])],
                               "br_mean": [np.nanmean(truth.mean_br_ratio)],
                               "br_std_mean": [np.nanstd(truth.mean_br_ratio)],
                               "rb_low": [float(rb_thresholds["ratio_low"])],
                               "rb_high": [float(rb_thresholds["ratio_high"])],
                               "gb_low": [float(gb_thresholds["ratio_low"])],
                               "gb_high": [float(gb_thresholds["ratio_high"])],
                               "ndvi_mean": [float(np.mean(truth.mean_ndvi))],
                               "ndvi_std": [float(np.std(truth.mean_ndvi))],
                               "max_max_dist_green": [float(max_max_distance_green)],
                               "max_max_dist_red": [float(max_max_distance_red)],
                               "min_max_dist_green": [float(min_max_distance_green)],
                               "min_max_dist_red": [float(min_max_distance_red)],
                               "mean_max_dist_green": [float(mean_max_distance_green)],
                               "mean_max_dist_red": [float(mean_max_distance_red)],
                               "red_green_spatial_angle_low": [float(np.nanquantile(truth.red_green_spatial_angle,
                                                                                    q=[0.005]))],
                               "red_green_spatial_angle_high": [float(np.nanquantile(truth.red_green_spatial_angle,
                                                                                     q=[0.995]))],
                               "min_red_green_spatial_angle": [float(np.nanmin(truth.red_green_spatial_angle))],
                               "max_red_green_spatial_angle": [float(np.nanmax(truth.red_green_spatial_angle))],
                               "mean_red_green_spatial_angle": [float(np.nanmean(truth.red_green_spatial_angle))],
                               "std_red_green_spatial_angle": [float(np.nanstd(truth.red_green_spatial_angle))],
                               "mean_skewness": [float(np.nanmean(truth["skewness"]))],
                               "std_skewness": [float(np.nanstd(truth["skewness"]))],
                               "mean_kurtosis": [float(np.nanmean(truth["kurtosis"]))],
                               "std_kurtosis": [float(np.nanstd(truth["kurtosis"]))],
                               "mean_std": [float(np.nanmean(truth["std"]))],
                               "std_std": [float(np.nanstd(truth["std"]))],
                               "min_std": [float(np.nanmin(truth["std"]))],
                               "max_std": [float(np.nanmax(truth["std"]))],
                               "min_std_at_max_blue": [float(np.nanmin(truth["std_at_max_blue"]))],
                               "mean_std_at_max_blue": [float(np.nanmean(truth["std_at_max_blue"]))],
                               "q1_std_at_max_blue": [float(np.nanquantile(truth["std_at_max_blue"], [0.005]))],
                               "q99_std_at_max_blue": [float(np.nanquantile(truth["std_at_max_blue"], [0.995]))],
                               "q1_mean_var": [float(np.nanquantile(truth["mean_var"], [0.005]))],
                               "mean_std_var": [float(np.nanmean(truth["std_var"]))]})
    print(thresholds)
    thresholds.astype(np.float64).to_csv(os.path.join(os.path.dirname(truth_csv), "thresholds.csv"))
    #cluster_spectra(spectra_csv, 10)


def get_indices(a, value, b=None):
    indices = np.array(np.where(a == value))
    if len(indices.flatten()) > 2:  # criteria matched by several values
        try:
            difference = np.array([np.abs(indices[:, i] - b) for i in range(indices.shape[1])])
            indices = indices[:, np.where(difference == difference.max())[0][0]]  # max distance from b
        except TypeError:  # b not given
            indices = indices[:, 0]  # take first match
    return indices.flatten()


def calc_blue_thresholds(maximum_blue):
    return {"blue_low": np.nanquantile(maximum_blue, q=[0.01]),
            "blue_high": np.nanquantile(maximum_blue, q=[0.99])}


def calc_green_thresholds(maximum_green):
    return {"green_low": np.nanquantile(maximum_green, q=[0.01]),
            "green_high": np.nanquantile(maximum_green, q=[0.99])}


def calc_red_thresholds(maximum_red):
    return {"red_low": np.nanquantile(maximum_red, q=[0.01]),
            "red_high": np.nanquantile(maximum_red, q=[0.99])}


def calc_ratio_thresholds(max_ratio):
    return {"ratio_low": np.nanquantile(max_ratio, [0.01]),
            "ratio_high": np.nanquantile(max_ratio, [0.97])}


def calc_ratio_thresholds1(min_ratio, max_ratio):
    return {"ratio_low": np.nanquantile(min_ratio, [0.99]),
            "ratio_high": np.nanquantile(max_ratio, [0.99])}


def calc_ndvi_thresholds(maximum_ndvi):
    return {"ndvi_high": np.nanquantile(maximum_ndvi, q=[0.70])}


def calc_angles(stack, ratios, detections):
    # spatio-spectral test
    red, green, blue = stack[0].copy(), stack[1].copy(), stack[2].copy()
    green_criteria = np.nansum(ratios[2:4], 0)
    red_criteria = np.nansum(ratios[0:2], 0)
    try:
        blue_y, blue_x = [index[0] for index in np.where(detections == 1)]
        green[blue_y, blue_x] = 0
        green_y, green_x = crop_2d_indices(np.where(green_criteria == np.nanmax(green_criteria)))
        red[blue_y, blue_x] = 0
        red[green_y, green_x] = 0
        red_y, red_x = crop_2d_indices(np.where(red_criteria == np.nanmax(red_criteria)))
    except ValueError:
        return
    rgb_vector = np.hstack([ratios[4:6, blue_y, blue_x],
                            ratios[2:4, green_y, green_x],
                            ratios[0:2, red_y, red_x],
                            ratios[6, blue_y, blue_x],
                            ratios[6, green_y, green_x],
                            ratios[6, red_y, red_x],
                            ratios[7, blue_y, blue_x],
                            ratios[7, green_y, green_x],
                            ratios[7, red_y, red_x],
                            stack[2, blue_y, blue_x],
                            stack[2, green_y, green_x],
                            stack[2, red_y, red_x],
                            stack[1, green_y, green_x],
                            stack[1, blue_y, blue_x],
                            stack[1, red_y, red_x],
                            stack[0, red_y, red_x],
                            stack[0, blue_y, blue_x],
                            stack[0, green_y, green_x]])
    # spatial 2d vectors
    blue_max_indices = [blue_y, blue_x]
    blue_red_spatial_vector = calc_vector(blue_max_indices, [red_y, red_x])
    blue_green_spatial_vector = calc_vector(blue_max_indices, [green_y, green_x])
    red_green_spatial_angle = calc_vector_angle_in_degrees(blue_red_spatial_vector, blue_green_spatial_vector)
    return {"rgb_vector": rgb_vector,
            "red_green_spatial_angle": red_green_spatial_angle}


def cluster_rgb_vectors(truth_csv, thresholds_csv, rgb_vectors_csv, num_clusters):
    truth = pd.read_csv(truth_csv, index_col=0)
    thresholds = pd.read_csv(thresholds_csv, index_col=0)
    # vectors n*9
    col_names = []
    for i in range(7):
        col_names = col_names + ["rgb_vector" + str(i) + str(j) for j in [0, 1, 2]]
    vectors = np.vstack([truth[col] for col in col_names]).swapaxes(0, 1)
    # cluster with KMeans
    k_means = KMeans(n_clusters=n_clusters).fit(vectors)
    # calculate mean of clusters
    cluster_cores = np.zeros((num_clusters, vectors.shape[1]))
    labels = k_means.labels_
    for i, label in enumerate(np.unique(labels)):
        label_match = np.where(labels == label)[0]
        cluster_cores[i] = np.nanmean(vectors[label_match, :], axis=0)
    # calculate all angles between clusters
    rsquared = np.zeros(n_clusters)
    slopes = np.zeros(n_clusters)
    for i in range(n_clusters):
        angles_cluster = []
        slopes_cluster = []
        for j in range(n_clusters):
            if j != i:
                regression = linregress(cluster_cores[i], cluster_cores[j])
                angles_cluster.append(regression.rvalue)
                slopes_cluster.append(regression.slope)
        rsquared[i] = np.nanmean(np.array(angles_cluster))  # mean angle
        slopes[i] = np.nanmean(slopes_cluster)
    thresholds["min_rgb_rsquared"] = [float(np.nanmin(rsquared))]
    thresholds["max_rgb_rsquared"] = [float(np.nanmax(rsquared))]
    thresholds["mean_rgb_rsquared"] = [float(np.nanmean(rsquared))]
    thresholds["std_rgb_rsquared"] = [float(np.nanstd(rsquared))]
    thresholds["rgb_rsquared_low"] = [float(np.nanmin(rsquared))]
    thresholds["rgb_rsquared_high"] = [float(np.nanmax(rsquared))]
    thresholds["mean_slope"] = [float(np.nanmean(slopes))]
    thresholds["max_slope"] = [float(np.nanmax(slopes))]
    thresholds["min_slope"] = [float(np.nanmin(slopes))]
    thresholds["std_slope"] = [float(np.nanstd(slopes))]
    thresholds.to_csv(thresholds_csv)
    rgb_vectors_pd = pd.DataFrame()
    for i, idx in enumerate(col_names):
        rgb_vectors_pd[idx] = cluster_cores[:, i]
    rgb_vectors_pd.to_csv(rgb_vectors_csv)


def cluster_spectra(spectra_csv, num_clusters):
    spectra = pd.read_csv(spectra_csv)
    vectors = np.vstack([spectra[col] for col in spectra.columns[1:]]).swapaxes(0, 1)
    # cluster with KMeans
    k_means = KMeans(n_clusters=n_clusters).fit(vectors)
    # calculate mean of clusters
    cluster_cores = np.zeros((num_clusters, vectors.shape[1]))
    labels = k_means.labels_
    for i, label in enumerate(np.unique(labels)):
        label_match = np.where(labels == label)[0]
        cluster_cores[i] = np.nanmean(vectors[label_match, :], axis=0)
    spectra_cores = pd.DataFrame()
    spectra_cores["B08"] = cluster_cores[:, 0]
    spectra_cores["B02"] = cluster_cores[:, 1]
    spectra_cores["B03"] = cluster_cores[:, 2]
    spectra_cores["B04"] = cluster_cores[:, 3]
    spectra_cores.to_csv(os.path.join(os.path.dirname(spectra_csv), "spectra_cores.csv"))


def crop_2d_indices(indices):
    """
    :param indices: tuple of np int64 indices as returned by np.where
    :return: np int32 indices. Cropped if longer than 1
    """
    return np.array([index_arr[0] for index_arr in indices]).astype(np.int32)


def get_max_index_slice(arr, index_2d):
    """
    :param arr: np 3d array
    :param index_2d: np int 2d indices of length 2 (y,x)
    :return: np int32 indices of length 3 (z,y,x) pointing to
    """
    max_slice = arr[:, index_2d[0], index_2d[1]]
    z_index = np.where(max_slice == max_slice.max())
    return np.array([z_index[0][0], index_2d[0], index_2d[1]]).astype(np.int32)


def calc_vector_angle_in_degrees(a, b):
    cos = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
    if np.abs(cos) >= 1:
        return 0
    else:
        return np.degrees(np.arccos(cos))


def calc_vector_length(vector):
    """
    :param vector: np array vector
    :return: np float32
    """
    squared = np.float32([element**2 for element in vector])
    return np.sqrt(squared.sum()).astype(np.float32)


def calc_vector(b, a):
    """
    :param b: 1d np.float32 array or array-like
    :param a: 1d np.float32 array or array-like
    :return: 2d np.float32 array, a vector pointing to origin
    """
    vector = []
    for i in range(len(b)):
        try:
            vector.append(np.float32(b[i] - a[i]))
        except IndexError:
            raise IndexError("origin and target must be of equal length")
    return np.array(vector).astype(np.float32)


def create_truth_csv(file_path):
    truth_pd = pd.DataFrame(columns=["image_file", "box_number",
                                     "max_red", "min_red",
                                     "max_green", "min_green",
                                     "max_blue", "min_blue",
                                     "min_br_ratio", "max_br_ratio",
                                     "min_bg_ratio", "max_bg_ratio",
                                     "min_ndvi", "max_ndvi",
                                     "max_dist_green", "max_dist_red",
                                     "blue_red_angle", "blue_green_angle", "red_green_angle",
                                     "blue_red_length", "blue_green_length"])
    truth_pd.to_csv(file_path)


def shift_lat(lat, offset):
    return lat - (np.abs(lat[1] - lat[0])) / (1 / offset)


def shift_lon(lon, offset):
    return lon + (np.abs(lon[1] - lon[0])) / (1 / offset)


def get_smallest_deviation(a, value):
    dev = np.abs(a - value)
    return int(np.where(dev == dev.min())[0][0])


def extract_rgb_spectra(spectra_ml_pd, sub_reflectances, sub_ratios, ndvi):
    sub_copy = sub_reflectances.copy() * 10
    sub_ratios_copy = sub_ratios.copy()
    red_criteria = sub_copy[0] + sub_ratios_copy[0]
    red_y, red_x = np.where(red_criteria == np.nanmax(red_criteria))
    try:
        sub_copy[:, red_y[0], red_x[0]] = np.nan  # avoid double of pixel
        sub_ratios_copy[:, red_y[0], red_x[0]] = np.nan
        green_criteria = sub_copy[1] + sub_ratios_copy[1]
        green_y, green_x = np.where(green_criteria == np.nanmax(green_criteria))
        sub_copy[:, green_y[0], green_x[0]] = np.nan
        sub_ratios_copy[:, green_y[0], green_x[0]] = np.nan
        blue_criteria = sub_copy[2] + sub_ratios_copy[2]
        blue_y, blue_x = np.where(blue_criteria == np.nanmax(blue_criteria))
    except IndexError:
        return spectra_ml_pd
    if len(red_y) == 0 or len(green_y) == 0 or len(blue_y) == 0:
        return spectra_ml_pd
    for label, label_int, y, x in zip(("red", "green", "blue"),
                                      (4, 3, 2),
                                      [red_y, green_y, blue_y],
                                      [red_x, green_x, blue_x]):
        row_idx = len(spectra_ml_pd)
        y, x = y[0], x[0]
        spectra_ml_pd.loc[row_idx, "label"] = label
        spectra_ml_pd.loc[row_idx, "label_int"] = label_int
        spectra_ml_pd.loc[row_idx, "red"] = sub_reflectances[0, y, x]
        spectra_ml_pd.loc[row_idx, "green"] = sub_reflectances[1, y, x]
        spectra_ml_pd.loc[row_idx, "blue"] = sub_reflectances[2, y, x]
        spectra_ml_pd.loc[row_idx, "nir"] = sub_reflectances[3, y, x]
        spectra_ml_pd.loc[row_idx, "reflectance_std"] = np.nanstd(sub_reflectances[0:3, y, x], 0)
        spectra_ml_pd.loc[row_idx, "ndvi"] = ndvi[y, x]
        spectra_ml_pd.loc[row_idx, "red_blue_ratio"] = sub_ratios[0, y, x]
        spectra_ml_pd.loc[row_idx, "green_red_ratio"] = sub_ratios[1, y, x]
        spectra_ml_pd.loc[row_idx, "blue_red_ratio"] = sub_ratios[2, y, x]
        spectra_ml_pd.loc[row_idx, "green_blue_ratio"] = sub_ratios[3, y, x]
    return spectra_ml_pd


def add_background(out_pd, reflectances, ratios, ndvi, n_background):
    # pick random indices from non nans
    not_nan_reflectances = np.int8(~np.isnan(reflectances[0:4]))
    not_nan_ndvi = np.int8(~np.isnan(ndvi))
    not_nan_ratios = np.int8(~np.isnan(ratios))
    not_nan = np.min(not_nan_reflectances, 0) * not_nan_ndvi * np.min(not_nan_ratios, 0)
    not_nan_y, not_nan_x = np.where(not_nan == 1)
    random_indices = np.random.randint(0, len(not_nan_y), n_background)
    for random_idx in zip(random_indices):
        y_arr_idx, x_arr_idx = not_nan_y[random_idx], not_nan_x[random_idx]
        row_idx = len(out_pd)
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
    return out_pd


def get_osm_mask(bbox, crs, reference_arr, lat_lon_dict, dir_out):
    osm_file = get_roads(bbox, ["motorway", "trunk", "primary"], OSM_BUFFER,
                         dir_out, str(bbox).replace(", ", "_").replace("-", "minus")[1:-1] + "_osm_roads", str(crs),
                         reference_arr)
    osm_vec = gpd.read_file(osm_file)
    ref_xr = xr.DataArray(data=reference_arr, coords=lat_lon_dict, dims=["lat", "lon"])
    osm_raster = rasterize_osm(osm_vec, ref_xr).astype(np.float32)
    osm_raster[osm_raster != 0] = 1
    osm_raster[osm_raster == 0] = np.nan
    return osm_raster


def expose_anomalous_pixels(band_stack_np):
    w = 1000
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
    diff_red = band_stack_np[0] - (roads[0] / 2)
    diff_green = band_stack_np[1] - (roads[1] / 2)
    diff_blue = band_stack_np[2] - (roads[2] / 2)
    diff_stack = np.array([diff_red, diff_green, diff_blue])
    return diff_stack


if __name__ == "__main__":
    if not os.path.exists(dir_truth):
        os.mkdir(dir_truth)
    file_path_truth = os.path.join(dir_truth, "truth_analysis.csv")
    file_path_spectra = os.path.join(dir_truth, "spectra.csv")
    file_path_spectra_ml = os.path.join(dir_truth, "spectra_ml.csv")
    if os.path.exists(file_path_truth) and overwrite_truth_csv:
        os.remove(file_path_truth)
    if os.path.exists(file_path_spectra) and overwrite_truth_csv:
        os.remove(file_path_spectra)
    if not os.path.exists(file_path_truth):
        create_truth_csv(file_path_truth)
    if not os.path.exists(file_path_spectra):
        spectra_pd = pd.DataFrame()
        spectra_pd.to_csv(file_path_spectra)
    spectra_ml_pd = pd.DataFrame()
    spectra_ml_pd.to_csv(file_path_spectra_ml)
    for tile in tiles:
        print(tile)
        imgs = np.array(glob.glob(dir_imgs + os.sep + "*" + tile + "*.tif"))
        lens = np.int32([len(x) for x in imgs])
        img_file = imgs[np.where(lens == lens.min())[0]][0]
        boxes_truth = gpd.read_file(glob.glob(dir_truth_labels + os.sep + "*" + tile + "*.gpkg")[0])
  #      n_boxes = len(boxes)
   #     n_training = np.int32(np.round(n_boxes * (training_percentage / 100)))
    #    n_validation = np.int32(np.round(n_boxes * (1 - (training_percentage / 100))))
     #   boxes_range = list(range(n_boxes))
      #  indices_training = random.sample(range(n_boxes), k=n_training)
   #     for idx in indices_training:
    #        del boxes_range[boxes_range.index(idx)]
     #   indices_validation = boxes_range
      #  boxes_training = boxes.iloc[indices_training]
       # boxes_validation = boxes.iloc[indices_validation]
        # save validation boxes
       # boxes_validation.index = range(len(boxes_validation))
        #boxes_validation.to_file(os.path.join(dir_truth_labels,
         #                                     os.path.basename(img_file).split(".tif")[0] + "_validation.gpkg"),
             #                    driver="GPKG")
        # extract stats from training boxes
       # boxes_training.index = range(len(boxes_training))
        extract_statistics(img_file, boxes_truth, int(tiles_pd[tiles_pd["training_tiles"] == tile]["n_retain"]),
                           file_path_truth, file_path_spectra, file_path_spectra_ml)
  #  analyze_statistics(file_path_truth, file_path_spectra)
  #  cluster_rgb_vectors(file_path_truth, os.path.join(dir_truth, "thresholds.csv"),
   #                     os.path.join(dir_truth, "rgb_vector_clusters.csv"), n_clusters)

