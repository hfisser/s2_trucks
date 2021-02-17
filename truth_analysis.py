import os
import glob
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio as rio
from osm_utils.utils import get_roads, rasterize_osm
from array_utils.io import rio_read_all_bands
from array_utils.math import rescale, normalized_ratio
from array_utils.geocoding import lat_from_meta, lon_from_meta, metadata_to_bbox_epsg4326

dir_main = "F:\\Masterarbeit\\DLR\\project\\1_truck_detection"
dir_imgs = os.path.join(dir_main, "data", "s2", "subsets")
dir_truth = os.path.join(dir_main, "truth")
dir_truth_labels = os.path.join(dir_main, "data", "labels")
dir_osm = os.path.join(dir_main, "code", "detect_trucks", "AUXILIARY", "osm")

tiles_pd = pd.read_csv(os.path.join(dir_main, "training", "tiles.csv"), sep=";")
tiles = list(tiles_pd["training_tiles"])

overwrite_truth_csv = True
training_percentage = 85

OSM_BUFFER = 30


def extract_statistics(image_file, boxes_gpd, n_retain, spectra_ml_csv):
    spectra_ml = pd.read_csv(spectra_ml_csv, index_col=0)
    arr, meta = rio_read_all_bands(image_file)
    osm_file = os.path.join(dir_osm, "osm%s" % os.path.basename(image_file))
#    if os.path.exists(osm_file):
 #       with rio.open(osm_file, "r") as src:
  #          osm_mask = src.read(1)
   # else:
    lat, lon = lat_from_meta(meta), lon_from_meta(meta)
    bbox_epsg4326 = list(np.flip(metadata_to_bbox_epsg4326(meta)))
    osm_mask = get_osm_mask(bbox_epsg4326, meta["crs"], arr[0], {"lat": lat, "lon": lon},
                            dir_osm)
    meta["count"] = 1
    meta["dtype"] = osm_mask.dtype
    with rio.open(osm_file, "w", **meta) as tgt:
        tgt.write(osm_mask, 1)
    arr *= osm_mask
    #arr[np.isnan(arr)] = 0.
   # arr = rescale(arr.copy(), 0, 1)
    #arr[arr == 0] = np.nan
    n_bands = 3
    ratios = np.zeros((n_bands + 1, arr.shape[1], arr.shape[2]))
    ratio_counterparts = [2, 0, 0]
    for band_idx in range(n_bands):
        ratios[band_idx] = normalized_ratio(arr[band_idx], arr[ratio_counterparts[band_idx]])
    ratios[3] = normalized_ratio(arr[1], arr[2])  # add green vs. blue
 #   reflectance_difference_stack = expose_anomalous_pixels(arr)
    lat, lon = lat_from_meta(meta), lon_from_meta(meta)
    # shift lat lon to pixel center
    lat_shifted, lon_shifted = shift_lat(lat, 0.5), shift_lon(lon, 0.5)
  #  for row_idx in np.random.choice(boxes_gpd.index, int(np.clip(len(boxes_gpd) - n_retain, 0, 1e+10)), replace=False):
 #       boxes_gpd.drop(row_idx, inplace=True)
#    boxes_gpd.index = range(len(boxes_gpd))
   # n_boxes = len(boxes_gpd)
  #  n_training = np.int32(np.round(n_boxes * (training_percentage / 100)))
 #   boxes_range = list(range(n_boxes))
#    indices_training = random.sample(range(n_boxes), k=n_training)
   # for idx in indices_training:
  #      del boxes_range[boxes_range.index(idx)]
 #   indices_validation = boxes_range
#    boxes_training = boxes_truth.iloc[indices_training]
   # boxes_validation = boxes_truth.iloc[indices_validation]
  #  # save validation boxes
 #   boxes_validation.index = range(len(boxes_validation))
#    boxes_validation.to_file(os.path.join(dir_truth_labels,
   #                                       os.path.basename(image_file).split(".tif")[0] + "_validation.gpkg"),
  #                           driver="GPKG")
 #   # extract stats from training boxes
#    boxes_training.index = range(len(boxes_training))
    boxes_training = boxes_gpd
    means_arr = [np.nanmean(arr[band_idx]) for band_idx in [0, 1, 2, 3]]
    for i in np.random.choice(list(range(len(boxes_training))), n_retain, replace=False):
        box = boxes_training.geometry[i].bounds
        x0, x1 = get_smallest_deviation(lon_shifted, box[0]), get_smallest_deviation(lon_shifted, box[2])
        y1, y0 = get_smallest_deviation(lat_shifted, box[1]), get_smallest_deviation(lat_shifted, box[3])
        sub_arr = arr[0:4, y0:y1 + 1, x0:x1 + 1].copy()
        sub_ratios = ratios[:, y0:y1 + 1, x0:x1 + 1].copy()
   #     sub_diffs = reflectance_difference_stack[:, y0:y1 + 1, x0:x1 + 1].copy()
        spectra_ml = extract_rgb_spectra(spectra_ml, sub_arr, sub_ratios, means_arr)
        arr[:, y0:y1 + 1, x0:x1 + 1] = np.nan  # mask out box reflectances in order to avoid using them as background
        ratios[:, y0:y1 + 1, x0:x1 + 1] = np.nan
    print("Number of training boxes: %s" % n_retain)
    # ensure equal number of blueish, greenish and reddish spectra
#    labels = ["red", "green", "blue"]
 #   n_given = [np.count_nonzero(spectra_ml["label"] == label) for label in labels]
  #  n_given_min = min(n_given)
   # labels.remove(labels[np.where(np.int16(n_given) == n_given_min)[0][0]])
    ## adjust labels in order to have equal number
    #for label in labels:
    #    indices = np.where(spectra_ml["label"] == label)[0]
    #    for row in np.random.choice(indices, len(indices) - n_given_min, replace=False):
    #        spectra_ml.drop(row, inplace=True)
    #    spectra_ml.index = range(len(spectra_ml))
    spectra_ml = add_background(spectra_ml, arr, ratios, means_arr, n_retain)
    spectra_ml.to_csv(spectra_ml_csv)


def get_indices(a, value, b=None):
    indices = np.array(np.where(a == value))
    if len(indices.flatten()) > 2:  # criteria matched by several values
        try:
            difference = np.array([np.abs(indices[:, i] - b) for i in range(indices.shape[1])])
            indices = indices[:, np.where(difference == difference.max())[0][0]]  # max distance from b
        except TypeError:  # b not given
            indices = indices[:, 0]  # take first match
    return indices.flatten()


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


def extract_rgb_spectra(t, sub_reflectances, sub_ratios, means):
    ndvi = normalized_ratio(sub_reflectances[3], sub_reflectances[0])
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
        return t
    if len(red_y) == 0 or len(green_y) == 0 or len(blue_y) == 0:
        return t
    for label, label_int, y, x in zip(("red", "green", "blue"),
                                      (4, 3, 2),
                                      [red_y, green_y, blue_y],
                                      [red_x, green_x, blue_x]):
        row_idx = len(t)
        y, x = y[0], x[0]
        stack = sub_reflectances[0:4, y, x]
        stack_normalized = sub_reflectances[0:4, y, x].copy() / means
        t.loc[row_idx, "label"] = label
        t.loc[row_idx, "label_int"] = label_int
        t.loc[row_idx, "red"] = stack[0]
        t.loc[row_idx, "green"] = stack[1]
        t.loc[row_idx, "blue"] = stack[2]
        t.loc[row_idx, "nir"] = stack[3]
        t.loc[row_idx, "ndvi"] = ndvi[y, x]
        t.loc[row_idx, "reflectance_std"] = np.nanstd(stack, 0)
        t.loc[row_idx, "reflectance_var"] = np.nanvar(stack, 0)
        t.loc[row_idx, "red_blue_ratio"] = normalized_ratio(stack[0], stack[2])
        t.loc[row_idx, "green_blue_ratio"] = normalized_ratio(stack[1], stack[2])
        t.loc[row_idx, "red_normalized"] = stack_normalized[0]
        t.loc[row_idx, "green_normalized"] = stack_normalized[1]
        t.loc[row_idx, "blue_normalized"] = stack_normalized[2]
        t.loc[row_idx, "nir_normalized"] = stack_normalized[3]
        t.loc[row_idx, "red_global_mean"] = means[0]
        t.loc[row_idx, "green_global_mean"] = means[1]
        t.loc[row_idx, "blue_global_mean"] = means[2]
        t.loc[row_idx, "nir_global_mean"] = means[3]
    return t


def add_background(t, reflectances, ratios, means, n_background):
    ndvi = normalized_ratio(reflectances[3], reflectances[0])
    label_int, label = 1, "background"
    not_nan_reflectances = np.int8(~np.isnan(reflectances[0:4]))
    not_nan_ratios = np.int8(~np.isnan(ratios))
    not_nan_y, not_nan_x = np.where((np.min(not_nan_reflectances, 0) * np.min(not_nan_ratios, 0)) == 1)
    random_indices = np.random.randint(0, len(not_nan_y), np.clip(n_background, 0, len(not_nan_y)))
    reflectances_normalized = np.zeros_like(reflectances)
    for band_idx, mean_value in zip(range(reflectances.shape[0]), means):
        reflectances_normalized[band_idx] = reflectances[band_idx].copy() / mean_value
    for random_idx in zip(random_indices):
        y_arr_idx, x_arr_idx = not_nan_y[random_idx], not_nan_x[random_idx]
        stack_normalized = reflectances_normalized[:, y_arr_idx, x_arr_idx]
        stack = reflectances[:, y_arr_idx, x_arr_idx]
        row_idx = len(t)
        t.loc[row_idx, "label_int"] = label_int
        t.loc[row_idx, "label"] = label
        t.loc[row_idx, "red"] = stack[0]
        t.loc[row_idx, "green"] = stack[1]
        t.loc[row_idx, "blue"] = stack[2]
        t.loc[row_idx, "nir"] = stack[3]
        t.loc[row_idx, "ndvi"] = ndvi[y_arr_idx, x_arr_idx]
        t.loc[row_idx, "reflectance_std"] = np.nanstd(stack[0:3], 0)
        t.loc[row_idx, "reflectance_var"] = np.nanvar(stack[0:3], 0)
        t.loc[row_idx, "red_normalized"] = stack_normalized[0]
        t.loc[row_idx, "green_normalized"] = stack_normalized[1]
        t.loc[row_idx, "blue_normalized"] = stack_normalized[2]
        t.loc[row_idx, "nir_normalized"] = stack_normalized[3]
        t.loc[row_idx, "red_blue_ratio"] = normalized_ratio(stack[0], stack[2])
        t.loc[row_idx, "green_blue_ratio"] = normalized_ratio(stack[1], stack[2])
        t.loc[row_idx, "red_global_mean"] = means[0]
        t.loc[row_idx, "green_global_mean"] = means[1]
        t.loc[row_idx, "blue_global_mean"] = means[2]
        t.loc[row_idx, "nir_global_mean"] = means[3]
    return t


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
    return np.float32([diff_red, diff_green, diff_blue])


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
        extract_statistics(img_file, boxes_truth, int(tiles_pd[tiles_pd["training_tiles"] == tile]["n_retain"]),
                           file_path_spectra_ml)
