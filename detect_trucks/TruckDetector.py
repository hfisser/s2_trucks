####################################################
# Author: Henrik Fisser, 2020
####################################################

from array_utils.plot import plot_img

import os, warnings
import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
from shapely.geometry import box
from scipy.stats import linregress, spearmanr
from datetime import datetime

from array_utils.math import normalized_ratio, rescale
from array_utils.geocoding import lat_from_meta, lon_from_meta, metadata_to_bbox_epsg4326
from osm_utils.utils import get_roads, rasterize_osm
from utils.ProgressBar import ProgressBar

warnings.filterwarnings("ignore")

dir_ancillary = os.path.join("F:" + os.sep + "Masterarbeit", "DLR", "project", "1_truck_detection", "truth")
THRESHOLDS = pd.read_csv(os.path.join(dir_ancillary, "thresholds.csv"), index_col=0)
RGB_VECTORS = pd.read_csv(os.path.join(dir_ancillary, "rgb_vector_clusters.csv"), index_col=0)

# assume reflectance rescaled to [0., 1.]
# REFLECTANCE
MIN_RED = THRESHOLDS["red_low"][0]
MAX_RED = THRESHOLDS["red_high"][0]
#MAX_RED_BOX = THRESHOLDS["box_mean_red_high"][0]
MIN_GREEN = THRESHOLDS["green_low"][0]
MAX_GREEN = THRESHOLDS["green_high"][0]
#MAX_GREEN_BOX = THRESHOLDS["box_mean_green_high"][0]
MIN_BLUE = THRESHOLDS["blue_low"][0]
MAX_BLUE = THRESHOLDS["blue_high"][0]
#MAX_BLUE_BOX = THRESHOLDS["box_mean_blue_high"][0]
MIN_RGB_STD = THRESHOLDS["min_std"][0] / 3
# VEGETATION
MAX_NDVI = THRESHOLDS["ndvi_mean"][0] + THRESHOLDS["ndvi_std"][0] * 3
# RATIOS
MIN_BLUE_RED_RATIO = 0
MIN_BLUE_GREEN_RATIO = 0
MIN_GREEN_BLUE_RATIO = 0
MIN_RED_BLUE_RATIO = 0
# SPATIAL
MEAN_MAX_DIST_GREEN = THRESHOLDS["mean_max_dist_green"][0]
MEAN_MAX_DIST_RED = THRESHOLDS["mean_max_dist_red"][0]
MAX_MAX_DIST_GREEN = THRESHOLDS["max_max_dist_green"][0]
MAX_MAX_DIST_RED = THRESHOLDS["max_max_dist_red"][0]
MAX_ANGLE_BR_BG = THRESHOLDS["mean_red_green_spatial_angle"][0] + THRESHOLDS["std_red_green_spatial_angle"][0] * 3
# SPECTRAL ANGLE
#MIN_R_SQUARED = THRESHOLDS["mean_rgb_rsquared"][0] - THRESHOLDS["std_rgb_rsquared"][0] * 3
DEFAULT_MIN_CORRELATION = 0.1
MAX_SLOPE = 10
MIN_SLOPE = 0.05

# Open Street Maps buffer
OSM_BUFFER = 25

# Sensing offset
SECONDS_OFFSET_B02_B04 = 1.01  # seconds
TRUCK_LENGTH = 18.75  # meters

HOME = os.path.dirname(__file__)


class Detector:
    def __init__(self, min_r_squared=None, min_blue_green_ratio=None, min_blue_red_ratio=None):
        """
        Detector class for detecting large moving vehicles on roads using Sentinel-2 data
        :param min_r_squared: float minimum correlation threshold
        :param min_blue_green_ratio: float minimum blue-green ratio for detection
        :param min_blue_red_ratio: float minimum blue-red ratio for detection
        """
        self.min_r_squared = min_r_squared
        self.min_blue_green = min_blue_green_ratio
        self.min_blue_red = min_blue_red_ratio
        self.min_score = None
        self.band_stack_np = None
        self.lat, self.lon = None, None
        self.trucks_np = None
        self.crs = None

    def pre_process(self, band_dict, metadata, subset_box=None):
        """
        rescales data to 0-1 and calculates lat, lon coordinates, masks to OSM roads
        :param band_dict: dict holding 3 arrays with shape (height, width), keys are B02, B03, B04, B08
        :param metadata: dict metadata from rasterio IO
        :param subset_box: dict with int ymin, ymax, xmin, xmax
        """
        self.min_r_squared = DEFAULT_MIN_CORRELATION #if self.min_r_squared is None else self.min_r_squared
        if not isinstance(band_dict, dict):
            raise TypeError("'band_dict' must be a dictionary")
        try:
            test = band_dict["B02"], band_dict["B03"], band_dict["B04"], band_dict["B08"]
        except KeyError:
            raise KeyError("'band_dict' must contain 'B02', 'B03', 'B04', 'B08'")
        if not isinstance(metadata, dict):
            raise TypeError("'metadata' must be a dictionary")
        self.crs = metadata["crs"]
        try:
            self.lat, self.lon = metadata["lat"], metadata["lon"]
        except KeyError:
            try:
                self.lat, self.lon = lat_from_meta(metadata), lon_from_meta(metadata)
            except KeyError as e:
                raise e
        box_utm = [np.min(self.lat), np.max(self.lon), np.max(self.lat), np.min(self.lon)]
        box_epsg4326 = metadata_to_bbox_epsg4326(metadata)
        dir_ancil = os.path.join(HOME, "AUXILIARY")
        if not os.path.exists(dir_ancil):
            os.mkdir(dir_ancil)
        box_epsg4326 = list(np.flip(box_epsg4326))
        osm_mask = self.get_osm_mask(box_epsg4326, metadata["crs"], band_dict["B02"],
                                     {"lat": self.lat, "lon": self.lon},
                                     dir_ancil)
        band_stack_np = np.array([band_dict["B04"], band_dict["B03"], band_dict["B02"], band_dict["B08"]])
        low_rgb_mask = self.calc_low_quantile_mask(band_stack_np[0:3], [0.2])  # mask out lowest 20 % reflectances
        #high_rgb_mask = self.calc_high_quantile_mask(band_stack_np[0:3], [0.98])  # mask out highest 1 % reflectances
#        band_stack_np[:, np.isnan(low_rgb_mask)] = np.nan
        #band_stack_np[:, np.isnan(high_rgb_mask)] = np.nan
        band_stack_np *= osm_mask
        try:
            band_stack_np = band_stack_np[:, subset_box["ymin"]:subset_box["ymax"], subset_box["xmin"]:subset_box["xmax"]]
            self.lat = self.lat[subset_box["ymin"]:subset_box["ymax"] + 1]
            self.lon = self.lon[subset_box["xmin"]:subset_box["xmax"] + 1]
        except TypeError:  # subset_box is allowed to be None
            pass
        band_stack_np_rescaled = band_stack_np.copy()
        band_stack_np = None
        band_stack_np_rescaled[np.isnan(band_stack_np_rescaled)] = 0
        band_stack_np_rescaled = rescale(band_stack_np_rescaled, 0, 1)
  #      band_stack_np_rescaled[:, band_stack_np_rescaled[0] > THRESHOLDS["red_high"][0]] = np.nan
   #     band_stack_np_rescaled[:, band_stack_np_rescaled[1] > THRESHOLDS["green_high"][0]] = np.nan
    #    band_stack_np_rescaled[:, band_stack_np_rescaled[2] > THRESHOLDS["blue_high"][0]] = np.nan
        band_stack_np_rescaled[band_stack_np_rescaled == 0] = np.nan
        return band_stack_np_rescaled

    def detect_trucks(self, band_stack_np):
        """
        Method for detecting large moving vehicles, calls ratio-based detection and object delineation
        :param band_stack_np: numpy ndarray containing the pre-processed Sentinel-2 reflectance bands
        :return: GeoDataframe containing the detected boxes
        """
        t0 = datetime.now()
        if not isinstance(band_stack_np, np.ndarray):
            raise TypeError("'band_stack_np' must be of type numpy.ndarray")
        self.band_stack_np = band_stack_np
        self._detect()
        detections = self._context_zoom()  # zoom into context around potential detection
        print("Duration: %s minutes" % ((datetime.now() - t0).total_seconds() / 60))
        return detections

    def _detect(self):
        """
        Detect pixels of superior blue reflectance based on band ratios
        """
        b02, b03, b04 = self.band_stack_np[2], self.band_stack_np[1], self.band_stack_np[0]
        min_quantile_blue, max_quantile_blue = np.nanquantile(b02, [0.4]), np.nanquantile(b02, [0.999])
        max_quantile_green, max_quantile_red = np.nanquantile(b03, [0.9]), np.nanquantile(b04, [0.9])
        bg_ratio, br_ratio = normalized_ratio(b02, b03), normalized_ratio(b02, b04)
        bg = np.int8(bg_ratio > np.nanmean(b02) - np.nanmean(b03))
        br = np.int8(br_ratio > np.nanmean(b02) - np.nanmean(b04))
        blue_min = np.int8(b02 > min_quantile_blue)  # exclude low 50 % blue
        blue_max = np.int8(b02 < max_quantile_blue)
        green_max = np.int8(b03 < max_quantile_green)
        red_max = np.int8(b04 < max_quantile_red)
        mask = self.expose_anomalous_pixels(self.band_stack_np)
        self.band_stack_np = self.band_stack_np * mask
        self.band_stack_np[self.band_stack_np == 0] = np.nan
        # ratios B02-B03 (blue-green) and B02-B04 (blue-red)
        std_min = np.int8(np.nanstd(self.band_stack_np[0:3], 0) * 10 >= THRESHOLDS["q1_std_at_max_blue"][0])
      #  self.trucks_np = np.int8(bg * br * blue_min * blue_max * std_min * green_max * red_max)
        self.trucks_np = np.int8(bg * br * blue_min * std_min)
        bg_ratio, br_ratio, blue_min, blue_max, green_max, red_max, std_min = None, None, None, None, None, None, None

    def _context_zoom(self):
        """
        Looks at the spatial context each detected pixel and calls method for delineating potential object
        :return: GeoDataframe containing the detected boxes
        """
        valid = np.where(self.trucks_np == 1)  # y, x indices
        boxes = [[], [], [], [], [], [], [], [], [], [], [], []]
        y_max, x_max = self.trucks_np.shape
        print("Context zoom\n%s" % (len(valid[0])))
        pb = ProgressBar(len(valid[0]), 50)
        for y, x, i in zip(valid[0], valid[1], range(len(valid[0]))):
            pb.update(i)
            if self.trucks_np[y, x] != 1:  # may be the case because previously eliminated
                continue
            radius_low = int(MEAN_MAX_DIST_RED) + 2
            radius_up = radius_low + 1
            # subset around potential detection
            y_low, y_up = y - radius_low, y + radius_up
            y_low, y_up = 0 if y_low < 0 else y_low, y_max if y_up > y_max else y_up
            x_low, x_up = x - radius_low, x + radius_up
            x_low, x_up = 0 if x_low < 0 else x_low, x_max if x_up > x_max else x_up
            self.trucks_np = self.eliminate_multi_detections(self.trucks_np, y, x)
            sub_stack = self.band_stack_np[:, y_low:y_up, x_low:x_up].copy()
            if np.count_nonzero(~np.isnan(sub_stack)) == 0:
                continue
            t0 = datetime.now()
            box_test_result = self._box_test(sub_stack)
            t1 = datetime.now()
           # print("Total: %s" % str((t1 - t0).total_seconds()))
            try:
                the_box = box_test_result["box"]
            except KeyError:
                continue
            else:
                box_metrics = box_test_result["box_metrics"]
                bounding_box = [the_box["xmin"], the_box["ymin"], the_box["xmax"], the_box["ymax"]]
                # get box in full array
                box_full_array = [x_low + bounding_box[0], y_low + bounding_box[1],
                                  x_low + bounding_box[2], y_low + bounding_box[3]]
                box_full_array[2] = self.lon.shape[0] - 1 if box_full_array[2] >= self.lon.shape[0] else box_full_array[2]
                box_full_array[3] = self.lat.shape[0] - 1 if box_full_array[3] >= self.lat.shape[0] else box_full_array[3]
                ymax, xmax = box_full_array[3] + 1, box_full_array[2] + 1
                ymax = self.lat.shape[0] - 1 if ymax >= self.lat.shape[0] else ymax  # may happen
                xmax = self.lon.shape[0] - 1 if xmax >= self.lon.shape[0] else xmax
                bounding_box = box(self.lon[box_full_array[0]],
                                   self.lat[box_full_array[1]],
                                   self.lon[xmax],
                                   self.lat[ymax])
                direction_degree = box_metrics["direction"]
                values = [bounding_box, box_metrics["spectral_angle"], box_metrics["slope"],
                          self.direction_degree_to_description(direction_degree), direction_degree,
                          box_test_result["quantile"], box_test_result["speed"], box_metrics["score"],
                          box_metrics["std"], box_metrics["red_mean"], box_metrics["green_mean"], box_metrics["blue_mean"]]
                for idx, value in enumerate(values):
                    boxes[idx].append(value)
        detections = gpd.GeoDataFrame({"rsquared": boxes[1],
                                       "slope": boxes[2],
                                       "direction_description": boxes[3],
                                       "direction_degree": boxes[4],
                                       "localization_quantile": boxes[5],
                                       "speed": boxes[6],
                                       "score": boxes[7],
                                       "std": boxes[8],
                                       "red_ratio": boxes[9],
                                       "green_ratio": boxes[10],
                                       "blue_ratio": boxes[11]},
                                      geometry=boxes[0],
                                      crs=self.crs)
        print("\nNumber of detections: %s" % (len(detections)))
        return detections

    def _box_test(self, subset):
        """
        looks at subset around detection and localizes object as box
        :param subset: numpy ndarray of shape (4, 9, 9) containing the reflectances of subset
        :return: dict with resulting detection box and its metrics
        """
        t0 = datetime.now()
        subset_copy = subset.copy()
        subset[:, normalized_ratio(subset[3], subset[0]) > MAX_NDVI] = np.nan
        detection_y, detection_x = int(subset.shape[1] / 2), int(subset.shape[2] / 2)  # index of detection (center)
        detection_yx = [detection_y, detection_x]
        if np.isnan(subset[0, detection_y, detection_x]):  # NDVI too high. Mask here, saves time
            return {}
        detection_stack = subset[:, detection_y, detection_x].copy()
        subset[:, detection_y, detection_x] = detection_stack.copy()
        if np.count_nonzero(~np.isnan(subset[0])) < 3:
            return {}
        n_bands = subset.shape[0] - 1
        ratios = np.zeros((n_bands * 2 + 2, subset.shape[1], subset.shape[2]))
        ratio_counterparts = [[1, 2], [0, 2], [0, 1]]
        for i in range(n_bands):
            for j, k in enumerate(ratio_counterparts[i]):
                ratios[i + i + j] = normalized_ratio(subset[i], subset[k])
        ratios[6] = np.nanstd(subset[0:3], 0) * 10
        ratios[7] = np.nanstd(ratios, 0) * 10
        ratios[:, np.isnan(ratios[0])] = np.nan
        # localize potential box through high quantile
        q = np.float32([0.99])
     #   print("Section 1 took: %s" % str((datetime.now() - t0).total_seconds()))
        t0 = datetime.now()
        qantiles_dummy = np.float32([1, 1])
        quantiles_sum = qantiles_dummy.copy()
        while np.count_nonzero(quantiles_sum) < 6 and q[0] > 0.5:
            quantiles_sum = self.quantile_filter(ratios, q)
            if quantiles_sum is None:
                quantiles_sum = qantiles_dummy.copy()
            q -= 0.01
        q += 0.01
  #      print("Section 2 took: %s" % str((datetime.now() - t0).total_seconds()))
        t0 = datetime.now()
        try:
            s = all(quantiles_sum == qantiles_dummy)
        except TypeError:  # then it's alright
            pass
        else:
            return {}
        try:
            quantiles_sum[quantiles_sum > 0] = 1
        except TypeError:
            return {}
   #     quantiles_sum = self.eliminate_single_nonzeros(quantiles_sum)
        if np.count_nonzero(quantiles_sum > 0) < 3:
            return {}
        for j, k, t in zip([0, 2, 4], [1, 3, 5], [MAX_MAX_DIST_RED + 1, MAX_MAX_DIST_GREEN + 1, 2]):
            subset, ratios, quantiles_sum = self._eliminate_distant_pixels(subset, ratios, ratios[j] + ratios[k],
                                                                           quantiles_sum, detection_yx, t, q)
        # apply cluster exposing method twice in order to account for changes introduced by filter
        y_low, x_low, y_up, x_up = detection_y - 1, detection_x - 1, detection_y + 2, detection_x + 2
        quantiles_sum[y_low:y_up, x_low:x_up] = np.zeros((3, 3))  # temporary
        spatial_cluster = self._expose_cluster(quantiles_sum, subset[0:3], False)
        # if a cluster has high amount of values exclude corners, potentially divide large cluster
        boxes, boxes_metrics, scores, clusters = [], [], [], []
     #   print("Section 3 took: %s" % str((datetime.now() - t0).total_seconds()))
        t0 = datetime.now()
        for cluster in np.unique(spatial_cluster[spatial_cluster != 0]):
            spatial_cluster[detection_y, detection_x] = cluster  # assign value of cluster to detection pixel
            ys, xs = np.where(spatial_cluster == cluster)
            try:
                a_box = [np.min(ys), np.min(xs), np.max(ys), np.max(xs)]
            except ValueError:
                continue
            box_arr = subset[0:3, a_box[0]:a_box[2]+1, a_box[1]:a_box[3]+1].copy()
       #     if (np.nanmean(np.nanstd(box_arr, 0)) * 10) < MIN_RGB_STD * 0.5:  # be tolerant here
        #        continue
            cluster_sub = spatial_cluster[a_box[0]:a_box[2]+1, a_box[1]:a_box[3]+1].copy()
            cluster_sub[np.isnan(box_arr[0])] = 0
            ys, xs = np.where(spatial_cluster == cluster)
            if len(ys) < 2:
                continue
            ys, xs = self.eliminate_outlier_indices(ys, xs)
            a_box = [np.min(ys), np.min(xs), np.max(ys), np.max(xs)]
            box_arr = subset[0:3, a_box[0]:a_box[2]+1, a_box[1]:a_box[3]+1].copy()
            if np.count_nonzero(~np.isnan(box_arr)) / 3 / (box_arr.shape[1] * box_arr.shape[2]) < 0.3:  # too few pixels
                continue
            box_ratios = ratios[:, a_box[0]:a_box[2]+1, a_box[1]:a_box[3]+1].copy()
            t0b = datetime.now()
            box_metrics = self._characterize_spatial_spectral(box_arr, box_ratios)
          #  a_box = self._crop_box(a_box, ratios, box_metrics["direction"], detection_yx)
            #print("Section 4b took: %s" % str((datetime.now() - t0b).total_seconds()))
           # box_arr = subset[0:3, a_box[0]:a_box[2] + 1, a_box[1]:a_box[3] + 1].copy()
            if all([box_arr.shape[1] <= 2, box_arr.shape[2] <= 2]):
                continue
            box_metrics = self.calc_score(box_metrics, box_arr)
            if self._spatial_spectral_match(box_metrics):
                clusters.append(cluster)
                boxes.append(a_box)
                boxes_metrics.append(box_metrics)
                scores.append(box_metrics["score"])
   #     print("Section 4 took: %s" % str((datetime.now() - t0).total_seconds()))
        t0 = datetime.now()
        scores = np.array(scores)
        try:
            max_score = np.max(scores)
            match = np.where(scores == max_score)[0][0]
        except ValueError:
            return {}
        box_metrics, selected_box = boxes_metrics[match], boxes[match]
        if np.std(selected_box) == 0:
            return {}
        if any(self.box_too_large(selected_box, MAX_MAX_DIST_RED)):
            selected_box = self._subset_by_ratios(ratios, selected_box)  # subset box to high quantile ratios
            if any(self.box_too_large(selected_box, MAX_MAX_DIST_RED)):
                subset_dict = self._subset_by_boxes(subset, ratios, selected_box, [3, 4])  # try default sub boxes
                try:
                    box_metrics = subset_dict["box_metrics"]
                except KeyError:
                    pass
                else:
                    a_box = subset_dict["selected_box"]
                    box_arr = subset[0:3, a_box[0]:a_box[2] + 1, a_box[1]:a_box[3] + 1]
                    box_metrics = self.calc_score(box_metrics, box_arr)
                    if not self._spatial_spectral_match(box_metrics):
                        return {}
        box_too_small = all([(selected_box[2] - selected_box[0] + 1) <= 2, (selected_box[3] - selected_box[1] + 1) <= 2])
        if box_too_small or box_metrics["score"] < self.min_score:
            return {}
        the_box = {"ymin": selected_box[0], "xmin": selected_box[1],
                   "ymax": selected_box[2], "xmax": selected_box[3]}
  #      print("Section 5 took: %s" % str((datetime.now() - t0).total_seconds()))
        return {"box": the_box,
                "box_metrics": box_metrics,
                "quantile": q[0],
                "speed": self.calc_speed(ratios[:, the_box["ymin"]:the_box["ymax"]+1, the_box["xmin"]:the_box["xmax"]+1])}

    def _characterize_spatial_spectral(self, sub_arr, sub_variables):
        """
        takes a subset of reflectance stack and corresponding variables (ratios, std) and returns metrics of correlation
        and spatial relationships
        :param sub_arr: numpy ndarray of shape (3, y, x) containing the reflectance bands
        :param sub_variables: numpy ndarray of shape (7, y, x) containing ratios of reflectance bands, RGB std
        and ratios std
        :return: dict containing the box metrics
        """
        return_dict = {}
        keys = ["spectral_angle", "spatial_angle", "slope", "red_length", "green_length", "direction",
                "blue_mean", "green_mean", "red_mean", "red_ratio_max", "green_ratio_max", "blue_ratio_max"]
        for key in keys:
            return_dict[key] = np.nan
        return_dict_copy = return_dict.copy()
        blue_ratios = np.nansum(sub_variables[4:6], 0) + sub_arr[2] * 10  # sum of blue ratios
        green_ratios = np.nansum(sub_variables[2:4], 0) + sub_arr[1] * 10  # sum of green ratios
        red_ratios = np.nansum(sub_variables[0:2], 0) + sub_arr[0] * 10  # sum of red ratios
        try:
            try:
                blue_y, blue_x = self.crop_2d_indices(np.where(blue_ratios == np.nanmax(blue_ratios)))
            except ValueError:
                return return_dict
            else:
                green_ratios[blue_y, blue_x] = np.nan  # set to nan in order to avoid double target
                green_y, green_x = self.crop_2d_indices(np.where(green_ratios == np.nanmax(green_ratios)))
                red_ratios[blue_y, blue_x] = np.nan  # avoid double target
                red_ratios[green_y, green_x] = np.nan  # ""
                red_y, red_x = self.crop_2d_indices(np.where(red_ratios == np.nanmax(red_ratios)))
        except IndexError:
            return return_dict
        # calculate spatial distances from blue and eliminate blue if not fitting, try again
#        green_length, red_length = 1, 0
 #       for y, x in zip([red_y, green_y], [red_x, green_x]):
  #          blue_ratios[y, x] = np.nan
        # if index of max green ratios is further away from blue than red try to find a better suitable combination
     #   while green_further_away and np.count_nonzero(~np.isnan(blue_ratios)) > 0:
      #      red_length = self.calc_vector_length(self.calc_vector([blue_y, blue_x], [red_y, red_x]))
       #     green_length = self.calc_vector_length(self.calc_vector([blue_y, blue_x], [green_y, green_x]))
        #    green_further_away = green_length > red_length
         #   if green_further_away:
          #      blue_ratios[blue_y, blue_x] = np.nan
           #     try:
            #        blue_y, blue_x = self.crop_2d_indices(np.where(blue_ratios == np.nanmax(blue_ratios)))
             #   except IndexError:
               #     return return_dict
        blue_indices = [blue_y, blue_x]
        blue_red_spatial_vector = self.calc_vector([red_y, red_x], blue_indices)  # spatial vector blue to red
        blue_green_spatial_vector = self.calc_vector([green_y, green_x], blue_indices)  # spatial vector blue to green
        return_dict = {"red_length": self.calc_vector_length(blue_red_spatial_vector),
                       "green_length": self.calc_vector_length(blue_green_spatial_vector),
                       "spatial_angle": self.calc_vector_angle_in_degrees(blue_red_spatial_vector,
                                                                          blue_green_spatial_vector)}
        if not self._spatial_spectral_match(return_dict):  # check that in order to reduce run time
            return return_dict_copy  # if spatial metrics do not satisfy thresholds return here alread
        given_vector = np.hstack([sub_variables[4:6, blue_y, blue_x],  # stack of variables and target pixels
                                  sub_variables[2:4, green_y, green_x],
                                  sub_variables[0:2, red_y, red_x],
                                  sub_variables[6, blue_y, blue_x],
                                  sub_variables[6, green_y, green_x],
                                  sub_variables[6, red_y, red_x],
                                  sub_variables[7, blue_y, blue_x],
                                  sub_variables[7, green_y, green_x],
                                  sub_variables[7, red_y, red_x],
                                  sub_arr[2, blue_y, blue_x],
                                  sub_arr[2, green_y, green_x],
                                  sub_arr[2, red_y, red_x],
                                  sub_arr[1, green_y, green_x],
                                  sub_arr[1, blue_y, blue_x],
                                  sub_arr[1, red_y, red_x],
                                  sub_arr[0, red_y, red_x],
                                  sub_arr[0, blue_y, blue_x],
                                  sub_arr[0, green_y, green_x]])
        col_names, spectral_angles, slopes, spearman = [], [], [], []
        for i in range(7):
            col_names = col_names + ["rgb_vector" + str(i) + str(j) for j in [0, 1, 2]]
        # calculate spearmanr correlations between given variables and all reference variables
        for row in RGB_VECTORS.iterrows():
            r = row[1]
            ref_vector = np.array([r[col_name] for col_name in col_names])
            regression = linregress(given_vector, ref_vector)
            spearman.append(spearmanr(given_vector, ref_vector)[0])
            #spectral_angles.append(regression.rvalue)
            slopes.append(regression.slope)
        # use mean of all spearmanr correlation coefficients as indicator for agreement with reference dataset
        return_dict["spectral_angle"] = np.nanmean(spearman)  #np.nanquantile(spectral_angles, [0.75])[0] - np.nanstd(spectral_angles)
        return_dict["slope"] = np.nanmean(slopes)
        return_dict["direction"] = self.calc_vector_direction_in_degree(np.mean(np.vstack([blue_red_spatial_vector,
                                                                                           blue_green_spatial_vector]),
                                                                                axis=0))
        return_dict["red_mean"] = np.nanmean(sub_arr[0])
        return_dict["green_mean"] = np.nanmean(sub_arr[1])
        return_dict["blue_mean"] = np.nanmean(sub_arr[2])
        return_dict["red_ratio_max"] = np.nanmax(np.nanmax(sub_variables[0:2]))
        return_dict["green_ratio_max"] = np.nanmax(np.nanmax(sub_variables[2:4]))
        return_dict["blue_ratio_max"] = np.nanmax(np.nanmax(sub_variables[4:6]))
        return return_dict

    def _subset_by_ratios(self, ratios, selected_box):
        original_box = selected_box.copy()
        box_ratios = ratios[:, selected_box[0]:selected_box[2]+1, selected_box[1]:selected_box[3]+1]
        q = np.float32([0.2])
        too_large_y, too_large_x = True, True
        while any([too_large_y, too_large_x]) and q[0] < 1:
            too_large_y, too_large_x = self.box_too_large(selected_box, MAX_MAX_DIST_RED)
            if any([too_large_y, too_large_x]):
                quantiles_sum = self.quantile_filter(box_ratios, q)
                if quantiles_sum is not None:
                    ys, xs = np.where(quantiles_sum != 0)
                    try:
                        selected_box = [min(ys), min(xs), max(ys), max(xs)]
                    except ValueError:
                        q += 0.01
                        continue
            q += 0.01
        if selected_box != original_box:
            selected_box[2] = original_box[0] + selected_box[2]
            selected_box[3] = original_box[1] + selected_box[3]
            selected_box[0] += original_box[0]
            selected_box[1] += original_box[1]
        return selected_box

    def _subset_by_boxes(self, subset, ratios, selected_box, window_sizes):
        box_arr = subset[0:3, selected_box[0]:selected_box[2] + 1, selected_box[1]:selected_box[3] + 1]
        box_ratios = ratios[:, selected_box[0]:selected_box[2] + 1, selected_box[1]:selected_box[3] + 1]
        boxes, boxes_metrics, boxes_rsquared, boxes_rgb_sums, boxes_spatial_angle = [], [], [], [], []
        for w in window_sizes:
            y_indices_low = np.arange(0, box_arr.shape[1] - w + 1, 1)
            x_indices_low = np.arange(0, box_arr.shape[2] - w + 1, 1)
            y_indices_up = [y + w for y in y_indices_low]
            x_indices_up = [x + w for x in x_indices_low]
            for y_low, y_up in zip(y_indices_low, y_indices_up):
                for x_low, x_up in zip(x_indices_low, x_indices_up):
                    sub_box_arr = box_arr[:, y_low:y_up, x_low:x_up]
                    sub_box_ratios = box_ratios[:, y_low:y_up, x_low:x_up]
                    box_metrics = self._characterize_spatial_spectral(sub_box_arr, sub_box_ratios)
                    if self._spatial_spectral_match(box_metrics):
                        max_values = [np.nanmax(sub_box_arr[i]) for i in range(sub_box_arr.shape[0])]
                        boxes.append([y_low, x_low, y_up - 1, x_up - 1])  # -1 due to indexing
                        boxes_metrics.append(box_metrics)
                        boxes_rsquared.append(box_metrics["spectral_angle"])
                        boxes_rgb_sums.append(np.sum(max_values))
                        boxes_spatial_angle.append(box_metrics["spatial_angle"])
        combined = np.array(boxes_rsquared) + np.array(boxes_rgb_sums) - np.array(boxes_spatial_angle)
        try:
            max_combined = np.max(combined)
        except ValueError:
            return {}
        try:
            match = np.where(combined == max_combined)[0][0]
        except IndexError:
            return {}
        new_box = boxes[match]
        selected_box[2] = selected_box[0] + new_box[2]
        selected_box[3] = selected_box[1] + new_box[3]
        selected_box[0] += new_box[0]
        selected_box[1] += new_box[1]
        return {"box_metrics": boxes_metrics[match], "selected_box": selected_box}

    def _eliminate_distant_pixels(self, sub_arr, ratios, band_ratios, quantiles_sum, center, threshold, quantile):
        try:
            ys, xs = np.where(band_ratios > np.nanquantile(band_ratios, quantile))
        except ValueError:
            return sub_arr
        else:
            for y, x in zip(ys, xs):
                if self.calc_vector_length(self.calc_vector(center, [y, x])) > threshold:
                    sub_arr[:, y, x] = np.nan
                    ratios[:, y, x] = np.nan
                    quantiles_sum[y, x] = 0
        return sub_arr, ratios, quantiles_sum

    def _expose_cluster(self, target_arr, band_stack, exclude_corners=True):
        target_arr[np.isnan(target_arr)] = 0
        if np.count_nonzero(target_arr) == 0:
            return target_arr
        try:
            center = [int(target_arr.shape[0] / 2), int(target_arr.shape[1] / 2)]
        except IndexError:
            return target_arr
        ys, xs = np.where(target_arr != 0)
        yet_seen, cluster_value, clusters = [], 0, target_arr.copy()
        for y, x in zip(ys, xs):
            distance_center = self.calc_vector_length(self.calc_vector([y, x], center)) - 1
            rgb_slice = band_stack[0:3, y, x]
            max_idx = np.where(rgb_slice == np.nanmax(rgb_slice))[0][0]
            distance_wrong = [distance_center > t for t in [MAX_MAX_DIST_RED, MEAN_MAX_DIST_RED, MEAN_MAX_DIST_GREEN]]
            max_idx_wrong = [True, max_idx not in [0, 1], max_idx not in [0, 1, 2]]
            should_continue = False
            for condition_a, condition_b in zip(distance_wrong, max_idx_wrong):
                if condition_a and condition_b:
                    clusters[y, x], should_continue = 0, True
                    break
            if should_continue:
                continue
            if not [y, x] in yet_seen:
                cluster_value += 1
            clusters, yet_seen = self._search_adjacent_non_zero(clusters, [y, x], cluster_value, yet_seen,
                                                                exclude_corners)
        return clusters

    def _crop_box(self, given_box, ratios, direction, detection_yx):
        max_size = MAX_MAX_DIST_RED * 2
        box_size = (given_box[2] - given_box[0] + 1) * (given_box[3] - given_box[1] + 1)
        direction_match = any(np.abs([x - direction for x in [0, 90, 180, 270]]) < 45)
        q = [0.5]
        while direction_match and box_size >= max_size and q[0] < 1:
            box_ratios = ratios[:, given_box[0]:given_box[2] + 1, given_box[1]:given_box[3] + 1]
            quantiles = self.quantile_filter(box_ratios, q)
            if quantiles is not None:
                try:
                    # always retain value 1 at detection
                    quantiles[np.abs(detection_yx[0] - given_box[0]), np.abs(detection_yx[1] - given_box[1])] = 1
                except IndexError:
                    pass
                ys, xs = np.where(quantiles != 0)
                try:
                    given_box[2] = int(given_box[0] + max(ys))
                    given_box[3] = int(given_box[1] + max(xs))
                    given_box[0] += min(ys)
                    given_box[1] += min(xs)
                except ValueError:
                    q[0] += 0.1
                    continue
                else:
                    box_size = (given_box[2] - given_box[0] + 1) * (given_box[3] - given_box[1] + 1)
            if box_size >= max_size:
                q[0] += 0.1
        return given_box

    def _search_adjacent_non_zero(self, arr, point, new_value, yet_seen, exclude_corners):
        """
        looks for non zeros in 3x3 window around point in array and assigns a new value to these non-zeros
        :param arr: np array
        :param point: list of int y, x indices
        :param new_value: int value to assign
        :param yet_seen: list of lists, each list is a point with int y, x indices that has been seen before
        :param exclude_corners: bool, if True the corners of 3x3 window are excluded
        :return: tuple of np array and list
        """
        arr_modified = arr.copy()
        original_value = arr_modified[point[0], point[1]].copy()
        arr_modified[point[0], point[1]] = 0
        ymin, ymax = point[0]-1, point[0]+2
        xmin, xmax = point[1]-1, point[1]+2
        ymin, xmin = 0 if ymin < 0 else ymin, 0 if xmin < 0 else xmin
        window_3x3 = arr_modified[ymin:ymax, xmin:xmax].copy()
        if exclude_corners:
            for corner_y, corner_x in zip([0, 0, 2, 2], [0, 2, 0, 2]):
                try:
                    window_3x3[corner_y, corner_x] = 0
                except IndexError:
                    continue
        ys, xs = np.where(window_3x3 != 0)
        for y_local, x_local in zip(ys, xs):
            y, x = ymin + y_local, xmin + x_local
            if [y, x] not in yet_seen:
                arr_modified[y, x] = new_value
                arr_modified, yet_seen = self._search_adjacent_non_zero(arr_modified, [y, x], new_value, yet_seen,
                                                                        exclude_corners)
                yet_seen.append([y, x])
        value = original_value if point in yet_seen else new_value
        if point not in yet_seen:
            yet_seen.append(point)
        arr_modified[point[0], point[1]] = value
        return arr_modified, yet_seen

    def calc_speed(self, ratios):
        resolution = 10  # meters
        blue_ratios = np.nansum(ratios[4:6], 0)
        red_ratios = np.nansum(ratios[0:2], 0)
        green_ratios = np.nansum(ratios[2:4], 0)
        try:
            max_blue, max_red, max_green = np.nanmax(blue_ratios), np.nanmax(red_ratios), np.nanmax(green_ratios)
        except IndexError:
            return 0
        diameter = (np.max(ratios.shape[1:3]) - (1.5 - max_blue)) * resolution
        kilometers_hour = (diameter * (3600 / SECONDS_OFFSET_B02_B04)) / 1000
        return kilometers_hour

    def _spatial_spectral_match(self, metrics_dict):
        is_match = True
        has_values = 3
#        try:
 #           ratios_means = [metrics_dict["red_ratio_max"], metrics_dict["green_ratio_max"], metrics_dict["blue_ratio_max"]]
  #      except KeyError:
   #         has_values -= 1
    #    else:
     #       ratios_high = np.max(ratios_means) > 0.2
      #      ratios_high_all = all([mean_value > 0.05 for mean_value in ratios_means])
       #     ratios_high_all = ratios_high_all or sum([mean_value > 0.25 for mean_value in ratios_means]) >= 2
        #    ratios_high_two = sum([mean_value > 0.15 for mean_value in ratios_means]) > 1
         #   is_match *= ratios_high * ratios_high_all * ratios_high_two
  #      try:
   #         is_match *= metrics_dict["std"] >= MIN_RGB_STD
    #    except KeyError:
     #       has_values -= 1
        try:
            is_match *= metrics_dict["spectral_angle"] >= self.min_r_squared
        except KeyError:
            has_values -= 1
        try:
            is_match *= metrics_dict["score"] >= self.min_score
        except KeyError:
            has_values -= 1
        try:
            green_length = metrics_dict["green_length"]
            red_length = metrics_dict["red_length"]
            is_match *= green_length < red_length
            is_match *= red_length < (MAX_MAX_DIST_RED + 0.5)
            is_match *= green_length < (MAX_MAX_DIST_GREEN + 0.5)
        except KeyError:
            has_values -= 1
  #      try:
   #         is_match *= metrics_dict["slope"] < MAX_SLOPE
    #        is_match *= metrics_dict["slope"] > MIN_SLOPE
     #   except KeyError:
      #      has_values -= 1
     #   try:
      #      is_match *= metrics_dict["spatial_angle"] < MAX_ANGLE_BR_BG
       # except KeyError:
        #    has_values -= 1
        if has_values == 0:
            return False
        else:
            return is_match

    @staticmethod
    def calc_score(metrics_dict, sub_arr):
        metrics_dict["std"] = np.nanmean(np.nanstd(sub_arr, 0)) * 10
        reflectance_means_sum = (metrics_dict["red_mean"] + metrics_dict["blue_mean"] + metrics_dict[
            "green_mean"]) * 10
        ratio_means_sum = metrics_dict["red_ratio_max"] + metrics_dict["green_ratio_max"] \
                          + metrics_dict["blue_ratio_max"]
        metrics_dict["score"] = metrics_dict["spectral_angle"] + metrics_dict["std"] - np.abs(
            1 - metrics_dict["slope"]) \
                               + reflectance_means_sum + ratio_means_sum - metrics_dict["spatial_angle"] / 100
        return metrics_dict

    @staticmethod
    def calc_primary_accuracy(detected_boxes, validation_boxes):
        out_keys = ["validation_percentage", "detection_percentage", "validation_intersection_percentage",
                    "detection_intersection_percentage"]
        out_dict = {}
        lens = [len(detected_boxes) == 0, len(validation_boxes) == 0]
        if lens[0]:
            print("No entries in 'detected_boxes'")
        if lens[1]:
            print("No entries in 'validation_boxes'")
        if any(lens):
            for key in out_keys:
                out_dict[key] = np.nan
            return out_dict
        intersections = {"validation": [], "detection": []}
        intersection_areas = {"validation": [], "detection": []}
        keys = ["validation", "detection"]
        for boxes_a, boxes_b, key in zip([validation_boxes, detected_boxes], [detected_boxes, validation_boxes], keys):
            for detected_box in boxes_a.geometry:
                for i, validation_box in enumerate(boxes_b.geometry):
                    if detected_box.intersects(validation_box):
                        intersections[key].append(i)
                        detected_gpd = gpd.GeoDataFrame({"geometry": [detected_box]}).set_geometry("geometry")
                        validation_gpd = gpd.GeoDataFrame({"geometry": [validation_box]}).set_geometry("geometry")
                        detected_gpd.crs = detected_boxes.crs
                        validation_gpd.crs = detected_gpd.crs
                        intersected = gpd.overlay(detected_gpd, validation_gpd, how="intersection")
                        intersection_areas[key].append(intersected.area[0] / detected_gpd.area[0] * 100)
        out_values = [(len(intersections["validation"]) / len(validation_boxes)) * 100,
                      (len(intersections["detection"]) / len(detected_boxes)) * 100,
                      np.nanmean(np.array(intersection_areas["validation"])),
                      np.nanmean(np.array(intersection_areas["detection"]))]
        for key, value in zip(out_keys, out_values):
            out_dict[key] = value
        return out_dict

    @staticmethod
    def eliminate_single_nonzeros(arr):
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                window_3x3 = arr[y-1:y+2, x-1:x+2]
                if np.count_nonzero(window_3x3[~np.isnan(window_3x3)]) < 2:
                    arr[y, x] = 0
        return arr

    @staticmethod
    def eliminate_outlier_indices(ys, xs):
        dtype_ys, dtype_xs = ys.dtype, xs.dtype
        ys, xs = ys.astype(np.float32), xs.astype(np.float32)
        unique_ys, unique_xs = np.unique(ys), np.unique(xs)
        n = len(ys)
        n_unique_ys, n_unique_xs = len(unique_ys), len(unique_xs)
        amount_unique_ys, amount_unique_xs = np.zeros(n_unique_ys), np.zeros(n_unique_xs)
        for unique_idx, amount_unique, indices in zip([unique_ys, unique_xs],
                                                      [amount_unique_ys, amount_unique_xs],
                                                      [ys, xs]):
            for i, idx in enumerate(unique_idx):
                amount_unique[i] = len(np.where(indices == idx)[0]) / n * 100
        for amounts, uniques, indices in zip([amount_unique_ys, amount_unique_xs], [unique_ys, unique_xs], [ys, xs]):
            if (amounts > 50).any():  # there is a major y
                outlier_idxs = np.where(amounts < 15)
                if len(outlier_idxs[0]) > 0:
                    for outlier_idx in outlier_idxs:
                        real_idx = uniques[outlier_idx]
                        to_nan = indices == real_idx
                        ys[to_nan] = np.nan  # eliminate y and x index
                        xs[to_nan] = np.nan
        ys, xs = ys[~np.isnan(ys)], xs[~np.isnan(xs)]
        return ys.astype(dtype_ys), xs.astype(dtype_xs)

    @staticmethod
    def quantile_filter(arr, quantile_value):
        """
        Targets values of specified quantile and eliminates isolated values
        :param arr: numpy ndarray of shape (3, height, width) -> RGB
        :param quantile_value: list with float quantile in range of 0 and 1
        :return: numpy 2d array of shape (height, width)
        """
        quantiles = np.array([arr[i] >= np.nanquantile(arr[i], quantile_value) for i in range(arr.shape[0])],
                             dtype=np.int8)
 #       quantiles_initial_sum = quantiles.sum(0)
      #  if np.count_nonzero(np.int8(quantiles_initial_sum > 0) * np.int8(quantiles_initial_sum < 3)) == 0:
       #     return None
        shape = quantiles.shape
        s = shape[1]
        buffers = [2, 2, 1, 1, 2, 2, s, s, s, s]
        for i in range(quantiles.shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    buffer = buffers[i]
                    y_low, y_up = y - buffer, y + buffer + 1
                    x_low, x_up = x - buffer, x + buffer + 1
                    y_low = 0 if y_low < 0 else y_low
                    x_low = 0 if x_low < 0 else x_low
                    y_up, x_up = shape[1] if y_up > shape[1] else y_up, shape[2] if x_up > shape[2] else x_up
                    y_low = y_low - 1 if y_up == (shape[1] + 1) else y_low
                    x_low = x_low - 1 if x_up == (shape[2] + 1) else x_low
                    y_up, x_up = y_up + 1 if y_low == 0 else y_up, x_up + 1 if x_low == 0 else x_up
                    original_value = quantiles[i, y, x]
                    if original_value == 0:
                        continue
                    quantiles_sub = quantiles[:, y_low:y_up, x_low:x_up].copy()
                    quantiles_sub[i] = np.zeros_like(quantiles_sub[i])  # look only for matches in other bands
                    sums = [np.nansum(quantiles_sub[j]) for j in range(quantiles_sub.shape[0])]
                    quantiles[i, y, x] = 0 if np.count_nonzero(sums) < 2 else original_value
        return quantiles.sum(0)

    @staticmethod
    def box_too_large(the_box, max_size):
        size_y, size_x = (the_box[2] - the_box[0] + 1), (the_box[3] - the_box[1] + 1)
        too_large_y = size_y > max_size
        too_large_x = size_x > max_size
        return too_large_y, too_large_x

    # not really needed
    @staticmethod
    def calc_low_ratios_mask(ratios, min_values_ratios):
        ratio_mask = np.zeros_like(ratios[0:3], dtype=np.int8)
        only_false = np.zeros(3, dtype=np.bool)
        # reflectance and ratio filter
        for i in range(ratio_mask.shape[0]):
            idx = 2 * i
            ratio_mask[i] = np.int8((ratios[idx] + ratios[idx + 1]) > min_values_ratios[i])
            only_false[i] = np.count_nonzero(ratio_mask) == 0
        ratio_mask = ratio_mask.sum(0)
        ratio_mask[ratio_mask > 2] = 0
        ratio_mask[(2 >= ratio_mask) * (ratio_mask > 0)] = 1
        return ratio_mask, only_false

    @staticmethod
    def calc_low_quantile_mask(reflectances, q):
        low_quantile_red = np.int8(reflectances[0] > np.nanquantile(reflectances[0], q))
        low_quantile_green = np.int8(reflectances[1] > np.nanquantile(reflectances[1], q))
        low_quantile_blue = np.int8(reflectances[2] > np.nanquantile(reflectances[2], q))
        low_quantile_mask = np.float32(low_quantile_red + low_quantile_green + low_quantile_blue)
        low_quantile_mask[low_quantile_mask == 0] = np.nan
        low_quantile_mask[low_quantile_mask > 0] = 1
        return low_quantile_mask

    @staticmethod
    def calc_high_quantile_mask(reflectances, q):
        high_quantile_red = np.int8(reflectances[0] < np.nanquantile(reflectances[0], q))
        high_quantile_green = np.int8(reflectances[1] < np.nanquantile(reflectances[1], q))
        high_quantile_blue = np.int8(reflectances[2] < np.nanquantile(reflectances[2], q))
        high_quantile_mask = np.float32(high_quantile_red + high_quantile_green + high_quantile_blue)
        high_quantile_mask[high_quantile_mask == 0] = np.nan
        high_quantile_mask[high_quantile_mask > 0] = 1
        return high_quantile_mask

    @staticmethod
    def expose_anomalous_pixels(band_stack_np):
        w = 100
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
        #max_diff = np.nanmax(band_stack_np[0:3] - np.nanmin(roads, 0), 0)
        #mask = np.int8(max_diff > np.nanquantile(max_diff, [0.6]))
        diff_red = band_stack_np[0] - (roads[0] / 2)
        diff_green = band_stack_np[1] - (roads[1] / 2)
        diff_blue = band_stack_np[2] - (roads[2] / 2)
        diff_stack = np.array([diff_red, diff_green, diff_blue])
        mask = np.zeros_like(diff_stack[0])
        for i in range(diff_stack.shape[0]):
            mask += np.int8(diff_stack[i] > np.nanquantile(diff_stack[i], [0.5]))
        #mask[mask != 0] = 1

        mask = np.int8(diff_stack)
 #       import rasterio as rio
#
  #      with rio.open("F:\\Masterarbeit\\DLR\\project\\1_truck_detection\\validation\\test.tiff", "r") as src:
 #           meta = src.meta
#
     #   meta["count"] = 3
    #    with rio.open("F:\\Masterarbeit\\DLR\\project\\1_truck_detection\\validation\\mask14.tiff", "w",
   #                   **meta) as src:
  #          for i in range(diff_stack.shape[0]):
 #               src.write((diff_stack[i]**4).astype(np.float64), i+1)
#
    #    mask[mask == 3] = 0
   #     meta["count"] = 1
  #      with rio.open("F:\\Masterarbeit\\DLR\\project\\1_truck_detection\\validation\\mask4.tiff", "w",
 #                     **meta) as src:
#            src.write(mask.astype(np.float64), 1)


        return mask

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

    @staticmethod
    def crop_2d_indices(indices):
        """
        :param indices: tuple of np int64 indices as returned by np.where
        :return: np int32 indices. Cropped if longer than 1
        """
        return np.array([index_arr[0] for index_arr in indices]).astype(np.int32)

    @staticmethod
    def calc_vector_direction_in_degree(vector):
        # [1,1] -> 45°
        # [-1,1] -> 135°
        # [-1,-1] -> 225°
        # [1,-1] -> 315°
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
    def calc_vector_angle_in_degrees(a, b):
        cos = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
        if np.abs(cos) >= 1:
            return 0
        else:
            return np.degrees(np.arccos(cos))

    @staticmethod
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

    @staticmethod
    def calc_vector_length(vector):
        """
        :param vector: np array vector
        :return: np float32
        """
        squared = np.float32([element ** 2 for element in vector])
        return np.sqrt(squared.sum()).astype(np.float32)

    @staticmethod
    def get_smallest_deviation(arr, value):
        dev = np.abs(arr - value)
        return int(np.where(dev == dev.min())[0][0])

    @staticmethod
    def eliminate_multi_detections(arr, y, x):
        y0 = y - 2 if (y - 2) >= 0 else y
        x0 = x - 2 if (x - 2) >= 0 else x
        y1 = y + 3 if (y + 3) <= arr.shape[0] else arr.shape[0]
        x1 = x + 3 if (x + 3) <= arr.shape[1] else arr.shape[1]
        arr[y0:y1, x0:x1] = np.zeros((y1 - y0, x1 - x0))
        arr[y, x] = 1  # detection of interest remains
        return arr

    @staticmethod
    def calc_context():
        resolution = 10  # meters
        speed = 85 * 1000  # meters per hour
        offset_green = 0.5  # from parallax angle
        offset_red = 1.  # from parallax angle
        dist_red = (speed / (3600 * (1 / offset_red))) / resolution  # m
        dist_green = (speed / (3600 * (1 / offset_green))) / resolution  # m
        return {"pixels_green": dist_green, "pixels_red": dist_red}
