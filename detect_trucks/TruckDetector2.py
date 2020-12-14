####################################################
# Author: Henrik Fisser, 2020
####################################################

from array_utils.plot import plot_img

import os, warnings
import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point
from datetime import datetime
from array_utils.math import normalized_ratio, rescale
from array_utils.geocoding import lat_from_meta, lon_from_meta, metadata_to_bbox_epsg4326
from osm_utils.utils import get_roads, rasterize_osm

warnings.filterwarnings("ignore")

# Open Street Maps buffer
OSM_BUFFER = 25

# for getting station locations
BAST_URL = "https://www.bast.de/BASt_2017/DE/Verkehrstechnik/Fachthemen/v2-verkehrszaehlung/Aktuell/" \
           "zaehl_aktuell_node.html;jsessionid=63610843F87B77C24C4320BC4EAD6647.live21304"

HOME = os.path.dirname(__file__)


class Detector2:
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
        self.dir_ancil = os.path.join(HOME, "AUXILIARY", "osm")
        self.min_score = None
        self.band_stack_np = None
        self.lat, self.lon = None, None
        self.box_epsg4326 = None
        self.osm_mask = None
        self.truck_measure = None
        self.crs = None
        self.metadata = None

    def pre_process(self, band_dict, metadata, subset_box=None):
        """
        rescales data to 0-1 and calculates lat, lon coordinates, masks to OSM roads
        :param band_dict: dict holding 3 arrays with shape (height, width), keys are B02, B03, B04, B08
        :param metadata: dict metadata from rasterio IO
        :param subset_box: dict with int ymin, ymax, xmin, xmax
        """
        self.metadata = metadata
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
        box_epsg4326 = metadata_to_bbox_epsg4326(metadata)
        if not os.path.exists(self.dir_ancil):
            os.mkdir(self.dir_ancil)
        self.box_epsg4326 = list(np.flip(box_epsg4326))
        osm_mask = self.get_osm_mask(self.box_epsg4326, metadata["crs"], band_dict["B02"],
                                     {"lat": self.lat, "lon": self.lon},
                                     self.dir_ancil)
        osm_mask[osm_mask != 0] = 1
        osm_mask[osm_mask == 0] = np.nan
        band_stack_np = np.array([band_dict["B04"], band_dict["B03"], band_dict["B02"], band_dict["B08"]])
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
        band_stack_np_rescaled[band_stack_np_rescaled == 0] = np.nan
        return band_stack_np_rescaled

    def reveal_trucks(self, band_stack_np):
        """
        Method for detecting large moving vehicles, calls ratio-based detection and object delineation
        :param band_stack_np: numpy ndarray containing the pre-processed Sentinel-2 reflectance bands
        :return: GeoDataframe containing the detected boxes
        """
        t0 = datetime.now()
        if not isinstance(band_stack_np, np.ndarray):
            raise TypeError("'band_stack_np' must be of type numpy.ndarray")
        self.band_stack_np = band_stack_np
        self._calc_truck_measure()
        print("Duration: %s minutes" % ((datetime.now() - t0).total_seconds() / 60))
        return self.truck_measure

    def _calc_truck_measure(self):
        # mask out very upper reflectances and vegetation
        refl_mask = np.ones_like(self.band_stack_np[0])
        for i in range(self.band_stack_np.shape[0]):
            refl_mask *= np.int8(self.band_stack_np[i] < 0.25)
        refl_mask_stack = np.zeros_like(self.band_stack_np)
        for i in range(self.band_stack_np.shape[0]):
            refl_mask_stack[i] = np.int8(self.band_stack_np[i] > 0.15)
        mask = refl_mask + np.int8(np.sum(refl_mask_stack) < 3)  # all RGB high and highly saturated -> mask out
    #    mask *= np.int8(normalized_ratio(self.band_stack_np[3], self.band_stack_np[0]) < 0.4)
     #   self.band_stack_np[:, mask == 0] = np.nan
        # calculate RGB variance in windows
        w = 4
        offset = 20  # from micro window
        self.truck_measure = np.zeros((int(self.band_stack_np.shape[1] / w), int(self.band_stack_np.shape[2] / w)))
        for y in range(self.truck_measure.shape[0]):
            for x in range(self.truck_measure.shape[1]):
                y_low, x_low = y * w, x * w
                y_up, x_up = y_low + w, x_low + w
                if np.count_nonzero(~np.isnan(self.band_stack_np[2, y_low:y_low + w, x_low:x_low + w])) == 0:
                    continue
                y_low_macro, x_low_macro = int(np.clip(y_low - offset, 0, 1e+20)), int(
                    np.clip(x_low - offset, 0, 1e+20))
                macro_subset = self.band_stack_np[0:3, y_low_macro:y_up + offset, x_low_macro:x_up + offset].copy()
                macro_subset[np.isnan(macro_subset)] = 0
                micro_subset = self.band_stack_np[0:3, y_low:y_low + w, x_low:x_low + w]
                macro_var = np.nanvar(macro_subset, 0)
                macro_var_quantile = np.nanquantile(macro_var, [0.25])[0]
                self.truck_measure[y, x] = np.nanmean(np.nanvar(micro_subset, 0)) - macro_var_quantile

    def calibrate(self, station_counts_csv, station_name, calibration_csv, date, hour):
        road_type = 1
        station_meta = self.get_station_meta(station_name)
        station_counts = pd.read_csv(station_counts_csv, sep=";")
        station_point = Point([station_meta["x"], station_meta["y"]])
        minutes = (5, 10, 20)
        # compare number of detections in reach to the one of count station
        date_station_format = date[2:].replace("-", "")  # e.g. "2018-12-31" -> "181231"
        time_match = (station_counts["Datum"] == int(date_station_format)) * (station_counts["Stunde"] == hour)
        station_counts_hour = station_counts[time_match]
        trucks_r1 = station_counts_hour["Lkw_R1"]
        trucks_r2 = station_counts_hour["Lkw_R2"]
        lat_step = (np.max(self.lat) - np.min(self.lat)) / self.truck_measure.shape[0]
        lon_step = (np.max(self.lon) - np.min(self.lon)) / self.truck_measure.shape[1]
        new_lat = np.arange(np.min(self.lat), np.max(self.lat), lat_step)[::-1]
        new_lon = np.arange(np.min(self.lon), np.max(self.lon), lon_step)
        osm_mask = self.get_osm_mask(self.box_epsg4326, self.metadata["crs"], self.truck_measure,
                                     {"lat": new_lat, "lon": new_lon}, self.dir_ancil)
        osm_mask[osm_mask != road_type] = np.nan
        truck_measure_road_type = self.truck_measure.copy()
      #  truck_measure_road_type *= osm_mask
        try:
            calibration_pd = pd.read_csv(calibration_csv)
        except FileNotFoundError:
            calibration_pd = pd.DataFrame()
        for minute in minutes:
            hour_proportion = minute / 60
            traveled_distance = (90 * hour_proportion) * 1000  # meters
            min_lat, max_lat = station_point.y - traveled_distance, station_point.y + traveled_distance
            min_lon, max_lon = station_point.x - traveled_distance, station_point.x + traveled_distance
            lat_min_deviation = np.abs(new_lat - min_lat)
            lon_min_deviation = np.abs(new_lon - min_lon)
            lat_max_deviation = np.abs(new_lat - max_lat)
            lon_max_deviation = np.abs(new_lon - max_lon)
            miny = np.where(lat_max_deviation == np.min(lat_max_deviation))[0][0]
            minx = np.where(lon_min_deviation == np.min(lon_min_deviation))[0][0]
            maxy = np.where(lat_min_deviation == np.min(lat_min_deviation))[0][0] + 1
            maxx = np.where(lon_max_deviation == np.min(lon_max_deviation))[0][0] + 1
            truck_measure_subset = truck_measure_road_type[miny:maxy, minx:maxx]
            idx = len(calibration_pd)
            d1, d2 = float(trucks_r1 * hour_proportion), float(trucks_r2 * hour_proportion)
            calibration_pd.loc[idx, "trucks_direction1"] = d1
            calibration_pd.loc[idx, "trucks_direction2"] = d2
            calibration_pd.loc[idx, "trucks_sum"] = d1 + d2
            calibration_pd.loc[idx, "truck_measure_sum"] = np.nansum(truck_measure_subset)
            calibration_pd.loc[idx, "truck_measure_mean"] = np.nanmean(truck_measure_subset)
            calibration_pd.loc[idx, "hour"] = hour
            calibration_pd.loc[idx, "minute"] = minute
        print()

    @staticmethod
    def get_osm_mask(bbox, crs, reference_arr, lat_lon_dict, dir_out):
        osm_file = get_roads(bbox, ["motorway", "trunk", "primary"], OSM_BUFFER,
                             dir_out, str(bbox).replace(", ", "_")[1:-1] + "_osm_roads", str(crs))
        osm_vec = gpd.read_file(osm_file)
        ref_xr = xr.DataArray(data=reference_arr, coords=lat_lon_dict, dims=["lat", "lon"])
        osm_raster = rasterize_osm(osm_vec, ref_xr).astype(np.float32)
        return osm_raster

    @staticmethod
    def get_station_meta(station_name):
        """
        gets UTM coordinates of BAST traffic count station from BAST webpage
        :param station_name: str in the format "Theeßen (3810)"
        :return: dict x and y UTM coordinate
        """
        response = requests.get(BAST_URL)
        page_str = response.text
        station_section = page_str.split("addBKGPoempel")[2:]
        utm_zone = 32
        for station_row in station_section:
            try:
                name = station_row.split(": ")[1].split(",")[0]
            except IndexError:
                continue
            if name == station_name:
                # get color as road type marker
                try:
                    color = station_row.split(",red,")
                    road_type = "motorway"
                except IndexError:
                    road_type = "primary"
                try:
                    # get coordinates
                    coordinates = station_row.split("results, ")[1].split(",")
                    x, y = coordinates[0].split('"')[1], coordinates[1]
                except IndexError:
                    continue
                try:
                    return {"x": float(x), "y": float(y), "utm_zone": utm_zone, "road_type": road_type}
                except ValueError:
                    continue
        print("No station name match")
