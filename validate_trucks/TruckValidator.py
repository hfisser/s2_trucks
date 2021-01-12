import os
import time
import requests
import shutil
import numpy as np
import rasterio as rio
import geopandas as gpd
import pandas as pd
import xarray as xr
from glob import glob
from shutil import copyfile
from shapely.geometry import Point
from shapely.geometry.linestring import LineString
from rasterio.merge import merge

from osm_utils.utils import get_roads, rasterize_osm
from array_utils.points import raster_to_points
from array_utils.geocoding import lat_from_meta, lon_from_meta
from SentinelHubDataAccess.SentinelHub import SentinelHub, DataCollection
from detect_trucks.RandomForestTrucks import RFTruckDetector

dir_main = "F:\\Masterarbeit\\DLR\\project\\1_truck_detection"
dir_validation = os.path.join(dir_main, "validation")
dir_osm = os.path.join(dir_main, "code", "detect_trucks", "AUXILIARY", "osm")
truth_csv = os.path.join(dir_main, "truth", "spectra_ml.csv")
aois_file = os.path.join(dir_validation, "data", "BAST", "validation_aois.gpkg")


SH_CREDENTIALS_FILE = os.path.join("F:" + os.sep + "sh", "sh.txt")
BAST_URL = "https://www.bast.de/BASt_2017/DE/Verkehrstechnik/Fachthemen/v2-verkehrszaehlung/Aktuell/" \
           "zaehl_aktuell_node.html;jsessionid=63610843F87B77C24C4320BC4EAD6647.live21304"
NAME_DATE = "Datum"
NAME_HOUR = "Stunde"
NAME_TR1 = "Lkw_R1"
NAME_TR2 = "Lkw_R2"

OSM_BUFFER = 25
hour, minutes, year = 11, 10, 2018

stations = dict()
stations_pd = pd.read_csv(os.path.join(os.path.dirname(aois_file), "validation_stations.csv"), sep=";")
for station_name, start in zip(stations_pd["Name"], stations_pd["S2_date_start"]):
    if "AS Dierdorf" in station_name:
        split = start.split(".")
        date = "-".join([split[2], split[1], split[0]])
        stations[station_name] = [date, date]


class Validator:
    def __init__(self, station_name, station_aois_file, dir_validation_home, dir_osm_data):
        aois = gpd.read_file(station_aois_file)
        self.dirs = {"validation": dir_validation_home, "osm": dir_osm_data,
                     "station_counts": os.path.join(dir_validation_home, "data", "BAST", "station_counts"),
                     "s2": os.path.join(dir_validation_home, "data", "s2"),
                     "detections": os.path.join(dir_validation_home, "detections")}
        for directory in self.dirs.values():
            if not os.path.exists(directory):
                os.mkdir(directory)
        self.validation_file = os.path.join(self.dirs["validation"], "validation_run.csv")
        self.station_name = station_name
        attribute_given = any(["(" + s + ")" in self.station_name for s in ["N", "E", "S", "W"]])
        self.station_name_clear = self.station_name.split(" (")[0]
        alternative_split = self.station_name.split(") (")[0].split(" (")[0]
        self.station_name_clear = alternative_split if attribute_given else self.station_name_clear
        self.station_name_clear = self.station_name_clear.replace(" ", "_")
        self.station_meta = self.get_station_meta(station_name)
        bbox = aois[aois["Name"] == station_name].geometry.bounds
        self.bbox_epsg4326 = (float(bbox.miny), float(bbox.minx), float(bbox.maxy), float(bbox.maxx))  # min lat, min lon, max lat, max lon
        self.crs = aois.crs
        self.lat, self.lon = None, None
        self.detections, self.osm_roads = None, None
        self.date = None
        self.detections_file, self.station_file, self.s2_data_file, = "", "", None

    def detect(self, period):
        self.date = period[0]
        band_names, resolution, folder = ["B04", "B03", "B02", "B08", "CLM"], 10, ""
        dir_save_archive = os.path.join(self.dirs["s2"], "archive")
        for directory in glob(os.path.join(os.path.dirname(dir_save_archive), "*")):  # keep this clean, only archive should be retained
            if directory != dir_save_archive:
                shutil.rmtree(directory)
        if not os.path.exists(dir_save_archive):
            os.mkdir(dir_save_archive)
        sh = SentinelHub()
        sh.set_credentials(SH_CREDENTIALS_FILE)
        sh_bbox = (self.bbox_epsg4326[1], self.bbox_epsg4326[0], self.bbox_epsg4326[3], self.bbox_epsg4326[2])  # diff. order
        splitted_boxes = sh.split_box(sh_bbox, resolution)  # bbox may be too large, hence split (if too large)
        lats, lons = [], []
        files = []
        merged_file = os.path.join(dir_save_archive, "s2_bands_%s_%s_%s_merged.tiff" % (self.station_name_clear,
                                                                                        period[0],
                                                                                        period[1]))
        for i, bbox in enumerate(splitted_boxes):
            curr_s2_data_file = os.path.join(dir_save_archive, "s2_bands_%s_%s_%s_box%s.tiff" % (self.station_name_clear,
                                                                                                 period[0],
                                                                                                 period[1], i))
            files.append(curr_s2_data_file)
            if not os.path.exists(merged_file) and not os.path.exists(curr_s2_data_file):
                band_stack, dir_data = sh.get_data(bbox, period, DataCollection.SENTINEL2_L2A, band_names,
                                                   resolution, self.dirs["s2"])
                folders = glob(os.path.join(dir_data, "*"))
                folders.remove(dir_save_archive)
                if len(folders) > 1:
                    print("Several files, don't know which to read from %s" % self.dirs["s2"])
                    raise FileNotFoundError
                else:
                    folder = folders[0]
                    reflectance_file = copyfile(glob(os.path.join(folder, "*.tiff"))[0], curr_s2_data_file)
                    if os.path.exists(folder) and os.path.exists(curr_s2_data_file):
                        shutil.rmtree(folder)  # remove original download file
                    else:
                        time.sleep(10)
                        shutil.rmtree(folder)
        if not os.path.exists(merged_file):
            with rio.open(files[0], "r") as src:
                meta = src.meta  # get meta
            merged_stack, transform = merge(files)
            meta = dict(transform=transform, height=merged_stack.shape[1], width=merged_stack.shape[2],
                        count=merged_stack.shape[0], driver="GTiff", dtype=merged_stack.dtype,
                        crs=meta["crs"])
            with rio.open(merged_file, "w", **meta) as tgt:
                for i in range(merged_stack.shape[0]):
                    tgt.write(merged_stack[i], i+1)
        detections_file = os.path.join(self.dirs["detections"], "s2_detections_%s_%s.gpkg" %
                                       (self.date, self.station_name_clear))
        try:
            with rio.open(merged_file, "r") as src:
                meta = src.meta
                band_stack_np = np.zeros((meta["height"], meta["width"], meta["count"]))
                for b in range(band_stack_np.shape[2]):
                    band_stack_np[:, :, b] = src.read(b + 1)
        except rio.errors.RasterioIOError as e:
            raise e
        band_stack_np = band_stack_np.swapaxes(0, 2).swapaxes(1, 2)  # z, y, x
        detector = RFTruckDetector()
        band_stack = detector.read_bands(merged_file)
        # transform to EPSG:4326
        t, epsg_4326 = meta["transform"], "EPSG:4326"
        bands_preprocessed = detector.preprocess_bands(band_stack, dir_osm)
        detector.train(bands_preprocessed, truth_csv, 10)
        prediction = detector.predict()
        prediction_boxes = detector.extract_objects(prediction)
        detector.prediction_boxes_to_gpkg(prediction_boxes, detections_file)
        station_folder = "zst" + self.station_name.split("(")[1].split(")")[0]
        wrong = len(station_folder) == 4
        station_folder = "zst" + self.station_name.split(") ")[1].split("(")[1][0:-1] if wrong else station_folder
        self.station_file = os.path.join(self.dirs["station_counts"], station_folder, station_folder + "_%s.csv" %
                                         str(year))
        self.detections_file = detections_file
        lats.append(lat_from_meta(meta))
        lons.append(lon_from_meta(meta))
        detector, band_stack_np = None, None
        self.lat = np.sort(np.unique(lats))[::-1]
        self.lon = np.sort(np.unique(lons))

    def validate(self):
        try:
            validation_pd = pd.read_csv(self.validation_file)
        except FileNotFoundError:
            validation_pd = pd.DataFrame()
        speed = 90
        self.detections = gpd.read_file(self.detections_file)
        self.prepare_s2_counts()
        hour_proportion = (minutes / 60)
        distance_traveled = hour_proportion * speed
        station_counts = pd.read_csv(self.station_file, sep=";")
        station_point = Point([self.station_meta["x"], self.station_meta["y"]])
        station_buffer = station_point.buffer(distance_traveled * 1000)
        station_buffer_gpd = gpd.GeoDataFrame({"id": [0]}, geometry=[station_buffer], crs=self.detections.crs)
        detections_in_reach = gpd.overlay(self.detections, station_buffer_gpd, "intersection")
    #    ref = xr.DataArray(np.zeros((len(self.lat), len(self.lon))),  coords={"lat": self.lat, "lon": self.lon},
     #                      dims=("lat", "lon"))
      #  osm_raster_aggregated = self.max_aggregate(rasterize_osm(self.osm_roads, ref), 200)
       # osm_raster_aggregated[~np.isnan(osm_raster_aggregated) * osm_raster_aggregated != 0] = 1
        # downsample lat lon to aggregated osm roads
   #     shape_new = osm_raster_aggregated.shape
     #   lat_aggregated = np.flip(np.arange(self.lat[-1], self.lat[0], (self.lat[0] - self.lat[-1]) / shape_new[0]))
      #  lon_aggregated = np.arange(self.lon[0], self.lon[-1], (self.lon[-1] - self.lon[0]) / shape_new[1])
        # create point grid in aoi
     #   road_points = raster_to_points(osm_raster_aggregated, {"lat": lat_aggregated, "lon": lon_aggregated},
      #                                 "id", "EPSG:" + str(self.detections.crs.to_epsg()))
  #      detections_in_reach = []
   #     for detection in self.detections.iterrows():
    #        detection = detection[1]
     ##       detection_point = detection.geometry.centroid
       #     detection_y, detection_x = detection_point.y, detection_point.x
        #    station_y, station_x = station_point.y, station_point.x
         #   traveled_distance = speed * hour_proportion
          #  line_to_detection = LineString([detection_point, station_point])
           # # passed by the station in number of minutes
   #         distance_matching = traveled_distance >= line_to_detection.length / 1000
    #        if not distance_matching:
     #           continue
      #      # check direct line to detection and compare with vehicle heading (only include if heading away)
       #     direct_vector_to_detection = self.calc_vector([station_y, station_x], [detection_y, detection_x])
        #    # calculate in which direction the station is
         #   direction_bins = np.arange(0, 359, 22.5, dtype=np.float32)
          #  station_direction = self.calc_vector_direction_in_degree(direct_vector_to_detection)
           # diffs = np.abs(direction_bins - station_direction)
            #lowest_diff_idx = np.where(diffs == diffs.min())[0][0]
            # get range of directions (180°)
  #          up = lowest_diff_idx + 4
   #         up = up - len(direction_bins) if up >= len(direction_bins) else up
    #        direction_range = np.sort([direction_bins[lowest_diff_idx - 4], direction_bins[int(up)]])
     #       # check if vehicle is traveling from station (count) or to station (drop)
      #      direction_matching = direction_range[0] < detection["direction_degree"] < direction_range[1]
       #     #if direction_matching:
        #    detections_in_reach.append(detection)
        # compare number of detections in reach to the one of count station
        date_station_format = self.date[2:].replace("-", "")  # e.g. "2018-12-31" -> "181231"
        time_match = (station_counts["Datum"] == int(date_station_format)) * (station_counts["Stunde"] == hour)
        station_counts_hour = station_counts[time_match]
        idx = len(validation_pd)
        for key, value in {"station_file": self.station_file, "s2_counts_file": self.s2_data_file,
                           "detections_file": self.detections_file,
                           "hour": hour, "n_minutes": minutes, "s2_counts": len(detections_in_reach) / 2}.items():
            validation_pd.loc[idx, key] = [value]
        for column in station_counts_hour.columns[9:]:  # counts from station
            # add counts proportional to number of minutes
            try:
                validation_pd.loc[idx, column] = [station_counts_hour[column].iloc[0] * hour_proportion]
            except TypeError:
                validation_pd.loc[idx, column] = np.nan
        validation_pd.to_csv(self.validation_file)

    def prepare_s2_counts(self):
        osm_file = get_roads(list(self.bbox_epsg4326), ["motorway", "trunk", "primary"], OSM_BUFFER,
                             self.dirs["osm"],
                             str(self.bbox_epsg4326).replace(", ", "_")[1:-1].replace(".", "_")
                             + "_osm_roads", "EPSG:" + str(self.detections.crs.to_epsg()))
        osm_roads = gpd.read_file(osm_file)
        # subset to road type of station
        station_road_type = [self.station_meta["road_type"]]
        if station_road_type[0] == "primary":
            station_road_type.append("trunk")
        osm_subset = []
        for road_type in np.unique(station_road_type):
            subset = osm_roads[osm_roads["osm_value"] == road_type]
            if len(subset) > 0:
                osm_subset.append(subset)
        self.osm_roads = pd.concat(osm_subset)
        osm_union = self.osm_roads.unary_union
        # subset s2 counts to road type
        detections_within = []
        for row in self.detections.iterrows():
            row = row[1]
            if row.geometry.intersects(osm_union):
                detections_within.append(row)
        self.detections = gpd.GeoDataFrame(detections_within, crs=self.detections.crs)

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
    def calc_vector_direction_in_degree(vector):
        # [1,1] -> 45°
        # [-1,1] -> 135°
        # [-1,-1] -> 225°
        # [1,-1] -> 315°
        y_offset = 90 if vector[0] < 0 else 0
        x_offset = 90 if vector[1] < 0 else 0
        offset = 180 if y_offset == 0 and x_offset == 90 else 0
        if vector[0] == 0:
            direction = 0.
        else:
            direction = np.degrees(np.arctan(np.abs(vector[1]) / np.abs(vector[0])))
        direction += offset + y_offset + x_offset
        return direction

    @staticmethod
    def max_aggregate(in_arr, scale):
        scale -= 1
        y_out, x_out = int(in_arr.shape[0] / scale), int(in_arr.shape[1] / scale)
        out_arr = np.zeros((y_out, x_out))
        for y in range(out_arr.shape[0]):
            for x in range(out_arr.shape[1]):
                y_low, x_low = y * scale, x * scale
                y_up, x_up = y_low + scale + 1, x_low + scale + 1
                out_arr[y, x] = int(np.count_nonzero(in_arr[y_low:y_up, x_low:x_up]) > 20)
        return out_arr


if __name__ == "__main__":
    try:
        os.remove(os.path.join(dir_validation, "validation_run.csv"))
    except FileNotFoundError:
        pass
    for station, acquisition_period in stations.items():
        print("Station: %s" % station)
        validator = Validator(station, aois_file, dir_validation, dir_osm)
        validator.detect(acquisition_period)
        validator.validate()
    validation = pd.read_csv(os.path.join(dir_validation, "validation_run.csv"))
