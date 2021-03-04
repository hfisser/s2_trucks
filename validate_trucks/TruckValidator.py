import os
import requests
import shutil
import numpy as np
import rasterio as rio
import geopandas as gpd
import pandas as pd
from glob import glob
from shapely.geometry import Point, box
from fiona.errors import DriverError
from datetime import date, timedelta

from osm_utils.utils import get_roads
from array_utils.geocoding import lat_from_meta, lon_from_meta
from SentinelHubDataAccess.SentinelHub import SentinelHub, DataCollection
from detect_trucks.RandomForestTrucks import RFTruckDetector

dir_main = "F:\\Masterarbeit\\DLR\\project\\1_truck_detection"
dir_data = os.path.join(dir_main, "data")
dir_validation = os.path.join(dir_main, "validation")
dir_labels = os.path.join(dir_data, "labels")
dir_osm = os.path.join(dir_main, "code", "detect_trucks", "AUXILIARY", "osm")
dir_training = os.path.join(dir_main, "training")
dir_s2_subsets = os.path.join(dir_data, "s2", "subsets")
truth_csv = os.path.join(dir_main, "truth", "spectra_ml.csv")
aois_file = os.path.join(dir_validation, "data", "BAST", "validation_aois.gpkg")
boxes_validation_file = os.path.join(dir_validation, "boxes_validation.csv")

SH_CREDENTIALS_FILE = os.path.join("F:" + os.sep + "sh", "sh.txt")
BAST_URL = "https://www.bast.de/BASt_2017/DE/Verkehrstechnik/Fachthemen/v2-verkehrszaehlung/Aktuell/" \
           "zaehl_aktuell_node.html;jsessionid=63610843F87B77C24C4320BC4EAD6647.live21304"
NAME_DATE = "Datum"
NAME_HOUR = "Stunde"
NAME_TR1 = "Lkw_R1"
NAME_TR2 = "Lkw_R2"

OSM_BUFFER = 20
hour, minutes, year = 10, 15, 2018

validate = "boxes"
validate = "bast"

stations = dict()
bast_dir = os.path.dirname(aois_file)
stations_pd = pd.read_csv(os.path.join(bast_dir, "validation_stations.csv"), sep=";")
for station_name, start, end in zip(stations_pd["Name"], stations_pd["S2_date_start"], stations_pd["S2_date_end"]):
    start, end = start.split("."), end.split(".")
    stations[station_name] = ["-".join([split[2], split[1], split[0]]) for split in [start, end]]


class Validator:
    def __init__(self, station_denotation, station_aois_file, dir_validation_home, dir_osm_data):
        aois = gpd.read_file(station_aois_file)
        self.dirs = {"validation": dir_validation_home, "osm": dir_osm_data,
                     "station_counts": os.path.join(dir_validation_home, "data", "BAST", "station_counts"),
                     "s2": os.path.join(dir_validation_home, "data", "s2"),
                     "detections": os.path.join(dir_validation_home, "detections")}
        for directory in self.dirs.values():
            if not os.path.exists(directory):
                os.mkdir(directory)
        self.station_name = station_denotation
        attribute_given = any(["(" + s + ")" in self.station_name for s in ["N", "E", "S", "W"]])
        self.station_name_clear = self.station_name.split(" (")[0]
        alternative_split = self.station_name.split(") (")[0].split(" (")[0]
        self.station_name_clear = alternative_split if attribute_given else self.station_name_clear
        self.station_name_clear = self.station_name_clear.replace(" ", "_")
        self.validation_file = os.path.join(self.dirs["validation"], "validation_run_%s.csv" % self.station_name_clear)
        self.station_name = self.station_name.split("(1)")[0] if self.station_name.endswith("(1)") else self.station_name
        self.station_meta = self.get_station_meta(self.station_name)
        bbox = aois[aois["Name"] == self.station_name].geometry.bounds
        self.bbox_epsg4326 = (float(bbox.miny), float(bbox.minx), float(bbox.maxy), float(bbox.maxx))  # min lat, min lon, max lat, max lon
        self.crs = aois.crs
        self.lat, self.lon = None, None
        self.detections, self.osm_roads = None, None
        self.date = None
        self.detections_file, self.s2_data_file, = "", None

    def validate_acquisition_wise(self, period):
        dates_between = self.generate_process_periods(period)
        for sub_period in dates_between:
            self.date = sub_period[0]
            band_names, resolution, folder = ["B04", "B03", "B02", "B08", "CLM"], 10, ""
            dir_save_archive = os.path.join(self.dirs["s2"], "archive")
            # keep this clean, only archive should be retained
            for directory in glob(os.path.join(os.path.dirname(dir_save_archive), "*")):
                if directory != dir_save_archive:
                    shutil.rmtree(directory)
            if not os.path.exists(dir_save_archive):
                os.mkdir(dir_save_archive)
            sh_bbox = (self.bbox_epsg4326[1], self.bbox_epsg4326[0], self.bbox_epsg4326[3], self.bbox_epsg4326[2])
            merged_file = os.path.join(dir_save_archive, "s2_bands_%s_%s_%s_merged.tiff" % (self.station_name_clear,
                                                                                            sub_period[0],
                                                                                            sub_period[1]))
            kwargs = dict(bbox=sh_bbox, period=sub_period, dataset=DataCollection.SENTINEL2_L2A,
                          bands=["B04", "B03", "B02", "B08"], resolution=resolution, dir_save=self.dirs["s2"],
                          merged_file=merged_file, mosaicking_order="leastCC")
            sh = SentinelHub()
            sh.set_credentials(SH_CREDENTIALS_FILE)
            data_available = sh.data_available(kwargs)
            if data_available:
                print("Processing: %s" % sub_period[0])
                kwargs_copy = kwargs.copy()
                kwargs_copy["bands"] = ["CLM"]  # get cloud mask in order to check if low cloud coverage
                kwargs_copy["merged_file"] = os.path.join(dir_save_archive, "clm.tiff")
                clm, data_folder = sh.get_data(**kwargs_copy)  # get only cloud mask
                has_obs = self.has_observations(kwargs_copy["merged_file"])
                try:
                    os.remove(kwargs_copy["merged_file"])  # cloud mask
                except FileNotFoundError:
                    pass
                if has_obs:
                    band_stack, folder = sh.get_data(**kwargs)  # get full data
                    detections_file = os.path.join(self.dirs["detections"], "s2_detections_%s_%s.gpkg" %
                                                   (self.date, self.station_name_clear))
                    detector = RFTruckDetector()
                    band_stack = detector.read_bands(merged_file)
                    detector.preprocess_bands(band_stack)
                    detector.mask_clouds(clm)
                    detector.train()
                    prediction = detector.predict()
                    prediction_boxes = detector.extract_objects(prediction)
                    try:
                        detector.prediction_boxes_to_gpkg(prediction_boxes, detections_file)
                    except ValueError:
                        pass
                    station_folder = "zst" + self.station_name.split("(")[1].split(")")[0]
                    wrong = len(station_folder) == 4
                    station_folder = "zst" + self.station_name.split(") ")[1].split("(")[1][0:-1] if wrong else station_folder
                    station_file = os.path.join(self.dirs["station_counts"], station_folder, station_folder + "_%s.csv" %
                                                str(year))
                    self.detections_file = detections_file
                    with rio.open(merged_file, "r") as src:
                        meta = src.meta
                    self.lat = lat_from_meta(meta)
                    self.lon = lon_from_meta(meta)
                    detector, band_stack_np = None, None
                    # run comparison
                    self.validate_with_bast(sub_period[0], detections_file, station_file, merged_file)

    def validate_with_bast(self, acquisition_date, detections_file, station_file, s2_data_file):
        try:
            validation_pd = pd.read_csv(self.validation_file)
        except FileNotFoundError:
            validation_pd = pd.DataFrame()
        speed = 80
        try:
            self.detections = gpd.read_file(detections_file)
        except DriverError:
            self.detections = gpd.GeoDataFrame()
        else:
            self.prepare_s2_counts()
        hour_proportion = (minutes / 60)
        distance_traveled = hour_proportion * speed
        station_counts = pd.read_csv(station_file, sep=";")
        station_point = Point([self.station_meta["x"], self.station_meta["y"]])
        buffer_distance = distance_traveled * 1000
        station_buffer = station_point.buffer(buffer_distance)
        if "Braunschweig-Flughafen" in self.station_name:  # rectangluar buffer due to another highway within buffer
            sy, sx = station_point.y, station_point.x
            station_buffer = box(sx - buffer_distance, sy - 1500, sx + buffer_distance, sy + 3000)
        station_buffer_gpd = gpd.GeoDataFrame({"id": [0]}, geometry=[station_buffer], crs=self.detections.crs)
        detections_in_buffer = gpd.overlay(self.detections, station_buffer_gpd, "intersection")
        b = np.float32([station_point.x, station_point.y])
        s2_direction1, s2_direction2 = 0, 0
        for row in detections_in_buffer.iterrows():
            detection_point = row[1].geometry.centroid
            a = np.float32([detection_point.x, detection_point.y])
            station_direction, heading = self.calc_vector_direction_in_degree(a - b), row[1].direction_degree
            # abs. difference between calculated heading and station direction must be < 90 (heading away from station)
            # this only works when the road is relatively balanced like highways, not many curves
            match = np.abs(station_direction - heading) < 90  # first try this
            if match:
                pass
            else:
                smaller_180 = [station_direction <= 180, heading <= 180]  # avoid degree edge 360 -> 1
                station_direction = station_direction + 360 if smaller_180[0] else station_direction
                heading = heading + 360 if smaller_180[1] else heading
                match = np.abs(station_direction - heading) < 90
            heading = row[1].direction_degree
            if match and heading >= 180:
                s2_direction1 += 1
            elif match and heading < 180:
                s2_direction2 += 1
        # compare number of detections in reach to the one of count station
        date_station_format = self.date[2:].replace("-", "")  # e.g. "2018-12-31" -> "181231"
        time_match = (station_counts["Datum"] == int(date_station_format)) * (station_counts["Stunde"] == hour)
        station_counts_hour = station_counts[time_match]
        idx = len(validation_pd)
        for key, value in {"station_file": station_file, "s2_counts_file": s2_data_file,
                           "detections_file": detections_file,
                           "date": acquisition_date, "hour": hour, "n_minutes": minutes,
                           "s2_direction1": s2_direction1, "s2_direction2": s2_direction2}.items():
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
        station_road_type.append(station_road_type[0] + "_link")
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

    def validate_boxes(self):
        tiles_pd = pd.read_csv(os.path.join(dir_training, "tiles.csv"), sep=";")
        try:
            os.remove(boxes_validation_file)
        except FileNotFoundError:
            pass
        boxes_validation_pd = pd.DataFrame()
        tiles = list(tiles_pd["validation_tiles"])
        for tile in tiles:
            print(tile)
            try:
                imgs = np.array(glob(dir_s2_subsets + os.sep + "*" + tile + "*.tif"))
            except TypeError:  # nan
                continue
            lens = np.int32([len(x) for x in imgs])
            img_file = imgs[np.where(lens == lens.max())[0]][0]
            name = os.path.basename(img_file).split(".tif")[0]
            print(img_file)
            # read labels
            validation_boxes = gpd.read_file(os.path.join(dir_labels,
                                                          os.path.basename(img_file).split("_y0")[0] + ".gpkg"))
            prediction_boxes_file = os.path.join(self.dirs["detections"], name + "_boxes")
            try:
                prediction_boxes = gpd.read_file(prediction_boxes_file)  # read if yet processed
            except DriverError:
                # detect on whole array
                rf_td = RFTruckDetector()
                band_data = rf_td.read_bands(img_file)
                lat, lon = lat_from_meta(rf_td.meta), lon_from_meta(rf_td.meta)
                extent = validation_boxes.total_bounds  # process only subset where boxes given
                diff_ymin, diff_ymax = np.abs(lat - extent[3]), np.abs(lat - extent[1])
                diff_xmin, diff_xmax = np.abs(lon - extent[0]), np.abs(lon - extent[2])
                ymin, ymax = np.argmin(diff_ymin), np.argmin(diff_ymax)
                xmin, xmax = np.argmin(diff_xmin), np.argmin(diff_xmax)
                rf_td.preprocess_bands(band_data, {"ymin": ymin, "xmin": xmin, "ymax": ymax + 1, "xmax": xmax + 1})
                rf_td.train()
                prediction_array = rf_td.predict()
                prediction_boxes = rf_td.extract_objects(prediction_array)
                rf_td.prediction_raster_to_gtiff(prediction_array,
                                                 os.path.join(self.dirs["detections"], name + "_raster"))
                rf_td.prediction_boxes_to_gpkg(prediction_boxes, prediction_boxes_file)
            producer_n, user_n = 0, 0
            percentage_intersection = []
            for prediction_box in prediction_boxes.geometry:
                for validation_box in validation_boxes.geometry:
                    if prediction_box.intersects(validation_box):
                        difference = prediction_box.difference(validation_box)
                        percentage_intersection.append((prediction_box.area - difference.area) / prediction_box.area)
                        producer_n += 1
                        break
            for validation_box in validation_boxes.geometry:
                for prediction_box in prediction_boxes.geometry:
                    if validation_box.intersects(prediction_box):
                        user_n += 1
                        break
            prediction_array, band_data, rf_td = None, None, None
            row_idx = len(boxes_validation_pd)
            boxes_validation_pd.loc[row_idx, "detection_file"] = prediction_boxes_file
            boxes_validation_pd.loc[row_idx, "producer_percentage"] = producer_n / len(prediction_boxes) * 100
            boxes_validation_pd.loc[row_idx, "user_percentage"] = user_n / len(validation_boxes) * 100
            boxes_validation_pd.loc[row_idx, "n_prediction_boxes"] = len(prediction_boxes)
            boxes_validation_pd.loc[row_idx, "n_validation_boxes"] = len(validation_boxes)
            boxes_validation_pd.loc[row_idx, "mean_area_intersection"] = np.mean(percentage_intersection)
        boxes_validation_pd.to_csv(boxes_validation_file)

    @staticmethod
    def get_station_meta(bast_station_name):
        """
        gets UTM coordinates of BAST traffic count station from BAST webpage
        :param bast_station_name: str in the format "Theeßen (3810)"
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
            if name == bast_station_name:
                # get color as road type marker
                color = station_row.split(",red,")
                if len(color) == 1:
                    road_type = "primary"
                else:
                    road_type = "motorway"
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

    @staticmethod
    def generate_process_periods(period_of_interest):
        d0 = [int(part) for part in period_of_interest[0].split("-")]
        d1 = [int(part) for part in period_of_interest[1].split("-")]
        d0_date, d1_date = date(d0[0], d0[1], d0[2]), date(d1[0], d1[1], d1[2])
        periods = []
        for days in range((d1_date - d0_date).days):
            d = (d0_date + timedelta(days)).strftime("%Y-%m-%d")
            periods.append([d, d])
        return periods

    @staticmethod
    def has_observations(cloud_mask_file):
        min_valid_percentage = 95
        rf_td = RFTruckDetector()
        clm = rf_td.read_bands(cloud_mask_file)
        clm[clm == 0] = 1  # nocloud=0
        clm[clm == 255] = 0  # cloud=255
        pseudo_band = np.random.random(len(clm.flatten())).reshape((clm.shape[0], clm.shape[1]))
        pseudo_band_stack = np.float32([pseudo_band] * 4)  # run pseudo preprocessing due to OSM masking
        rf_td.preprocess_bands(pseudo_band_stack)
        n_valid = np.count_nonzero(~np.isnan(rf_td.variables[0]))  # n valid pixels masked to OSM roads
        rf_td.mask_clouds(clm)
        n_valid_masked = np.count_nonzero(~np.isnan(rf_td.variables[0]))  # n valid pixels cloud-masked
        valid_percentage = (n_valid_masked / n_valid) * 100
        rf_td, clm = None, None
        return valid_percentage < min_valid_percentage


if __name__ == "__main__":
    if validate == "bast":
        try:
            os.remove(os.path.join(dir_validation, "validation_run.csv"))
        except FileNotFoundError:
            pass
    else:
        try:
            os.remove(boxes_validation_file)
        except FileNotFoundError:
            pass
    for station, acquisition_period in stations.items():
        validator = Validator(station, aois_file, dir_validation, dir_osm)
        if validate == "boxes":
            validator.validate_boxes()
            break
        else:
            print("%s\nStation: %s" % ("-" * 50, station))
            validator.validate_acquisition_wise(acquisition_period)
