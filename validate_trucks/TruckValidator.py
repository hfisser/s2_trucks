import os
import requests
import shutil
import numpy as np
import rasterio as rio
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from matplotlib import rcParams
from shapely.geometry import Point, box
from fiona.errors import DriverError
from datetime import date, timedelta
from scipy.stats import linregress
from sklearn.metrics import auc
from osm_utils.utils import get_roads
from array_utils.geocoding import lat_from_meta, lon_from_meta
from SentinelHubDataAccess.SentinelHub import SentinelHub, DataCollection
from detect_trucks.RandomForestTrucks import RFTruckDetector

rcParams["font.serif"] = "Times New Roman"
rcParams["font.family"] = "serif"

dir_main = "F:\\Masterarbeit\\DLR\\project\\1_truck_detection"
dir_data = os.path.join(dir_main, "data")
dir_validation = os.path.join(dir_main, "validation")
dir_validation_plots = os.path.join(dir_validation, "plots")
dir_labels = os.path.join(dir_data, "labels")
dir_osm = os.path.join(dir_main, "code", "detect_trucks", "AUXILIARY", "osm")
dir_training = os.path.join(dir_main, "training")
dir_s2_subsets = os.path.join(dir_data, "s2", "subsets")
#dir_s2_subsets = "G:\\subsets_validation"

truth_csv = os.path.join(dir_main, "truth", "spectra_ml.csv")
aois_file = os.path.join(dir_validation, "data", "BAST", "validation_aois1.gpkg")
boxes_validation_file = os.path.join(dir_validation, "boxes_validation.csv")

SH_CREDENTIALS_FILE = os.path.join("F:" + os.sep + "sh", "sh.txt")
BAST_URL = "https://www.bast.de/BASt_2017/DE/Verkehrstechnik/Fachthemen/v2-verkehrszaehlung/Aktuell/" \
           "zaehl_aktuell_node.html;jsessionid=63610843F87B77C24C4320BC4EAD6647.live21304"
NAME_DATE = "Datum"
NAME_HOUR = "Stunde"
NAME_TR1 = "Lzg_R1"
NAME_TR2 = "Lzg_R2"
S2_COLOR, BAST_COLOR = "#611840", "#3e6118"

osm_buffer = 20
hour, minutes, year = 10, 10, 2018

validate = "boxes"
validate = "bast"

stations = dict()
bast_dir = os.path.dirname(aois_file)
stations_pd = pd.read_csv(os.path.join(bast_dir, "validation_stations.csv"), sep=";")
for station_name, start, end in zip(stations_pd["Name"], stations_pd["S2_date_start"], stations_pd["S2_date_end"]):
    if "Winnweiler" in station_name:
        continue
    else:
        start, end = start.split("."), end.split(".")
        stations[station_name] = ["-".join([split[2], split[1], split[0]]) for split in [start, end]]


bs_flughafen = "Braunschweig-Flughafen"
additional_stations = {"Braunschweig": bs_flughafen,
                       "Wolfsburg": bs_flughafen,
                       "Hoiersdorf": bs_flughafen,
                       "Hinrichshagen": "Strasburg",
                       "Sottrum": "Bockel"}


class Validator:
    def __init__(self, station_denotation, station_aois_file, dir_validation_home, dir_osm_data):
        print(station_denotation)
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
        bbox = aois[aois["Name"] == self.station_name].geometry.bounds.values.flatten()
        if len(bbox) == 0:
            contains_name = np.array([self.station_name_clear in curr_name for curr_name in aois["Name"]])
            name_match = np.where(contains_name)[0][0]
            bbox = aois.iloc[name_match].geometry.bounds
        self.bbox_epsg4326 = (float(bbox[1]), float(bbox[0]), float(bbox[3]), float(bbox[2]))  # min lat, min lon, max lat, max lon
        self.crs = aois.crs
        self.lat, self.lon = None, None
        self.detections, self.osm_roads = None, None
        self.date = None
        self.detections_file, self.s2_data_file, = "", None

    def validate_acquisition_wise(self, period):
        dates_between = self.generate_process_periods(period)
        station_folder = "zst" + self.station_name.split("(")[1].split(")")[0]
        wrong = len(station_folder) == 4
        station_folder = "zst" + self.station_name.split(") ")[1].split("(")[1][0:-1] if wrong else station_folder
        station_file = os.path.join(self.dirs["station_counts"], station_folder, station_folder + "_%s.csv" %
                                    str(year))

        for sub_period in dates_between:
            print("At date: %s" % sub_period)
            self.date = sub_period[0]
            band_names, resolution, folder = ["B04", "B03", "B02", "B08", "CLM"], 10, ""
            dir_save_archive = os.path.join(self.dirs["s2"], "archive")

            dir_save_archive = "G:\\archive"

         #   if not os.path.exists(dir_save_archive):
          #      os.mkdir(dir_save_archive)
            area_id = additional_stations[self.station_name_clear] if self.station_name_clear in additional_stations.keys() \
                else self.station_name_clear
            sh_bbox = (self.bbox_epsg4326[1], self.bbox_epsg4326[0], self.bbox_epsg4326[3], self.bbox_epsg4326[2])
            detections_file = os.path.join(self.dirs["detections"], "s2_detections_%s_%s.gpkg" %
                                           (self.date, area_id))
            merged_file = os.path.join(dir_save_archive, "s2_bands_%s_%s_%s_merged.tiff" % (area_id,
                                                                                            sub_period[0],
                                                                                            sub_period[1]))
            if os.path.exists(detections_file):
                self.validate_with_bast(sub_period[0], detections_file, station_file, merged_file)
                continue
            else:

                if os.path.exists(merged_file):
                    detector = RFTruckDetector()
                    band_stack = detector.read_bands(merged_file)
                    detector.preprocess_bands(band_stack[0:4])
                    prediction = detector.predict()
                    prediction_boxes = detector.extract_objects(prediction)
                    try:
                        detector.prediction_boxes_to_gpkg(prediction_boxes, detections_file)
                    except ValueError:
                        print("Number of detections: %s, cannot write" % len(prediction_boxes))
                        continue
                    self.detections_file = detections_file
                    with rio.open(merged_file, "r") as src:
                        meta = src.meta
                    self.lat = lat_from_meta(meta)
                    self.lon = lon_from_meta(meta)
                    detector, band_stack_np = None, None
                    self.validate_with_bast(sub_period[0], detections_file, station_file, merged_file)
                    continue
                else:
                    continue

                kwargs = dict(bbox=sh_bbox, period=sub_period, dataset=DataCollection.SENTINEL2_L2A,
                              bands=["B04", "B03", "B02", "B08"], resolution=resolution, dir_save=self.dirs["s2"],
                              merged_file=merged_file, mosaicking_order="leastCC")
                data_yet_there, sh = os.path.exists(merged_file), SentinelHub()
                obs_file = os.path.join(dir_save_archive, "obs.csv")  # check if acquisition has been checked
                yet_checked = False
                try:
                    obs_pd = pd.read_csv(obs_file, index_col=0)
                except FileNotFoundError:
                    obs_pd = pd.DataFrame()
                    try:
                        obs_pd.to_csv(obs_file)
                    except FileNotFoundError:
                        os.mkdir(os.path.dirname(obs_file))
                        obs_pd.to_csv(obs_file)
                try:
                    yet_checked = sub_period[0] in np.array(obs_pd[merged_file])
                except KeyError:
                    pass
                finally:
                    if yet_checked:
                        continue
                if data_yet_there:
                    data_available = True
                else:
                    sh.set_credentials(SH_CREDENTIALS_FILE)
                    data_available = sh.data_available(kwargs)
                if data_available:
                    if data_yet_there:
                        has_obs = data_yet_there
                    else:
                        # check if data has enough non-cloudy observations
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
                        print("Processing: %s" % sub_period[0])
                        band_stack, folder = sh.get_data(**kwargs)  # get full data
                        detector = RFTruckDetector()
                        band_stack = detector.read_bands(merged_file)
                        detector.preprocess_bands(band_stack[0:4])
                        prediction = detector.predict()
                        prediction_boxes = detector.extract_objects(prediction)
                        try:
                            detector.prediction_boxes_to_gpkg(prediction_boxes, detections_file)
                        except ValueError:
                            print("Number of detections: %s, cannot write" % len(prediction_boxes))
                            continue
                        self.detections_file = detections_file
                        with rio.open(merged_file, "r") as src:
                            meta = src.meta
                        self.lat = lat_from_meta(meta)
                        self.lon = lon_from_meta(meta)
                        detector, band_stack_np = None, None
                        self.validate_with_bast(sub_period[0], detections_file, station_file, merged_file)  # run comparison
                    else:
                        # add date for file in order to avoid duplicate check
                        self.register_non_available_date(sub_period[0], obs_pd, obs_file, merged_file)
                else:
                    self.register_non_available_date(sub_period[0], obs_pd, obs_file, merged_file)
        self.plot_bast_comparison(self.validation_file)

    def validate_with_bast(self, acquisition_date, detections_file, station_file, s2_data_file):
        try:
            validation_pd = pd.read_csv(self.validation_file)
        except FileNotFoundError:
            validation_pd = pd.DataFrame()
        speed = 80
        try:
            self.detections = gpd.read_file(detections_file)
            self.detections = self.detections[self.detections["score"] > 1.2]
        except DriverError:
            self.detections = gpd.GeoDataFrame()
        else:
            self.prepare_s2_counts()
      #  self.detections = self.detections[self.detections.score >= 1.1]
        hour_proportion = (minutes / 60)
        distance_traveled = hour_proportion * speed
        station_counts = pd.read_csv(station_file, sep=";")
        station_point = Point([self.station_meta["x"], self.station_meta["y"]])
        buffer_distance = distance_traveled * 1000
        if "Braunschweig-Flughafen" in detections_file:
            station_buffer = box(station_point.x - buffer_distance, station_point.y - 1000,
                                 station_point.x + buffer_distance, station_point.y + 3000)
        else:
            station_buffer = station_point.buffer(buffer_distance)
        station_buffer_gpd = gpd.GeoDataFrame({"id": [0]}, geometry=[station_buffer],
                                              crs="EPSG:326" + str(self.station_meta["utm_zone"]))
        try:
            detections_in_buffer = gpd.overlay(self.detections.to_crs(station_buffer_gpd.crs),
                                               station_buffer_gpd, "intersection")
        except AttributeError:
            return
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
                           "s2_direction1": s2_direction1, "s2_direction2": s2_direction2,
                           "s2_sum": len(detections_in_buffer)}.items():
            validation_pd.loc[idx, key] = [value]
        for column in station_counts_hour.columns[9:]:  # counts from station
            # add counts proportional to number of minutes
            try:
                validation_pd.loc[idx, column] = [station_counts_hour[column].iloc[0] * hour_proportion]
            except TypeError:
                validation_pd.loc[idx, column] = np.nan
        validation_pd.to_csv(self.validation_file)

    def prepare_s2_counts(self):
        osm_file = get_roads(list(self.bbox_epsg4326), ["motorway", "trunk", "primary"], osm_buffer,
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
        tiles_pd = pd.read_csv(os.path.join(dir_training, "tiles.csv"), sep=",")
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
            validation_boxes = gpd.read_file(glob(os.path.join(dir_labels, "*%s*.gpkg" % tile))[0])
            try:
                prediction_boxes_file = glob(os.path.join(self.dirs["detections"], "*%s*.gpkg" % tile))[1000000]  # fail
            except IndexError:
                lens = np.int32([len(x) for x in imgs])
                img_file = imgs[np.where(lens == lens.max())[0]][0]
                name = os.path.basename(img_file).split(".tif")[0]
                print(img_file)
                # read labels
                prediction_boxes_file = os.path.join(self.dirs["detections"], name + "_boxes.gpkg")
                rf_td = RFTruckDetector()
                band_data = rf_td.read_bands(img_file)
                # subset to label extent
                lat, lon = lat_from_meta(rf_td.meta), lon_from_meta(rf_td.meta)
                extent = validation_boxes.total_bounds  # process only subset where boxes given
                diff_ymin, diff_ymax = np.abs(lat - extent[3]), np.abs(lat - extent[1])
                diff_xmin, diff_xmax = np.abs(lon - extent[0]), np.abs(lon - extent[2])
                ymin, ymax = np.argmin(diff_ymin), np.argmin(diff_ymax)
                xmin, xmax = np.argmin(diff_xmin), np.argmin(diff_xmax)
                rf_td.preprocess_bands(band_data, {"ymin": ymin, "xmin": xmin, "ymax": ymax + 1, "xmax": xmax + 1})
                # do detection
                prediction_array = rf_td.predict()
                prediction_boxes = rf_td.extract_objects(prediction_array)
                rf_td.prediction_raster_to_gtiff(prediction_array,
                                                 os.path.join(self.dirs["detections"], name + "_raster"))
                rf_td.prediction_boxes_to_gpkg(prediction_boxes, prediction_boxes_file)
            prediction_boxes = gpd.read_file(prediction_boxes_file)
            prediction_array, band_data, rf_td = None, None, None
            # iterate over score thresholds and plot precision and recall curve
            for score_threshold in np.arange(0, 2, 0.1):
                prediction_boxes = prediction_boxes[prediction_boxes["score"] >= score_threshold]
                tp = 0
                intersection_over_union = []
                for prediction_box in prediction_boxes.geometry:
                    for validation_box in validation_boxes.geometry:
                        if prediction_box.intersects(validation_box):
                            union = prediction_box.union(validation_box)
                            intersection = prediction_box.intersection(validation_box)
                            iou = intersection.area/union.area
                            if iou > 0.25:
                                intersection_over_union.append(iou)
                                tp += 1
                                break
            #    for validation_box in validation_boxes.geometry:
             #       for prediction_box in prediction_boxes.geometry:
              #          if validation_box.intersects(prediction_box):
               #             union = prediction_box.union(validation_box)
                #            intersection = prediction_box.intersection(validation_box)
                 #           iou = intersection.area/union.area
                  #          if iou > 0.25:
                   #             validation_positive += 1
                    #        break
                try:
                    precision = tp / len(prediction_boxes)
                except ZeroDivisionError:
                    precision = 0
                fn = len(validation_boxes) - tp
                fp = len(prediction_boxes) - tp
                recall = tp / (tp + fn)
                row_idx = len(boxes_validation_pd)
                boxes_validation_pd.loc[row_idx, "detection_file"] = prediction_boxes_file
                boxes_validation_pd.loc[row_idx, "accuracy"] = tp / (tp + fp + fn)
                boxes_validation_pd.loc[row_idx, "precision"] = precision
                boxes_validation_pd.loc[row_idx, "recall"] = recall
                boxes_validation_pd.loc[row_idx, "score_threshold"] = score_threshold
                boxes_validation_pd.loc[row_idx, "n_prediction_boxes"] = len(prediction_boxes)
                boxes_validation_pd.loc[row_idx, "n_validation_boxes"] = len(validation_boxes)
                boxes_validation_pd.loc[row_idx, "IoU"] = np.mean(intersection_over_union)
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
        """
        :param vector: array-like y, x
        :return:
        """
        # [1,1] -> 45°; [-1,1] -> 135°; [-1,-1] -> 225°; [1,-1] -> 315°
        direction = np.degrees(np.arctan2(vector[1], vector[0])) % 360
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
        min_valid_percentage = 90
        rf_td = RFTruckDetector()
        try:
            clm = rf_td.read_bands(cloud_mask_file)
        except rio.errors.RasterioIOError:
            return False
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
        return valid_percentage > min_valid_percentage

    @staticmethod
    def register_non_available_date(the_date, table, table_file, raster_file):
        try:
            table.loc[len(table[raster_file].dropna()), raster_file] = the_date
        except KeyError:
            table[raster_file] = [the_date]
        table.to_csv(table_file)

    @staticmethod
    def plot_bast_comparison(validation_file):
        s2_columns = ["s2_direction1", "s2_direction2"]
        s2_labels = ["S2 direction 1", "S2 direction 2"]
        lzg_columns, lzg_labels = [NAME_TR1, NAME_TR2], ["BAST Lzg direction 1", "BAST Lzg direction 2"]
        all_columns = np.hstack([s2_columns, lzg_columns, "KFZ_R1", "KFZ_R2"])
        all_labels = np.hstack([s2_labels, lzg_labels, "BAST Kfz direction 1", "BAST Kfz direction 2"])
        all_labels = ["Sentinel-2", "BAST Lzg", "BAST Kfz"]
        s2_lzg_columns = np.hstack([s2_columns, lzg_columns])
        s2_lzg_labels = np.hstack([s2_labels, lzg_labels])
        s2_lzg_labels = all_labels[:-1]
        for columns, labels, suffix in zip([s2_lzg_columns, all_columns], [s2_lzg_labels, all_labels],
                                           ["Lzg", "Lzg_Kfz"]):
          #  sns.set(rc={"figure.figsize": (9, 5)})
           # sns.set_theme(style="whitegrid")
            fix, ax = plt.subplots(figsize=(7, 3))
            validation = pd.read_csv(validation_file)
            unique_dates = np.unique(validation["date"])
            colors = [S2_COLOR, "#f542cb", BAST_COLOR, "#abc70a"]
            if len(columns) > 4:
                colors.append("#185a61")  # for kfzs
                colors.append("#4dd0de")
            counts = np.zeros((len(unique_dates), len(columns)), dtype=np.int16)
            for i, acquisition_date in enumerate(unique_dates):
                idx = np.where(validation["date"] == acquisition_date)[0][0]
                row = validation.iloc[idx]
                counts[i] = np.int16([np.int16(row[column]) for column in columns])
         #   for i, c in enumerate(colors):
          #      ax.plot(unique_dates, counts[:, i], color=c)
            ax.plot(unique_dates, np.sum(counts[:, 0:2], 1), color=S2_COLOR, linewidth=2)
            ax.plot(unique_dates, np.sum(counts[:, 2:4], 1), color=BAST_COLOR, linewidth=2)
            if "KFZ_R1" in columns:
                ax.plot(unique_dates, np.sum(counts[:, 4:], 1), color="#185a61", linewidth=2)
            plt.legend(labels, bbox_to_anchor=(1.25, 0.5), fontsize=10, loc="center right")
            plt.subplots_adjust(bottom=0.2, right=0.9)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            ax.xaxis.set_tick_params(labelsize=10)
            ax.yaxis.set_tick_params(labelsize=10)
            title = "Sentinel-2 & BAST counts " + validation_file.split("_")[-1].split(".csv")[0]
      #      plt.title(title, fontsize=12)
            plt.tight_layout()
            fname = title.replace(" ", "_")
            plt.savefig(os.path.join(dir_validation_plots, "%s_%s_lineplot.png" % (fname, suffix)), dpi=500)
            plt.close()
            # scatter
            fig, ax = plt.subplots(figsize=(6, 4))
            s2, bast = counts[:, 0] + counts[:, 1], counts[:, counts.shape[1] - 2] + counts[:, counts.shape[1] - 1]
            c, position = S2_COLOR, [10, 10]
            ax.scatter(x=s2, y=bast, color=c)
            regress = linregress(x=s2, y=bast)
            m, b = np.polyfit(s2, bast, 1)
            ax.plot(s2, m * s2 + b, color="#e5df1b", alpha=0.8)
            plt.text(np.nanquantile(s2, [0.99])[0] + 15, np.nanquantile(bast, [0.85])[0],
                     "Lin. regression\npearson r-value=%s\nslope=%s" % (np.round(regress.rvalue, 2),
                                                                 np.round(regress.slope, 2)), fontsize=11)
            plt.subplots_adjust(right=0.9)
            label = labels[-1][:8]
            plt.ylabel(label)
            plt.xlabel("Sentinel-2 count")
#            plt.title("Sentinel-2 trucks vs. %s" % label)
            plt.tight_layout()
            plt.savefig(os.path.join(dir_validation_plots, "%s_%s_scatterplot.png" % (fname, label.replace(" ", ""))),
                        dpi=500)
            plt.close()

    @staticmethod
    def plot_box_validation_recall_precision(box_validation_csv):
        c = "#330432"
        boxes_validation = pd.read_csv(box_validation_csv)
        unique_files = np.unique(boxes_validation["detection_file"])
        fig, axes = plt.subplots(2, int(len(unique_files) * 0.5), figsize=(10, 4))
        countries = {"T34UDC": "Poland", "T29SND": "Portugal", "T31UFS": "Belgium", "T33TUN": "Austria",
                     "T35TMK": "Romania", "T37MCT": "Kenya", "T52SDE": "South Korea", "T12SUC": "USA",
                     "T60HUD": "New Zealand", "T21HUB": "Argentina"}
        labels, aucs, f1_scores, accuracies, scores = [], [], [], [], None
        max_recalls, max_precisions = [], []
        for idx, file, ax in zip(range(len(unique_files)), unique_files, axes.flatten()):
            tile = file.split("_")[-3]
            labels.append(tile + " (%s)" % countries[tile])
            boxes_validation_subset = boxes_validation[boxes_validation["detection_file"] == file]
            scores = np.float32(boxes_validation_subset["score_threshold"])
            precision = np.float32(boxes_validation_subset["precision"])
            recall = np.float32(boxes_validation_subset["recall"])
            accuracies.append(np.float32(boxes_validation_subset["accuracy"]))
            f_score = 2 * ((precision * recall) / (precision + recall))
            f_score[np.isnan(f_score)] = 0
            ax.plot(recall, precision, color=c, linewidth=3)
            f1_scores.append(f_score)
            ax.set_title(tile + " (%s)" % countries[tile])
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, 1)
            ax.set_ylabel("Precision")
            ax.set_xlabel("Recall")
            max_recalls.append(np.max(recall))
            max_precisions.append(np.max(precision))
        plt.tight_layout()
        plt.savefig(os.path.join(dir_validation_plots, "recall_precision_box_validation_lineplot.png"), dpi=600)
        plt.close()
        # plot f-score and accuracy
        f1_scores_np = np.float32(f1_scores)
        max_f1_scores = np.max(f1_scores_np, 1)
        mean_f_score = np.mean(max_f1_scores)
        accuracies_np = np.float32(accuracies)
        max_accuracies = np.max(accuracies, 1)
        for ylabel, values, max_scores in zip(["Accuracy", "F1-score"], [accuracies, f1_scores],
                                             [max_accuracies, max_f1_scores]):
            fig, axes = plt.subplots(2, int(len(unique_files) * 0.5), figsize=(10, 4))
            for idx, ax, these_values, max_score, label in zip(range(len(values)), axes.flatten(), values, max_scores,
                                                               labels):
                ax.plot(scores, values[idx], color=c, linewidth=3)
                ax.set_title(label)
                ax.set_ylim(0, 1)
                ax.set_xlim(0, np.max(scores))
                ax.set_ylabel(ylabel)
                ax.set_xlabel("Detection score")
                z = 0.01 if ax == axes.flatten()[6] else 0.03
                ax.text(scores[np.where(these_values == max_score)[0][0]] + 0.05, max_score + z, np.round(max_score, 2))
            plt.tight_layout()
            plt.savefig(os.path.join(dir_validation_plots, "%s_vs_score_validation_lineplot.png" % ylabel), dpi=600)
            plt.close()
        #  p = plt.plot(scores, np.nanmean(np.float32(f_scores), 0), linewidth=2.5, linestyle="--", color="black")
        #plt.ylim(0, 1)
        #plt.xlim(0, np.max(scores))
        #plt.ylabel("F-score")
        #plt.xlabel("Detection score")
        #plt.subplots_adjust(right=0.7)
        #labels.append("Mean")
        #plt.legend(labels, loc="center right", bbox_to_anchor=(1.47, 0.5))
    
    @staticmethod
    def plot_bast_summary():
        s2_dir1, s2_dir2 = "s2_direction1", "s2_direction2"
        files = list(set(glob(os.path.join(dir_validation, "*.csv"))) - set(glob(os.path.join(dir_validation,
                                                                                              "*validation.csv"))))
        s2_values, bast_values = [], []
        for file in files:
            validation_pd = pd.read_csv(file)
            start_date = validation_pd.date.iloc[0]
            start_idx = np.where(validation_pd.date == start_date)[0][-1]
            validation_pd = validation_pd[start_idx:]
            s2_values.append(np.float32(validation_pd[s2_dir1] + validation_pd[s2_dir2]))
            bast_values.append(np.float32(validation_pd[NAME_TR1] + validation_pd[NAME_TR2]))
        s2_values_flat, bast_values_flat = np.hstack(s2_values), np.hstack(bast_values)
        regress = linregress(s2_values_flat, bast_values_flat)
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.scatter(s2_values_flat, bast_values_flat, color=S2_COLOR)
        ax.set_ylim(-10, 200)
        ax.set_xlim(-10, 200)
        ax.set_ylabel("Traffic count station")
        ax.set_xlabel("Sentinel-2")
        rmse = np.sqrt(np.square(np.subtract(bast_values_flat, s2_values_flat)).mean())
        m, b = np.polyfit(s2_values_flat, bast_values_flat, 1)
        formula = "y = %sx + %s" % (np.round(m, 2), np.round(b, 2))
        ax.plot(s2_values_flat, m * s2_values_flat + b, color="#e5df1b", alpha=0.8)
        summary = "n=%s\nLin. regression\npearson r-value: %s\nslope: %s\nFormula: %s\np-value: %s\n\nRMSE: %s" % (
            len(s2_values_flat), np.round(regress.rvalue, 2), np.round(regress.slope, 2), formula, np.round(regress.pvalue, 2), rmse)
        plt.text(210, 40, summary, fontsize=11)
        plt.text(-100, 40, " ")
        plt.tight_layout()
        plt.savefig(os.path.join(dir_validation_plots, "bast_overall_scatterplot.png"), dpi=500)
        plt.close()
        rmse = np.sqrt(np.square(np.subtract(bast_values_flat, s2_values_flat)).mean())
        # by weekday
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        n_weekday = np.zeros(7)
        r_values = np.zeros_like(n_weekday)
        mean_by_weekday = {"s2": [], "bast": [], "bast_car": []}
        for weekday in np.arange(7):
            s2_values, bast_values, bast_car_values = [], [], []
            for file in files:
                validation_pd = pd.read_csv(file)
                start_date = validation_pd.date.iloc[0]
                start_idx = np.where(validation_pd.date == start_date)[0][-1]
                validation_pd = validation_pd[start_idx:]
                for idx, acquistion_date in enumerate(validation_pd.date):
                    d = date.fromisoformat(acquistion_date)
                    if d.weekday() == weekday:
                        row = validation_pd.iloc[idx]
                        s2_values.append(np.float32(row[s2_dir1] + row[s2_dir2]))
                        bast_values.append(np.float32(row[NAME_TR1] + row[NAME_TR2]))
                        bast_car_values.append(np.float32(row["KFZ_R1"] + row["KFZ_R2"]))
            n_weekday[weekday] = len(s2_values)
            s2_values_flat, bast_values_flat = np.hstack(s2_values), np.hstack(bast_values)
            bast_car_values_flat = np.hstack(bast_car_values)
            mean_by_weekday["s2"].append(np.mean(s2_values_flat))
            mean_by_weekday["bast"].append(np.mean(bast_values_flat))
            mean_by_weekday["bast_car"].append(np.mean(bast_car_values_flat))
         #   ax.scatter(s2_values_flat, bast_values_flat, color=S2_COLOR, s=0.01)
            regress = linregress(s2_values_flat, bast_values_flat)
            r_values[weekday] = regress.rvalue
          #  ax.scatter(s2_values_flat, bast_values_flat, color=S2_COLOR)
         #   m, b = np.polyfit(s2_values_flat, bast_values_flat, 1)
          #  formula = "y = %sx + %s" % (np.round(m, 2), np.round(b, 2))
          #  ax.plot(s2_values_flat, m * s2_values_flat + b, color=S2_COLOR, alpha=0.8)
          #  plt.text(-5, 220, "Lin. regression\npearson r-value: %s\nslope: %s\n%s" %
           #          (np.round(regress.rvalue, 2), np.round(regress.slope, 2), formula), fontsize=11)
        s2, bast = np.float32(mean_by_weekday["s2"]), np.float32(mean_by_weekday["bast"])
        fig, ax = plt.subplots(figsize=(6, 3))
        #ax = axes[0]
        ax.plot(weekdays, s2, color=S2_COLOR, linewidth=2.5)
        bast_std = str(np.round(np.std(bast), 2))
        s2_std = str(np.round(np.std(s2), 2))
       # stat_str = "BAST\nstandard deviation: %s" % bast_std
      #  ax.text(4.95, bast[5] + 5, stat_str)
        ax.plot(weekdays, bast, color=BAST_COLOR, linewidth=2.5)
        ax.legend(["Sentinel-2", "BAST Lzg"], loc="upper right")
     #   stat_str = stat_str.replace("BAST", "Sentinel-2").replace(bast_std, s2_std)
     #   stat_str = stat_str + "\nMean share of BAST: %s %s" % (np.round(np.mean(np.float32(s2 / bast) * 100), 2), "%")
     #   ax.text(3, s2[3] - 15, stat_str)
        #ax = axes[0]
        regress = linregress(s2, bast)
    #    ax.scatter(s2, bast, color="black")
     #   ax.set_xlabel("Sentinel-2")
      #  ax.set_ylabel("BAST Lzg")
        ax.text(0, 3, "pearson r-value: %s" % (np.round(regress.rvalue, 2)), fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(dir_validation_plots, "mean_values_by_weekday_lineplot_scatterplot.png"), dpi=500)
        plt.close()


if __name__ == "__main__":
    if validate == "bast":
        try:
            os.remove(os.path.join(dir_validation, "validation_run.csv"))
        except FileNotFoundError:
            pass
    else:
        try:
            os.remove(boxes_validation_file)
            print("")
        except FileNotFoundError:
            pass
    for station, acquisition_period in stations.items():
        try:
            validator = Validator(station, aois_file, dir_validation, dir_osm)
        except KeyError:
            continue
        if validate == "boxes":
            validator.validate_boxes()
            validator.plot_box_validation_recall_precision(boxes_validation_file)
            break  # messy..
        else:
            print("%s\nStation: %s" % ("-" * 50, station))
          #  validator.validate_acquisition_wise(acquisition_period)
            validator.plot_bast_summary()
