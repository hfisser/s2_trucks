import os, utm, requests
import numpy as np
import rasterio as rio
import geopandas as gpd
import pandas as pd
from glob import glob
from shutil import copyfile
from shapely.geometry import Point
from shapely.geometry.linestring import LineString

from osm_utils.utils import get_roads, rasterize_osm
from array_utils.points import raster_to_points
from array_utils.geocoding import lat_from_meta, lon_from_meta
from SentinelHubDataAccess.SentinelHub import SentinelHub, DataCollection
from detect_trucks.TruckDetector import Detector

dir_validation = "F:\\Masterarbeit\\DLR\\project\\1_truck_detection\\validation"
dir_osm = os.path.join(os.path.dirname(dir_validation), "data", "osm")
aois_file = os.path.join(dir_validation, "data", "BAST", "validation_aois.gpkg")


SH_CREDENTIALS_FILE = os.path.join("F:" + os.sep + "sh", "sh.txt")
BAST_URL = "https://www.bast.de/BASt_2017/DE/Verkehrstechnik/Fachthemen/v2-verkehrszaehlung/Aktuell/" \
           "zaehl_aktuell_node.html;jsessionid=63610843F87B77C24C4320BC4EAD6647.live21304"
NAME_DATE = "Datum"
NAME_HOUR = "Stunde"
NAME_TR1 = "Lkw_R1"
NAME_TR2 = "Lkw_R2"

OSM_BUFFER = 30
hour, minutes, year = 10, 10, 2018

stations_a2 = ["Immensen (3489)", "Theeßen (3810)", "Alleringersleben (3837)", "Peine (3306)"]
stations = {"Theeßen (3810)": ["2018-11-28", "2018-11-28"],
            "Schuby (1189)": ["2018-05-05", "2018-05-05"],
            "Salzbergen (3499)": ["2018-06-07", "2018-06-07"],
            "Nieder Seifersdorf (4123)": ["2018-10-31", "2018-10-31"],
            "AS Dierdorf VQ Nord (7781)": ["2018-05-08", "2018-05-08"],
            "Offenburg (8054)": ["2018-09-27", "2018-09-27"],
            "Reussenberg (8168)": ["2018-04-27", "2018-04-27"],
            "Lenting (S) (9090)": ["2018-04-07", "2018-04-07"],
            "Röstebachtalbrücke (4372)": ["2018-04-10", "2018-04-10"],
            "Sprakensehl (4702)": ["2018-04-20", "2018-04-20"],  # Bundesstraße
            "Crailsheim-Süd (8827)": ["2018-04-27", "2018-04-27"]}  # Bundesstraße


class Validator:
    def __init__(self, station_name, station_aois_file, dir_validation_home, dir_osm_data):
        aois = gpd.read_file(station_aois_file)
        self.dirs = {"validation": dir_validation_home, "osm": dir_osm_data,
                     "station_counts": os.path.join(dir_validation, "data", "BAST", "station_counts"),
                     "s2": os.path.join(dir_validation_home, "data", "s2"),
                     "detections": os.path.join(dir_validation, "detections")}
        for directory in self.dirs.values():
            if not os.path.exists(directory):
                os.mkdir(directory)
        self.validation_file = os.path.join(self.dirs["validation"], "validation_run.csv")
        self.station_name = station_name
        self.station_meta = self.get_station_meta(station_name)
        bbox = aois[aois["Name"] == station_name].geometry.bounds
        self.bbox_wgs84 = (bbox.miny[0], bbox.minx[0], bbox.maxy[0], bbox.maxx[0])  # min lat, min lon, max lat, max lon
        self.crs = aois.crs
        self.lat, self.lon = None, None
        self.detections, self.osm_roads = None, None
        self.date = None
        self.detections_file, self.s2_data_file = None, None

    def detect(self, period):
        self.date = period[0]
        band_names = ["B04", "B03", "B02", "CLM"]
        resolution = 10
        dir_save_archive = os.path.join(self.dirs["s2"], "archive")
        if not os.path.exists(dir_save_archive):
            os.mkdir(dir_save_archive)
        self.s2_data_file = os.path.join(dir_save_archive, "s2_bands_%s_%s_%s.tiff" % (self.station_name, period[0],
                                                                                       period[1]))
        if not os.path.exists(self.s2_data_file):
            sh = SentinelHub()
            sh.set_credentials(SH_CREDENTIALS_FILE)
            band_stack, dir_data = sh.get_data(self.bbox_wgs84, period, DataCollection.SENTINEL2_L2A, band_names,
                                               resolution, self.dirs["s2"])
            files = glob(dir_data + os.sep + "*.tiff")
            if len(files) > 1:
                print("Several files, don't know which to read from %s" % self.dirs["s2"])
                raise FileNotFoundError
            else:
                reflectance_file = copyfile(files[0], self.s2_data_file)
        # read through rio in order to easily have metadata
        try:
            with rio.open(self.s2_data_file) as src:
                meta = src.meta
                band_stack_np = np.zeros((meta["width"], meta["height"], meta["n_bands"]))
                for b in range(band_stack_np.shape[0]):
                    band_stack_np[b] = src.read(b + 1)
        except rio.errors.RasterioIOError as e:
            raise e
        # decrease size of reference lat, lon raster in order to calculate road distances on it
        meta["width"], meta["height"] = int(meta["width"] / 50), int(meta["height"] / 50)
        self.lat, self.lon = lat_from_meta(meta), lon_from_meta(meta)
        band_stack_np = band_stack_np.swapaxes(0, 2)  # z, y, x
        detector = Detector()
        band_stack_np = detector.pre_process({"B08": band_stack_np[3], "B04": band_stack_np[0],
                                              "B03": band_stack_np[1], "B02": band_stack_np[2]}, meta, None)
        # remove original band stack file that has been moved to archive yet
        os.remove(files[0])
        self.detections = detector.detect_trucks(band_stack_np)
        self.detections_file = os.path.join(self.dirs["detections"], "s2_detections_%s_%s.gpkg" %
                                            (self.date, self.station_name))
        self.detections.to_file(self.detections_file)

    def validate(self):
        try:
            validation_pd = pd.read_csv(self.validation_file)
        except FileNotFoundError:
            validation_pd = pd.DataFrame()
        hour_proportion = (minutes / 60)
        station_folder = "zst" + self.station_name.split("(")[1].split(")")[0]
        station_file = station_folder + "_%s.csv" % str(year)
        station_counts = pd.read_csv(os.path.join(self.dirs["station_counts"], station_folder, station_file), sep=";")
        x, y = self.station_meta["x"], self.station_meta["y"]
        station_point = Point(np.flip(utm.to_latlon(x, y, self.station_meta["utm_zone"], "N")))
        osm_raster = rasterize_osm(self.osm_roads, np.zeros((len(self.lat), len(self.lon))))
        osm_raster[~np.isnan(osm_raster) * osm_raster != 0] = 1
        # create point grid in aoi
        road_points = raster_to_points(osm_raster, {"lat": self.lat, "lon": self.lon}, "id", self.crs)
        # subset points to osm road polygons in order to construct line from detection to station
        road_points_clipped = gpd.sjoin(road_points,
                                        gpd.GeoDataFrame({"id": [0]}, geometry=[self.osm_roads.unary_union]),
                                        "left", "within")
        detections_in_reach = []
        for detection in self.detections.iterrows():
            detection = detection[1]
            detection_point = detection.geometry.centroid
            points_between_detection_and_station = []
            detection_y, detection_x = detection_point.y, detection_point.x
            station_y, station_x = station_point.y, station_point.x
            sorted_ys = np.sort([detection_y, station_y])
            sorted_xs = np.sort([detection_x, station_x])
            for road_point in road_points_clipped.geometry:
                if sorted_ys[0] < road_point.y > sorted_ys[1] and sorted_xs[0] < road_point.x > sorted_xs[1]:
                    points_between_detection_and_station.append(road_point)
            line_to_detection = LineString(points_between_detection_and_station)  # create line from detection to station
            traveled_distance = detection["speed"] * hour_proportion
            distance_matching = traveled_distance >= line_to_detection.length  # passed by the station in number of minutes
            # check direct line to detection and compare with vehicle heading (only include if heading away)
            direct_vector_to_detection = self.calc_vector([station_y, station_x], [detection_y, detection_x])
            # calculate in which direction the station is
            direction_bins = np.arange(0, 359, 22.5, dtype=np.float32)
            offset = int(len(direction_bins) / 4)
            station_direction = self.calc_vector_direction_in_degree(direct_vector_to_detection)
            diffs = np.abs(direction_bins - station_direction)
            lowest_diff_idx = np.where(diffs == diffs.min())[0][0]
            # get range of directions (180°)
            direction_range = np.sort([direction_bins[lowest_diff_idx - 4], direction_bins[lowest_diff_idx + 4]])
            # check if vehicle is traveling from station (count it) or to station (drop it)
            direction_matching = direction_range[0] < detection["direction_degree"] < direction_range[1]
            if distance_matching and direction_matching:
                detections_in_reach.append(detection)
        # compare number of detections in reach to the one of count station
        date_station_format = self.date[2:].replace("-", "")  # e.g. "2018-12-31" -> "181231"
        time_match = (station_counts["Datum"] == int(date_station_format)) * (station_counts["Stunde"] == hour)
        station_counts_hour = station_counts[time_match]
        idx = len(validation_pd)
        for key, value in {"station_file": station_file, "s2_counts_file": self.s2_data_file,
                           "detections_file": self.detections_file,
                           "hour": hour, "n_minutes": minutes, "s2_counts": len(detections_in_reach)}.items():
            validation_pd.loc[idx, key] = [value]
        for column in station_counts_hour.columns[9:]:  # counts from station
            # add counts proportional to number of minutes
            validation_pd.loc[idx, column] = [station_counts_hour[column].iloc[0] * hour_proportion]
        validation_pd.to_csv(self.validation_file)

    def prepare_s2_counts(self):
        osm_file = get_roads(list(self.bbox_wgs84), ["motorway", "trunk", "primary"], OSM_BUFFER,
                             self.dirs["osm"], str(self.bbox_wgs84).replace(", ", "_")[1:-1].replace(".", "_")
                             + "_osm_roads", str(self.crs))
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
            if row.intersects(osm_union):
                detections_within.append(row)
        self.detections = pd.concat(detections_within)

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


if __name__ == "__main__":
    for station, acquisition_period in stations.items():
        print("Validating at station: %s" % station)
        validator = Validator(station, aois_file, dir_validation, dir_osm)
        validator.detect(acquisition_period)
        validator.validate()
