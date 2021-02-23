import os
import pickle
import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
import rasterio as rio
import matplotlib.pyplot as plt
from shapely.geometry import Point
from validate_trucks.TruckValidator import Validator
from glob import glob
from scipy.stats import linregress
from shapely.geometry import box
from SentinelHubDataAccess.SentinelHub import SentinelHub, DataCollection
from detect_trucks.RandomForestTrucks import RFTruckDetector

SH_CREDENTIALS_FILE = os.path.join("F:" + os.sep + "sh", "sh.txt")

resolution = 10

bast_station = "Braunschweig-Flughafen (3429)"

dir_main = "F:\\Masterarbeit\\DLR\\project\\1_truck_detection"
dir_comparison = os.path.join(dir_main, "comparison")
dir_comparison_detections = os.path.join(dir_comparison, "detections")
dir_comparison_detections_boxes = os.path.join(dir_comparison_detections, "boxes")
dir_comparison_detections_rasters = os.path.join(dir_comparison_detections, "rasters")
dir_comparison_plots = os.path.join(dir_comparison, "plots")
dir_validation = os.path.join(dir_main, "validation")
dir_validation_data = os.path.join(dir_validation, "data", "s2", "archive")
dir_comparison_s5p = os.path.join(dir_comparison, "OUT_S5P")
dir_comparison_insitu = os.path.join(dir_comparison, "OUT_Insitu")
dir_comparison_wind = os.path.join(dir_comparison, "OUT_Wind")
dir_validation = os.path.join(dir_main, "validation")
dir_osm = os.path.join(dir_main, "code", "detect_trucks", "AUXILIARY", "osm")
aoi_file = os.path.join(dir_comparison, "aoi_h_bs.geojson")

for directory in [dir_comparison_detections, dir_comparison_detections_boxes, dir_comparison_detections_rasters,
                  dir_comparison_plots]:
    if not os.path.exists(directory):
        os.mkdir(directory)

aoi_file = os.path.join(dir_comparison, "aoi_h_bs.geojson")
uba_stations_locations_file = os.path.join(dir_comparison_insitu, "station_locations.csv")
uba_dates = "20180410,20180420,20180507,20180520,20180522,20180606,20180611,20180724,20180726,20180803,20180823," \
            "20180919,20181012,20181014".split(",")


process_dates = ["10-04-2018",
                 "20-04-2018",
                 "07-05-2018",
                 "20-05-2018",
                 "22-05-2018",
                 "06-06-2018",
                 "11-06-2018",
                 "24-07-2018",
                 "03-08-2018",
                 "23-08-2018",
                 "19-09-2018",
                 "12-10-2018"]
comparison_variables = ["var_VCDtropo", "var_mod_NO2_AK_coulumn"]
lon_crop = 10.6695
uba_station_buffer = 10000  # meters


class Comparison:
    def __init__(self, dates, aoi_file):
        self.dates = dates
        self.bbox = gpd.read_file(aoi_file).to_crs("EPSG:4326").geometry.bounds

    def run_comparison(self):
        for date in self.dates:
            print("Processing: %s" % date)
            sh = SentinelHub()
            sh.set_credentials(SH_CREDENTIALS_FILE)
            sh_bbox = tuple(list(self.bbox.iloc[0]))  # xmin, ymin, xmax, ymax
            file_str = "_".join([str(coord) for coord in sh_bbox]) + "_" + date.replace("-", "_")
            merged_file = os.path.join(dir_validation_data, "s2_bands_%s.tiff" % file_str)
            if os.path.exists(merged_file):
                pass
            else:
                band_stack, folder = sh.get_data(sh_bbox, [date, date], DataCollection.SENTINEL2_L2A,
                                                 ["B04", "B03", "B02", "B08", "CLM"], resolution, dir_validation_data,
                                                 merged_file)
                band_stack = None
            detections_file = os.path.join(dir_comparison_detections_boxes, "%s_detections.gpkg" % file_str)
            if os.path.exists(detections_file):
                pass
            else:
                rf_td = RFTruckDetector()
                band_stack = rf_td.read_bands(merged_file)
                rf_td.preprocess_bands(band_stack[0:4])
                rf_td.train()
                prediction_array = rf_td.predict()
                prediction_boxes = rf_td.extract_objects(prediction_array)
                rf_td.prediction_boxes_to_gpkg(prediction_boxes, detections_file)
            for comparison_variable in comparison_variables:
                file_prefix = {"var_mod_NO2_AK_coulumn": "test_tropomi_NO2_",
                               "var_VCDtropo": "test_tropomi_NO2_"}[comparison_variable]
                dir_target = [dir_comparison_s5p, dir_comparison_s5p][int("NO2" in file_prefix)]
                split = file_str.split("_")
                d, m, y = split[-3], split[-2], split[-1]
                try:
                    comparison_raster_file = glob(os.path.join(dir_target, file_prefix + "%s%s%s*.nc" % (y, m, d)))[0]
                except IndexError:
                    continue
                self.compare_s5p_no2(detections_file, comparison_raster_file, comparison_variable, "-".join([y, m, d]))
        self.compare_insitu_no2(glob(os.path.join(dir_comparison_detections_boxes, "*.gpkg")))

    def plot_s2_series(self):
        weekdays = {"2018-05-22": "Tuesday", "2018-06-06": "Wednesday", "2018-06-11": "Monday",
                    "2018-07-24": "Tuesday", "2018-08-03": "Friday", "2018-08-23": "Thursday",
                    "2018-09-19": "Wednesday", "2018-10-12": "Friday", "2018-04-10": "Tuesday",
                    "2018-04-20": "Friday", "2018-05-07": "Monday", "2018-05-20": "Sunday"}
        detection_files = glob(os.path.join(dir_comparison_detections_boxes, "*.gpkg"))
        dates, n_detections = [], []
        for detection_file in detection_files:
            str_split = detection_file.split("_")
            date = "-".join([str_split[-2], str_split[-3], str_split[-4]])
            dates.append(date)
            n_detections.append(len(gpd.read_file(detection_file)))
        date_sort = np.argsort(dates)
        dates, n_detections = np.array(dates)[date_sort], np.int16(n_detections)[date_sort]
        dates = [date + " (%s)" % weekdays[date] for date in dates]
        plt.close()
        plt.plot_date(dates, n_detections, xdate=True, color="#7b0c7c", alpha=0.8)
        plt.ylabel("Detected trucks")
        plt.title("Number of detected trucks Sentinel-2")
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.4)
        plt.axes().xaxis.set_tick_params(labelsize=8)
        plt.axes().yaxis.set_tick_params(labelsize=8)
        plt.savefig(os.path.join(dir_comparison_plots, "s2_detections_series.png"))
        plt.close()
        self.compare_station_counts(np.array(detection_files)[date_sort], dates)  # call here because we have the files and dates

    def compare_s5p_no2(self, detections_file, raster_file, raster_variable_name, date_of_interest):
        """
        reads detections and raster tiles, number of detections in each cell is compared with n detections
        :param detections_file: str file path to detections
        :param raster_file: str file path of raster
        :param raster_variable_name: str name of the variable in the netcdf
        :param date_of_interest: str date
        :return:
        """
        # load Sentinel-5P tiles and rasterize detections on that grid
        detections = gpd.read_file(detections_file)
        detections_basename = os.path.basename(detections_file)
        detections = detections.to_crs("EPSG:4326")  # because rasters are given as 4326
        reference_array = xr.open_dataset(raster_file)
        lat, lon = reference_array.lat.values[::-1], reference_array.lon.values
        diff_lon_crop_lon = np.abs(lon - lon_crop)
        crop_idx = np.where(diff_lon_crop_lon == np.min(diff_lon_crop_lon))[0][0] + 1
        lon = lon[crop_idx:]  # crop lon
        lat_resolution = np.mean(lat[1:] - lat[:-1])
        lon_resolution = np.mean(lon[1:] - lon[:-1])
        box_str = "_".join([str(np.min(coord)) + "_" + str(np.max(coord)) for coord in [lat, lon]])
        comparison_array = reference_array[raster_variable_name].values[:, crop_idx:]
        s2_trucks_array = np.zeros_like(comparison_array)
        # iterate over cells and count number of s2 trucks
        for y in range(s2_trucks_array.shape[0]):
            for x in range(s2_trucks_array.shape[1]):
                ymin, xmin = lat[y], lon[x]
                try:
                    ymax = lat[y + 1]
                except IndexError:
                    ymax = lat[y] + lat_resolution
                try:
                    xmax = lon[x + 1]
                except IndexError:
                    xmax = lon[x] + lon_resolution
                cell_box_gpd = gpd.GeoDataFrame({"id": [0]}, geometry=[box(xmin, ymin, xmax, ymax)],
                                                crs=detections.crs)  # raster cell as box, count boxes within
                s2_trucks_array[y, x] = len(gpd.clip(detections, cell_box_gpd))  # number of detections in cell
        detections_raster_file = os.path.join(dir_comparison_detections_rasters, detections_basename + box_str +
                                              "reference%s.tiff" % raster_variable_name)
        # trucks raster to gtiff
        meta = dict(dtype=np.float32, count=1, crs=detections.crs, height=s2_trucks_array.shape[0],
                    width=s2_trucks_array.shape[1], driver="GTiff", nodata=None)
        meta["transform"] = rio.transform.from_bounds(np.min(lon), np.min(lat), np.max(lon), np.max(lat), len(lon),
                                                      len(lat))
        with rio.open(detections_raster_file, "w", **meta) as tgt:
            tgt.write(s2_trucks_array, 1)
        # plot detections vs. comparison array
        comparison_array[s2_trucks_array == 0] = np.nan
        s2_trucks_array[s2_trucks_array == 0] = np.nan
        wind = xr.open_dataset(
            os.path.join(dir_comparison_wind, "Wind_U_V_%s.nc" % date_of_interest.replace("-", "")))
        wind_direction = self.calc_wind_direction(wind)
        for wind_direction in [1, 2, 3]:
            wind_name = "wind%s.pickle" % wind_direction  # list pickle with values of this wind direction
            comparison_wind_file = os.path.join(dir_comparison_s5p, wind_name)
            s2_wind_file = os.path.join(dir_comparison_detections_boxes, wind_name)
            comparison_wind = pickle.load(open(comparison_wind_file, "rb"))
            s2_wind = pickle.load(open(s2_wind_file, "rb"))
            # filter by wind direction
            ys, xs = np.where(comparison_array == wind_direction)
            for y, x in zip(ys, xs):
                comparison_wind.append(comparison_array[y, x])
                s2_wind.append(s2_trucks_array[y, x])
            pickle.dump(comparison_wind, open(comparison_wind_file, "wb"))
            pickle.dump(s2_wind, open(s2_wind_file, "wb"))

#            plt.scatter(comparison_array_flat, s2_array_flat)  # xy of flat arrays
#           folder = os.path.dirname(raster_file).split(os.sep)[-1]
#          comparison_name = {"OUT_S5P": "S5P NO2 total column", "OUT_Insitu": "Insitu"}[folder]
#         plt.ylabel("S2 trucks")
#        plt.xlabel(comparison_name)
#       plt.title("S2 trucks vs. %s on %s" % (comparison_name, date_of_interest))
#      plt.savefig(os.path.join(dir_comparison_plots, raster_variable_name + "_vs_%s.png" %
#                              detections_basename.split(".gpkg")[0]))
#    plt.close()

    @staticmethod
    def compare_insitu_no2(detection_files):
        crs = gpd.read_file(detection_files[0]).crs
        uba_station_locations_pd = pd.read_csv(uba_stations_locations_file, sep=";", index_col=0)
        values = np.zeros((len(uba_station_locations_pd), 2, len(detection_files)))
        for row_idx in range(len(uba_station_locations_pd)):
            row = uba_station_locations_pd.iloc[row_idx]
            station_name = row.name
            station_point_gpd = gpd.GeoDataFrame({"id": [0], "geometry": [Point([row.lat, row.lon])]}, crs="EPSG:4326")
            station_point_gpd_utm = station_point_gpd.to_crs(crs)
            station_buffer = station_point_gpd_utm.buffer(uba_station_buffer)
            station_file = os.path.join(dir_comparison_insitu, "_".join([station_name, "NO2", "year", "2018", ".nc"]))
            station_data = xr.open_dataset(station_file)
            station_obs = station_data.obs.values
            dates = []
            for idx, detection_file in enumerate(detection_files):
                detections = gpd.read_file(detection_file)
                file_split = detection_file.split("_")
                date = file_split[-2] + file_split[-3] + file_split[-4]
                dates.append(date)
                date_idx = np.where(np.array(uba_dates) == date)[0][0]
                no2_of_hour = station_obs[date_idx * 24 + 10]   # hour 10 of day of interest in flat variable
                values[row_idx, 0, idx] = no2_of_hour
                values[row_idx, 1, idx] = len(gpd.clip(detections, station_buffer))  # detections in buffer proximity
            values[np.isnan(values)] = 0
            y, x = values[row_idx][1], values[row_idx][0]
            y[x == 0] = 0
            x[y == 0] = 0
            regress = linregress(x, y)
            try:
                m, b = np.polyfit(x, y, 1)
            except np.linalg.LinAlgError:  # only zeros (nans)
                continue
            plt.plot(x, m * x + b, color="#2b2b2b")
            plt.scatter(x, y, color="#c404ab")
            plt.ylabel("S2 trucks")
            plt.xlabel("UBA station NO2 (10-11 AM)")
            plt.title("UBA station %s" % station_name, fontsize=12)
            plt.axes().xaxis.set_tick_params(labelsize=8)
            plt.axes().yaxis.set_tick_params(labelsize=8)
            plt.text(np.nanquantile(x, [0.025])[0],
                     np.nanquantile(y, [0.95])[0], "Lin. regression\nrsquared: %s\nslope: %s" % (np.round(regress.rvalue, 2),
                                                                             np.round(regress.slope, 2)),
                     fontsize=8)
            plt.savefig(os.path.join(dir_comparison_plots, station_name + "_vs_sentinel2_trucks_scatter.png"), dpi=200)
            plt.close()

    @staticmethod
    def calc_wind_direction(wind_xr):
        v_wind, u_wind = wind_xr.MeridionalWind.values, wind_xr.ZonalWind.values
        wind_direction = np.zeros_like(wind_xr.ZonalWind.values)
        for y in range(wind_direction.shape[0]):
            for x in range(wind_direction.shape[1]):
                vector = [v_wind[y, x], u_wind[y, x]]
                offset = 180 if all([value < 0 for value in vector]) or vector[1] < 0 else 0
                offset = 90 if all([vector[0] < 0, vector[1] > 0]) else offset
                offset += 90 if all([vector[0] > 0, vector[1] < 0]) else 0
                if vector[0] == 0:
                    direction = 0.
                else:
                    direction = np.degrees(np.arctan(np.abs(vector[1]) / np.abs(vector[0])))
                wind_direction[y, x] = direction + offset
        return wind_direction

    @staticmethod
    def compare_station_counts(detection_files, dates):
        # compare the processed dates with BAST station data
        validator = Validator(bast_station, aoi_file, dir_validation, dir_osm)
        station_folder = "zst" + validator.station_name.split("(")[1].split(")")[0]
        wrong = len(station_folder) == 4
        station_folder = "zst" + validator.station_name.split(") ")[1].split("(")[1][0:-1] if wrong else station_folder
        validator.station_file = os.path.join(validator.dirs["station_counts"], station_folder, station_folder +
                                              "_2018.csv")
        validator.validation_file = os.path.join(dir_validation, "series_comparison.csv")  # not default validation file
        try:
            os.remove(validator.validation_file)  # nothing should be added to existing file, hence delete
        except FileNotFoundError:
            pass
        for detection_file, date in zip(detection_files, dates):
            validator.date = date.split(" (")[0]
            validator.detections_file = detection_file
            validator.validate_with_bast()
        comparison_pd = pd.read_csv(validator.validation_file)
        station_counts = [np.float32(comparison_pd[column]) for column in ["Lzg_R1", "Lzg_R2"]]
        s2_counts = [np.float32(comparison_pd[column]) for column in ["s2_direction1", "s2_direction2"]]
        s2_colors, bast_colors = ["#e692ea", "#82068c"], ["#c6eb5a", "#8fb22a"]
        plt.figure(figsize=[10, 8])
        for s2_count_direction, s2_color in zip(s2_counts, s2_colors):
            plt.plot_date(dates, s2_count_direction, xdate=True, color=s2_color, alpha=0.5, ms=5)
            plt.plot(dates, s2_count_direction, color=s2_color)
        for bast_count_direction, bast_color in zip(station_counts, bast_colors):
            plt.plot_date(dates, bast_count_direction, xdate=True, color=bast_color, alpha=0.5, ms=5)
            plt.plot(dates, bast_count_direction, color=bast_color)
        plt.ylabel("Count")
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.2)
        s2_direction, bast_direction = "S2 direction ", "BAST Lzg direction "
        excl = "_"
        plt.legend([excl, s2_direction + "1", excl, s2_direction + "2", excl, bast_direction + "1", excl,
                    bast_direction + "2", excl], fontsize=8)
        plt.title("Trucks Sentinel-2 & BAST station Braunschweig-Flughafen", fontsize=12)
        plt.axes().xaxis.set_tick_params(labelsize=8)
        plt.axes().yaxis.set_tick_params(labelsize=8)
        plt.savefig(os.path.join(dir_comparison_plots, "s2_hannover_braunschweig_station_comparison_series.png"),
                    dpi=200)
        plt.close()


if __name__ == "__main__":
    if not os.path.exists(dir_comparison_detections):
        os.mkdir(dir_comparison_detections)
    comparison = Comparison(process_dates, aoi_file)
    comparison.run_comparison()
    comparison.plot_s2_series()
    print("Done")
