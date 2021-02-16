import os
import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
import rasterio as rio
import matplotlib.pyplot as plt

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
dir_validation = os.path.join(dir_main, "validation")
dir_osm = os.path.join(dir_main, "code", "detect_trucks", "AUXILIARY", "osm")
aoi_file = os.path.join(dir_comparison, "aoi_h_bs.geojson")

for directory in [dir_comparison_detections, dir_comparison_detections_boxes, dir_comparison_detections_rasters,
                  dir_comparison_plots]:
    if not os.path.exists(directory):
        os.mkdir(directory)

aoi_file_path = os.path.join(dir_comparison, "aoi_h_bs.geojson")
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
comparison_variable = "var_mod_NO2_AK_coulumn"


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
            file_prefix = {"var_mod_NO2_AK_coulumn": "test_tropomi_NO2_"}[comparison_variable]
            split = file_str.split("_")
            d, m, y = split[-3], split[-2], split[-1]
            try:
                comparison_raster_file = glob(os.path.join(dir_comparison_s5p, file_prefix + "%s%s%s*.nc" % (y, m, d)))[0]
            except IndexError:
                continue
            self.compare(detections_file, comparison_raster_file, comparison_variable)

    @staticmethod
    def compare(detections_file, raster_file, raster_variable_name):
        """
        reads detections and raster tiles, number of detections in each cell is compared with n detections
        :param detections_file: str file path to detections
        :param raster_file: str file path of raster
        :param raster_variable_name: str name of the variable in the netcdf
        :return:
        """
        # load Sentinel-5P tiles and rasterize detections on that grid
        detections = gpd.read_file(detections_file)
        detections_basename = os.path.basename(detections_file)
        detections = detections.to_crs("EPSG:4326")  # because rasters are given as 4326
        reference_array = xr.open_dataset(raster_file)
        lat, lon = reference_array.lat.values[::-1], reference_array.lon.values
        lat_resolution = np.mean(lat[1:] - lat[0:-1])
        lon_resolution = np.mean(lon[1:] - lon[0:-1])
        box_str = "_".join([str(np.min(coord)) + "_" + str(np.max(coord)) for coord in [lat, lon]])
        comparison_array = reference_array[raster_variable_name].values
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
                                              ".tiff")
        # trucks raster to gtiff
        meta = dict(dtype=np.float32, count=1, crs=detections.crs, height=s2_trucks_array.shape[0],
                    width=s2_trucks_array.shape[1], driver="GTiff", nodata=None)
        meta["transform"] = rio.transform.from_bounds(np.min(lon), np.min(lat), np.max(lon), np.max(lat), len(lon),
                                                      len(lat))
        with rio.open(detections_raster_file, "w", **meta) as tgt:
            tgt.write(s2_trucks_array, 1)
        # plot detections vs. comparison array
        s2_array_flat, comparison_array_flat = s2_trucks_array.flatten(), comparison_array.flatten()
        plt.scatter(s2_array_flat, comparison_array_flat)  # xy of flat arrays
        folder = os.path.dirname(raster_file).split(os.sep)[-1]
        comparison_name = {"OUT_S5P": "S5P", "OUT_Insitu": "Insitu"}[folder]
        regression = linregress(s2_array_flat, comparison_array_flat)
        slope, intercept = regression.slope, regression.intercept
        plt.xlabel = "s2_trucks"
        plt.ylabel = comparison_name
        plt.title = "S2 trucks vs. " + comparison_name
       # plot(s2_array_flat, slope * x + intercept)
      #  plot.text(10, 0, "r2: %s\nslope: %s" % (str(slope), str(intercept)))
        plt.savefig(os.path.join(dir_comparison_plots, detections_basename.split(".gpkg")[0] + "_vs%s.png" %
                                 comparison_name))
        plt.close()

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
                    dpi=150)
        plt.close()


if __name__ == "__main__":
    if not os.path.exists(dir_comparison_detections):
        os.mkdir(dir_comparison_detections)
    comparison = Comparison(process_dates, aoi_file_path)
    comparison.run_comparison()
    comparison.plot_s2_series()
    print("Done")
