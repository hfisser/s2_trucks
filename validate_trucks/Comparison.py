import os
import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
import rasterio as rio
import matplotlib.pyplot as plt

from glob import glob
from scipy.stats import linregress
from shapely.geometry import box
from SentinelHubDataAccess.SentinelHub import SentinelHub, DataCollection
from detect_trucks.RandomForestTrucks import RFTruckDetector

SH_CREDENTIALS_FILE = os.path.join("F:" + os.sep + "sh", "sh.txt")

resolution = 10

dir_main = "F:\\Masterarbeit\\DLR\\project\\1_truck_detection"
dir_comparison = os.path.join(dir_main, "comparison")
dir_comparison_detections = os.path.join(dir_comparison, "detections")
dir_comparison_detections_boxes = os.path.join(dir_comparison_detections, "boxes")
dir_comparison_detections_rasters = os.path.join(dir_comparison_detections, "rasters")
dir_comparison_plots = os.path.join(dir_comparison, "plots")
dir_validation = os.path.join(dir_main, "validation")
dir_validation_data = os.path.join(dir_validation, "data", "s2", "archive")
dir_comparison_s5p = os.path.join(dir_comparison, "OUT_S5P")

for directory in [dir_comparison_detections, dir_comparison_detections_boxes, dir_comparison_detections_rasters]:
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

process_dates = [process_dates[0]]


class Comparison:
    def __init__(self, dates, aoi_file):
        self.dates = dates
        self.bbox = gpd.read_file(aoi_file).to_crs("EPSG:4326").geometry.bounds

    def process_s2(self):
        for date in self.dates:
            print("Processing: %s" % date)
            sh = SentinelHub()
            sh.set_credentials(SH_CREDENTIALS_FILE)
            sh_bbox = tuple(list(self.bbox.iloc[0]))  # xmin, ymin, xmax, ymax
            file_str = "_".join([str(coord) for coord in sh_bbox]) + date.replace("-", "_")
            merged_file = os.path.join(dir_validation_data, "s2_bands_%s.tiff" % file_str)
            band_stack, folder = sh.get_data(sh_bbox, [date, date], DataCollection.SENTINEL2_L2A,
                                             ["B04", "B03", "B02", "B08", "CLM"], resolution, dir_validation_data,
                                             merged_file)
            rf_td = RFTruckDetector()
            rf_td.preprocess_bands(band_stack[0:4])
            rf_td.train()
            prediction_array = rf_td.predict()
            prediction_boxes = rf_td.extract_objects(prediction_array)
            detections_file = os.path.join(dir_comparison_detections_boxes, "%s_detections.gpkg" % file_str)
            rf_td.prediction_boxes_to_gpkg(prediction_boxes, detections_file)
            tiles_files = self.compare(detections_file, dir_comparison_s5p)
            
    @staticmethod
    def compare(detections_file, raster_directory):
        """
        reads detections and raster tiles, number of detections in each cell is compared with n detections
        :param detections_file: str file path to detections
        :param raster_directory: str directory where netcdf rasters are saved
        :return:
        """
        # load Sentinel-5P tiles and rasterize detections on that grid
        detections = gpd.read_file(detections_file)
        detections_basename = os.path.basename(detections_file)
        detections = detections.to_crs("EPSG:4326")  # because rasters are given as 4326
        raster_files = glob(os.path.join(raster_directory, "*.nc"))
        s2_trucks_arrays, comparison_arrays = [], []
        for raster_file in raster_files:
            reference_array = xr.open_dataset(raster_file)
            lat, lon = reference_array.lat.values, reference_array.lon.values
            lat_resolution = np.mean(lat[1:] - lat[0:-1])
            lon_resolution = np.mean(lon[1:] - lon[0:-1])
            lat_ul, lon_ul = lat - lat_resolution, lon - lon_resolution  # upper left corner coordinates
            box_str = "_".join([str(coord[0]) + "_" + str(coord[-1]) for coord in [lat_ul, lon_ul]])
            s2_trucks_array = np.zeros_like(reference_array.var_mod_NO2_AK_coulumn.values)
            # iterate over cells and count number of s2 trucks
            for y in s2_trucks_array:
                for x in s2_trucks_array:
                    ymin, ymax = lat_ul[y], lat_ul[y + 1]
                    xmin, xmax = lon_ul[x], lon_ul[x + 1]
                    cell_box_gpd = gpd.GeoDataFrame({"id": [0]}, geometry=[box(xmin, ymin, xmax, ymax)],
                                                    crs=detections.crs)
                    s2_trucks_array[y, x] = len(gpd.clip(detections, cell_box_gpd))  # number of detections in cell
            detections_raster_file = os.path.join(dir_comparison_detections_rasters,
                                                  detections_basename + box_str + ".nc")
            s2_trucks_xr = xr.Dataset({"s2_trucks": s2_trucks_array}, coords={"lat": lat, "lon": lon})
            s2_trucks_xr.to_netcdf(detections_raster_file)
            s2_trucks_arrays.append(s2_trucks_array)
            comparison_arrays.append(reference_array.var_mod_NO2_AK_coulumn.values)
        s2_trucks_arrays_np, comparison_arrays_np = np.float32(s2_trucks_arrays), np.float32(comparison_arrays)
        # plot detections vs. comparison array
        s2_array_flat, comparison_array_flat = s2_trucks_arrays_np.flatten(), comparison_arrays_np.flatten()
        plot = plt.scatter(s2_array_flat, comparison_array_flat)  # xy of flat arrays
        folder = os.path.dirname(raster_directory)
        comparison_name = {"OUT_S5P": "S5P", "OUT_Insitu": "Insitu"}[folder]
        regression = linregress(s2_array_flat, comparison_array_flat)
        slope, intercept = regression.slope, regression.intercept
        plt.xlabel = "s2_trucks"
        plt.ylabel = comparison_name
        plt.title = "S2 trucks vs. " + comparison_name
        plt.plot(s2_array_flat, slope * x + intercept)
        plt.text(10, 0, "r2: %s\nslope: %s" % (str(slope), str(intercept)))
        plt.savefig(os.path.join(dir_comparison_plots, detections_basename, "_vs%s.png" % comparison_name))


if __name__ == "__main__":
    if not os.path.exists(dir_comparison_detections):
        os.mkdir(dir_comparison_detections)
    comparison = Comparison(process_dates, aoi_file_path)
    comparison.process_s2()
