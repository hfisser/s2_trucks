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
from shapely.geometry import Point, box
from shapely.geometry.linestring import LineString
from fiona.errors import DriverError
from rasterio.merge import merge

from osm_utils.utils import get_roads, rasterize_osm
from array_utils.points import raster_to_points
from array_utils.geocoding import lat_from_meta, lon_from_meta
from SentinelHubDataAccess.SentinelHub import SentinelHub, DataCollection
from detect_trucks.RandomForestTrucks import RFTruckDetector

SH_CREDENTIALS_FILE = os.path.join("F:" + os.sep + "sh", "sh.txt")

resolution = 10

dir_main = "F:\\Masterarbeit\\DLR\\project\\1_truck_detection"
dir_comparison = os.path.join(dir_main, "comparison")
dir_validation = os.path.join(dir_main, "validation")
dir_validation_data = os.path.join(dir_validation, "data", "s2", "archive")

aoi_file_path = os.path.join(dir_comparison, "aoi_h_bs.geojson")
process_dates = []


class Comparison:
    def __init__(self, dates, aoi_file):
        self.dates = dates
        self.bbox = gpd.read_file(aoi_file).geometry.bounds

    def process_s2(self):
        for date in self.dates:
            sh = SentinelHub()
            sh.set_credentials(SH_CREDENTIALS_FILE)
            sh_bbox = tuple(list(self.bbox.iloc[0]))
            splitted_boxes = sh.split_box(sh_bbox, resolution)  # bbox may be too large, hence split (if too large)
            merged_file = os.path.join(dir_validation_data, "s2_bands_%s_%s_merged.tiff" %
                                       (self.station_name_clear, date))
            



