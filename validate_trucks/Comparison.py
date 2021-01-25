import os
import geopandas as gpd
import pandas as pd

from SentinelHubDataAccess.SentinelHub import SentinelHub, DataCollection
from detect_trucks.RandomForestTrucks import RFTruckDetector

SH_CREDENTIALS_FILE = os.path.join("F:" + os.sep + "sh", "sh.txt")

resolution = 10

dir_main = "F:\\Masterarbeit\\DLR\\project\\1_truck_detection"
dir_comparison = os.path.join(dir_main, "comparison")
dir_comparison_detections = os.path.join(dir_comparison, "detections")
dir_validation = os.path.join(dir_main, "validation")
dir_validation_data = os.path.join(dir_validation, "data", "s2", "archive")

aoi_file_path = os.path.join(dir_comparison, "aoi_h_bs.geojson")
process_dates = ["10.04.2018"]


class Comparison:
    def __init__(self, dates, aoi_file):
        self.dates = dates
        self.bbox = gpd.read_file(aoi_file).to_crs("EPSG:4326").geometry.bounds

    def process_s2(self):
        for date in self.dates:
            print("Processing: %s" % date)
            sh = SentinelHub()
            sh.set_credentials(SH_CREDENTIALS_FILE)
            sh_bbox = tuple(list(self.bbox.iloc[0]))
            merged_file = os.path.join(dir_validation_data, "s2_bands_%s.tiff"
                                       % "_".join([str(coord) for coord in sh_bbox]))
            band_stack, folder = sh.get_data(sh_bbox, [date, date], DataCollection.SENTINEL2_L2A,
                                             ["B04", "B03", "B02", "B08", "CLM"], resolution, dir_validation_data,
                                             merged_file)
            rf_td = RFTruckDetector()
            rf_td.preprocess_bands(band_stack[0:4])
            rf_td.train()
            prediction_array = rf_td.predict()
            prediction_boxes = rf_td.extract_objects(prediction_array)
            detections_file = os.path.join(dir_comparison_detections, "")
            rf_td.prediction_boxes_to_gpkg(prediction_boxes, detections_file)


if __name__ == "__main__":
    if not os.path.exists(dir_comparison_detections):
        os.mkdir(dir_comparison_detections)
    comparison = Comparison(process_dates, aoi_file_path)
    comparison.process_s2()
