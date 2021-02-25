import os
import geopandas as gpd
import numpy as np

from detect_trucks.RandomForestTrucks import RFTruckDetector
from SentinelHubDataAccess.SentinelHub import SentinelHub
from SentinelHubDataAccess.SentinelHub import SentinelHub, DataCollection

SH_CREDENTIALS_FILE = os.path.join("F:" + os.sep + "sh", "sh.txt")

dirs = dict(main="F:\\Masterarbeit\\DLR\\project\\1_truck_detection\\application")
for folder in ["detection_boxes", "detection_rasters", "s2_data"]:
    dirs[folder] = os.path.join(dirs["main"], folder)
    if not os.path.exists(dirs[folder]):
        os.mkdir(dirs[folder])

aoi_file_path = os.path.join(dirs["main"], "aoi_kenya.gpkg")
bands = ["B04", "B03", "B02", "B08", "CLM"]
process_periods = [["2019-03-05", "2019-03-05"], ["2019-03-15", "2019-03-15"]]
resolution = 10
min_valid_percentage = 90  # minimum % valid pixels after cloud masking (masked to OSM roads)


class TrucksProcessor:
    def process(self, aoi_file, periods):
        aoi = gpd.read_file(aoi_file)
        aoi_sh = aoi.geometry.total_bounds
        sh = SentinelHub()
        sh.set_credentials(SH_CREDENTIALS_FILE)
        file_str = "%s_%s_%s_%s_%s" % (aoi_sh[0], aoi_sh[1], aoi_sh[2], aoi_sh[3],
                                       os.path.basename(aoi_file).replace(".gpkg", ""))
        for period in periods:
            file_str_period = file_str + "_%s_%s" % (period[0], period[1])
            s2_data_file = os.path.join(dirs["s2_data"], "s2_bands_%s.tiff" % file_str_period)
            detection_boxes_file = os.path.join(dirs["detection_boxes"], "detection_boxes_%s.gpkg" % file_str_period)
            kwargs = dict(bbox=aoi_sh, period=period, dataset=DataCollection.SENTINEL2_L2A, bands=bands,
                          resolution=resolution, dir_save=dirs["s2_data"], merged_file=s2_data_file,
                          mosaicking_order="leastCC")
            data_available = sh.data_available(kwargs)
            if data_available:
                band_stack, data_folder = sh.get_data(**kwargs)
                band_stack = None
                rf_td = RFTruckDetector()
                band_stack = rf_td.read_bands(s2_data_file)
                rf_td.preprocess_bands(band_stack[0:4])
                n_valid = np.count_nonzero(~np.isnan(rf_td.variables[0]))  # n valid pixels masked to OSM roads
                band_stack[4, band_stack[4] == 0] = 1  # nocloud=0
                band_stack[4, band_stack[4] == 255] = 0  # cloud=255
                rf_td.mask_clouds(band_stack[4])
                n_valid_masked = np.count_nonzero(~np.isnan(rf_td.variables[0]))  # n valid pixels cloud-masked
                valid_percentage = (n_valid_masked / n_valid) * 100
                if valid_percentage < min_valid_percentage:
                    continue
                else:
                    prediction_array = rf_td.predict()
                    detection_boxes = rf_td.extract_objects(prediction_array)
                    detection_boxes["detection_date"] = period[0]
                    rf_td.prediction_boxes_to_gpkg(detection_boxes, detection_boxes_file)
            else:
                continue


if __name__ == "__main__":
    processor = TrucksProcessor()
    processor.process(aoi_file_path, process_periods)
