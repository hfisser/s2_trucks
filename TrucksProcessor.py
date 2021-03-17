import os
import geopandas as gpd
import numpy as np
import pandas as pd
from datetime import date, timedelta
from detect_trucks.RandomForestTrucks import RFTruckDetector
from SentinelHubDataAccess.SentinelHub import SentinelHub, DataCollection

SH_CREDENTIALS_FILE = os.path.join("F:" + os.sep + "sh", "sh.txt")

dirs = dict(main="F:\\Masterarbeit\\DLR\\project\\1_truck_detection\\application")
for folder in ["detection_boxes", "detection_rasters", "s2_data"]:
    dirs[folder] = os.path.join(dirs["main"], folder)
    if not os.path.exists(dirs[folder]):
        os.mkdir(dirs[folder])
dirs["save_archive"] = os.path.join(dirs["s2_data"], "archive")
if not os.path.exists(dirs["save_archive"]):
    os.mkdir(dirs["save_archive"])

aoi_file_path = os.path.join(dirs["main"], "aoi_nairobi.gpkg")
bands = ["B04", "B03", "B02", "B08"]
overall_period = ["2019-01-01", "2019-12-31"]
resolution = 10
min_valid_percentage = 90  # minimum % valid pixels after cloud masking (masked to OSM roads)


class TrucksProcessor:
    def __init__(self, given_period):
        self.process_periods = self.generate_process_periods(given_period)

    def process(self, aoi_file):
        csep = "-" * 100
        sh = SentinelHub()
        aoi = gpd.read_file(aoi_file).geometry.iloc[0]
        splitted_boxes = sh.split_large_polygon_epsg4326(aoi, resolution)
        n_dates_processed = 0
        obs_file = os.path.join(dirs["save_archive"], "obs_%s.csv" % os.path.basename(aoi_file).split(".gpkg")[0])
        for idx, sh_box in enumerate(splitted_boxes):
            print("%s\nBox [%s/%s]" % (csep, idx, len(splitted_boxes)))
            sh_box_list = list(sh_box)
            file_str = "%s_%s_%s_%s_%s" % (sh_box_list[0], sh_box_list[1], sh_box_list[2], sh_box_list[3],
                                           os.path.basename(aoi_file).replace(".gpkg", ""))
            for period in self.process_periods:
                file_str_period = file_str + "_%s_%s" % (period[0], period[1])
                s2_data_file = os.path.join(dirs["save_archive"], "s2_bands_%s.tiff" % file_str_period)
                data_yet_there, sh, yet_checked = os.path.exists(s2_data_file), SentinelHub(), False
                try:
                    obs_pd = pd.read_csv(obs_file, index_col=0)
                except FileNotFoundError:
                    obs_pd = pd.DataFrame()
                    obs_pd.to_csv(obs_file)
                try:
                    yet_checked = period[0] in np.array(obs_pd[s2_data_file])
                except KeyError:
                    pass
                finally:
                    if yet_checked:
                        continue
                detection_boxes_file = os.path.join(dirs["detection_boxes"], "detection_boxes_%s.gpkg" % file_str_period)
                kwargs = dict(bbox=sh_box_list, period=period, dataset=DataCollection.SENTINEL2_L2A, bands=bands,
                              resolution=resolution, dir_save=dirs["save_archive"], merged_file=s2_data_file,
                              mosaicking_order="leastCC")
                if data_yet_there:
                    data_available = True
                else:
                    sh.set_credentials(SH_CREDENTIALS_FILE)
                    data_available = sh.data_available(kwargs)
                if data_available:
                    if data_yet_there:
                        has_obs = data_yet_there
                    else:
                        kwargs_copy = kwargs.copy()
                        kwargs_copy["bands"] = ["CLM"]
                        clm, data_folder = sh.get_data(**kwargs_copy)  # get only cloud mask
                        has_obs = self.has_observations(s2_data_file)
                        try:
                            os.remove(kwargs_copy["merged_file"])  # cloud mask
                        except FileNotFoundError:
                            pass
                    if has_obs:
                        print("Processing date: %s" % period[0])
                        # get data and do detection
                        kwargs["bands"] = bands[0:-1]  # excl. clm, we have it yet
                        band_stack, data_folder = sh.get_data(**kwargs)  # get full data
                        band_stack = None
                        rf_td = RFTruckDetector()
                        band_stack = rf_td.read_bands(s2_data_file)
                        try:
                            rf_td.preprocess_bands(band_stack[0:4])
                        except ValueError:
                            continue
                        prediction = rf_td.predict()
                        detection_boxes = rf_td.extract_objects(prediction)
                        detection_boxes["detection_date"] = period[0]
                        rf_td.prediction_boxes_to_gpkg(detection_boxes, detection_boxes_file)
                        n_dates_processed += 1
                    else:
                        # add date for file in order to avoid duplicate check
                        self.register_non_available_date(period[0], obs_pd, obs_file, s2_data_file)
                else:
                    self.register_non_available_date(period[0], obs_pd, obs_file, s2_data_file)

        print("Processed %s dates\n%s" % (n_dates_processed, csep))

    @staticmethod
    def has_observations(cloud_mask_file):
        rf_td = RFTruckDetector()
        clm = rf_td.read_bands(cloud_mask_file)
        clm[clm == 0] = 1  # nocloud=0
        clm[clm == 255] = 0  # cloud=255
        pseudo_band = np.random.random(len(clm.flatten())).reshape((clm.shape[0], clm.shape[1]))
        pseudo_band_stack = np.float32([pseudo_band] * 4)  # run pseudo preprocessing due to OSM masking
        try:
            rf_td.preprocess_bands(pseudo_band_stack)
        except ValueError:
            return False
        n_valid = np.count_nonzero(~np.isnan(rf_td.variables[0]))  # n valid pixels masked to OSM roads
        rf_td.mask_clouds(clm)
        n_valid_masked = np.count_nonzero(~np.isnan(rf_td.variables[0]))  # n valid pixels cloud-masked
        valid_percentage = (n_valid_masked / n_valid) * 100
        rf_td, clm = None, None
        return valid_percentage > min_valid_percentage

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
    def register_non_available_date(the_date, table, table_file, raster_file):
        try:
            table.loc[len(table[raster_file].dropna()), raster_file] = the_date
        except KeyError:
            table[raster_file] = [the_date]
        table.to_csv(table_file)


if __name__ == "__main__":
    processor = TrucksProcessor(overall_period)
    processor.process(aoi_file_path)
