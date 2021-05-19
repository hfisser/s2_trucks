import os
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt
from matplotlib import rcParams
from glob import glob
from datetime import datetime
from shapely.geometry import box, Point
from rasterio.merge import merge
from scipy.stats import linregress
from array_utils.geocoding import get_coords_array
from utils.Essentials import WEEKDAYS, MONTHS_SHORT, MONTHS_LONG
from utils.ProgressBar import ProgressBar
from rasterio.transform import Affine
from SentinelHubDataAccess.SentinelHub import SentinelHub

NAIROBI_UTM = "EPSG:32737"
S2_COLOR = "#611840"
WEEKDAY_LIST = np.array(WEEKDAYS())
MONTHS_LIST = np.array(MONTHS_SHORT())
MONTHS_LIST_LONG = np.array(MONTHS_LONG())

rcParams["font.serif"] = "Times New Roman"
rcParams["font.family"] = "serif"

dirs = dict(main="F:\\Masterarbeit\\DLR\\project\\1_truck_detection\\application")
for folder in ["detection_boxes", "detection_rasters", "mean_rasters", "s2_data", "aq_stations"]:
    dirs[folder] = os.path.join(dirs["main"], folder)
    if not os.path.exists(dirs[folder]):
        os.mkdir(dirs[folder])
dirs["save_archive"] = os.path.join(dirs["s2_data"], "archive")
dirs["validation_plots"] = os.path.join(os.path.dirname(dirs["main"]), "validation", "plots")
aoi_file_path = os.path.join(dirs["main"], "aoi_nairobi.gpkg")


class KenyaValidator:
    def __init__(self):
        self.abc = ""

    def main(self):
        self.compare_with_aq_stations()
        #detection_dict = self.create_detection_series_plots()

    def create_detection_series_plots(self):
        station_aois = self.create_station_aois()
        station_counts = self.prep_station_counts()
        files = np.array(glob(os.path.join(dirs["detection_boxes"], "*")))
        files = files[[os.path.isfile(f) for f in files]]
        aoi_names = [self.get_aoi_str(f) for f in files]
        dates = [self.get_date(f) for f in files]
        weekdays_aoi = [WEEKDAY_LIST[datetime.fromisoformat(d).weekday()] for d in dates]
        aoi_names_unique = np.unique(aoi_names)
        aoi_names_unique = aoi_names_unique[aoi_names_unique != "Mariakani"]
        n_detections = dict.fromkeys(aoi_names_unique)
        detection_weekdays = dict.fromkeys(aoi_names_unique)
        detection_dates = dict.fromkeys(aoi_names_unique)
        weekday_counts = dict.fromkeys(WEEKDAY_LIST)
        for aoi_name in aoi_names_unique:
            files_aoi = glob(os.path.join(dirs["detection_boxes"], "*%s*" % aoi_name))
            dates_aoi = np.array([self.get_date(f) for f in files_aoi])
            n_detections[aoi_name] = np.zeros(len(files_aoi))
            detection_weekdays[aoi_name] = np.array([WEEKDAY_LIST[datetime.fromisoformat(d).weekday()] for d in dates_aoi])
            detection_dates[aoi_name] = dates_aoi
            for idx, file in enumerate(files_aoi):
                detections = gpd.read_file(file)
                if "No detections" in detections.columns:
                    n_detections[aoi_name][idx] = 0
                else:
                    n_detections[aoi_name][idx] = len(gpd.clip(
                        detections, station_aois[station_aois["name"] == aoi_name].to_crs(detections.crs)))
            for w in WEEKDAY_LIST:
                if weekday_counts[w] is None:
                    weekday_counts[w] = []
                try:
                    weekday_detections = n_detections[aoi_name][detection_weekdays[aoi_name] == w]
                except IndexError:
                    continue
                else:
                    weekday_counts[w].append(np.nanmean(weekday_detections))
        weekday_means = np.zeros(len(WEEKDAY_LIST))
        for i, w in enumerate(weekday_counts.keys()):
            weekday_means[i] = np.nanmean(np.hstack(weekday_counts[w]))
        # create lineplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 5))
        for aoi_name, ax in zip(n_detections.keys(), axes.flatten()):
            counts_aoi = n_detections[aoi_name]
            dates_aoi = np.array([np.datetime64(d) for d in detection_dates[aoi_name]])
            dates_argsort = np.argsort(dates_aoi)
            dates_formatted = [MONTHS_LIST[int(str(d).split("-")[1]) - 1] + "-" + str(d).split("-")[-1] for d in dates_aoi]
            dates_formatted = np.array(dates_formatted)[dates_argsort]
            ax.plot(dates_formatted, counts_aoi[dates_argsort], color=S2_COLOR, linewidth=2)
            ax.set_xticks(dates_formatted)
            ax.set_xticklabels(dates_formatted, rotation=90)
            ax.set_title(aoi_name)
            ax.set_yticks(np.int16(ax.get_yticks()))
            ax.set_ylim(0, np.max(counts_aoi) * 1.1)
            off = 0.25 if aoi_name == "Athi River" else 0.5
            ax.text(off, np.max(counts_aoi) * 0.87, "Mean: %s\nStd.: %s" % (np.round(np.nanmean(counts_aoi), 2),
                                                                            np.round(np.nanstd(counts_aoi), 2)))
           # plt.xlim(0, len(date_sort) - 1)
            ax.set_xlim(0, len(dates_formatted) - 1)
        fig.tight_layout()
        fig.savefig(os.path.join(dirs["validation_plots"], "kenya_detections_grouped_lineplot.png"), dpi=600)
        plt.close(fig)
        # create comparison scatter with station
        s2_counts = {nm: np.nanmean(n_detections[nm]) for nm in n_detections.keys()}
        fig, ax = plt.subplots(figsize=(3, 2.75))
        ax.scatter(s2_counts.values(), station_counts.values(), s=40, color=S2_COLOR)
        ax.set_xlim(0, np.max(list(s2_counts.values())) * 1.1)
        ax.set_ylabel("Weigh station")
        ax.set_xlabel("Sentinel-2")
        regress = linregress(list(s2_counts.values()), list(station_counts.values()))
        ax.text(0.5, np.max(list(station_counts.values())) * 0.95,
                "pearson r-value: %s" % (np.round(regress.rvalue, 2)))
        for name, yoff, xoff in zip(s2_counts.keys(), [0.89, 1.4, 1.02, 1.08], [0.77, 1.05, 1.05, 1.05]):
            ax.text(s2_counts[name] * xoff, station_counts[name] * yoff, name)
        fig.tight_layout()
        fig.savefig(os.path.join(dirs["validation_plots"], "kenya_s2_station_scatter.png"), dpi=600)
        plt.close(fig)
        return {"n_detections": n_detections, "weekdays": detection_weekdays, "dates": detection_dates}

    @staticmethod
    def prep_station_counts():
        counts_pd = pd.read_csv(os.path.join(dirs["main"], "kenha_station_data_2018.csv"))
        station_counts = {}
        for station_name in np.unique(counts_pd["station_name"]):
            station_counts[station_name] = np.nanmean(counts_pd[counts_pd["station_name"] == station_name]["count"])
        del station_counts["Mariakani"]
        return station_counts

    @staticmethod
    def create_station_aois():
        m = [39.437765, -3.835263]
        stations = {"Athi River": [36.963629, -1.430875],
                    "Busia": [34.138946, 0.448257],
                    "Gilgil": [36.326938, -0.523195],
                    "Webuye": [34.752944, 0.606695]}
        stations = {key: Point(value[0], value[1]) for key, value in stations.items()}
        station_points = gpd.GeoDataFrame({"geometry": stations.values(), "name": stations.keys()}, crs="EPSG:4326")
        station_points.to_file(os.path.join(dirs["main"], "station_points.gpkg"), driver="GPKG")
        station_aois = station_points.to_crs(NAIROBI_UTM).buffer(15000)
        return gpd.GeoDataFrame({"geometry": station_aois, "name": stations.keys()}, crs=NAIROBI_UTM).to_crs("EPSG:4326")

    def compare_with_aq_stations(self):
        nairobi_str = "36.69794740804024_-1.4111878854315625_36.95546936051763_-1.1352715077772113"
        detection_files = glob(os.path.join(dirs["detection_boxes"], "archive", "*%s*.gpkg" % nairobi_str))
        station_files = glob(os.path.join(dirs["aq_stations"], "*.csv"))
        measurements = pd.DataFrame()
        station_points, station_buffers = {}, {}
        for row_idx, detection_file in enumerate(detection_files):
            date = self.get_date(detection_file)
            other_files_of_date = glob(os.path.join(dirs["detection_boxes"], "archive", "*%s*.gpkg" % date))
            other_detections = []
            for other_file in other_files_of_date:
                other_detections.append(gpd.read_file(other_file))
            y, m = date.split("-")[0], MONTHS_LIST_LONG[int(date.split("-")[1]) - 1]
            m = m[0].lower() + m[1:]
            try:
                station_data = pd.read_csv(glob(os.path.join(dirs["aq_stations"], "_".join([m, y, "*.csv"])))[0], sep=";")
            except IndexError:
                continue
            station_data = station_data[station_data["value_type"] == "P2"]
            for sensor_id in [100, 44, 43]:
                sensor_data = station_data[station_data["sensor_id"] == sensor_id]
                timestamp = sensor_data.timestamp
                error, n = True, -1
                times = ["10", "09", "11", "08", "12", "13", "14", "15", "16"]
                while error and n < (len(times) - 1):
                    n += 1
                    match = self.get_ts_match_station_data(timestamp, date, times[n])
                    try:
                        if len(match) > 1:
                            measurements.loc[row_idx, str(sensor_id)] = np.nanmean(np.float32(sensor_data[match].value))
                        else:
                            measurements.loc[row_idx, str(sensor_id)] = sensor_data[match].iloc[0].value
                        error = False if np.count_nonzero(match) > 0 else True
                    except IndexError:
                        error = True
                if error:
                    measurements.loc[row_idx, str(sensor_id)] = np.nan
                    continue
                else:
                    row = sensor_data[match].iloc[0]
                    lon, lat = row.lon, row.lat
                    measurements.loc[row_idx, str(sensor_id)] = row.value
                    measurements.loc[row_idx, str(sensor_id) + "_date"] = date
                    station_point = gpd.GeoDataFrame({"geometry": [Point(float(lon), float(lat))]},
                                                       crs="EPSG:4326")
                    station_points[sensor_id] = station_point
                    station_buffer = station_point.to_crs(NAIROBI_UTM).buffer(5000)
                    station_buffers[sensor_id] = station_buffer
                    g = gpd.GeoDataFrame(pd.concat(other_detections, join="inner"))
                    g.index = range(len(g))
                    clipped = gpd.clip(g, station_buffer)
                    measurements.loc[row_idx, str(sensor_id) + "_Sentinel-2"] = len(clipped)

        fig, axes = plt.subplots(1, 4, figsize=(12, 2.5), gridspec_kw={"width_ratios": [2, 1, 2, 1]})
        idx = -1
        for station_name in ["44", "100"]:
            idx += 1
            if (idx % 2) == 0:
                dates = np.array(list(measurements[station_name + "_date"]))
                station_values = np.float32(list(measurements[station_name]))
                s2_values = np.float32(list(measurements["%s_Sentinel-2" % station_name]))
                not_nan = ~np.isnan(station_values) * ~np.isnan(s2_values)
                station_values, s2_values, dates = station_values[not_nan], s2_values[not_nan], dates[not_nan]
                ax = axes.flatten()[idx]
                ax.plot(dates, station_values, color="#4d617c", linewidth=2)
                ylabel = "Station PM2.5 [µg/m3]"
                ax.set_ylabel(ylabel)
                ax.set_title("Station ID: %s" % station_name)
                ax.set_xticklabels(dates, rotation=90)
                ax.set_xlim(0, len(dates) - 1)
                ax1 = ax.twinx()
                ax1.plot(dates, s2_values, color=S2_COLOR, linewidth=2)
                ax1.set_ylabel("Sentinel-2 trucks")
                ax1.set_xticklabels(dates, rotation=90)
                ax = axes.flatten()[idx + 1]
                ax.scatter(s2_values, station_values, s=9, color=S2_COLOR)
                ax.set_ylabel(ylabel)
                ax.set_xlabel("Sentinel-2 trucks")
                try:
                    m, b = np.polyfit(s2_values, station_values, 1)
                except np.linalg.LinAlgError:
                    pass
                else:
                    ax.plot(s2_values, m * s2_values + b, color="black")
                ax.set_title("pearson r-value: %s" % np.round(linregress(s2_values, station_values).rvalue, 2))
                idx += 1
        fig.tight_layout()
        fig.savefig(os.path.join(dirs["validation_plots"], "kenya_s2_trucks_pm2.5_station.png"), dpi=500)
        plt.close(fig)

        station_files = glob(os.path.join(dirs["main"], "*pm2.5.csv"))
        detection_files = glob(os.path.join(dirs["detection_boxes"], "*nairobi*.gpkg"))
        detection_dates = np.array([self.get_date(f) for f in detection_files])
        detections = [gpd.read_file(f) for f in detection_files]
        rvalues = {}
        for station_file in station_files:
            data = pd.read_csv(station_file)
            buffer = gpd.GeoDataFrame({"geometry": [Point(
                data.iloc[0].longitude, data.iloc[0].latitude)]}, crs="EPSG:4326").to_crs(NAIROBI_UTM).buffer(5000)
            n_detections = np.zeros(len(detections))
            for i, detection_boxes in enumerate(detections):
                n_detections[i] = len(gpd.clip(detection_boxes.to_crs(NAIROBI_UTM), buffer))
            station_measurements, s2_detections = {}, {}
            for i, date in enumerate(np.array(data.utc)[::-1]):
                split = date.split("T")
                the_date, the_hour = split[0], split[1].split(":")[0]
                if the_hour == "10" and the_date in detection_dates:
                    station_measurements[the_date] = data.loc[i, "value"]
                    s2_detections[the_date] = n_detections[np.where(detection_dates == the_date)[0][0]]
            rvalues[os.path.basename(station_file)] = linregress(list(s2_detections.values()),
                                                                 list(station_measurements.values())).rvalue
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(station_measurements.keys(), station_measurements.values(), color="#00FF00")
            ax1 = ax.twinx()
            ax1.plot(s2_detections.keys(), np.int16(list(s2_detections.values())), color=S2_COLOR)

    @staticmethod
    def get_ts_match_station_data(ts, date, time):
        return np.array([d.split("T")[0] == date and d.split("T")[1].split(":")[0] == time for d in ts])

    @staticmethod
    def get_date(file):
        return os.path.basename(file).split(".")[-2].split("_")[-1]

    @staticmethod
    def get_aoi_str(file):
        return os.path.basename(file).split("_2018")[0].split("_")[-1]


if __name__ == "__main__":
    kenya_validator = KenyaValidator()
    kenya_validator.main()
