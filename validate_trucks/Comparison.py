import os
import pickle
import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
import rasterio as rio
import matplotlib.pyplot as plt
import seaborn as sns
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
    def __init__(self, dates, aoi_file_path):
        self.dates = dates
        self.bbox = gpd.read_file(aoi_file_path).to_crs("EPSG:4326").geometry.bounds

    def run_comparison(self):
        detection_files, uba_no2_arrays = [], []
        for date in self.dates:
            print(date)
            sh = SentinelHub()
            sh.set_credentials(SH_CREDENTIALS_FILE)
            sh_bbox = tuple(list(self.bbox.iloc[0]))  # xmin, ymin, xmax, ymax
            file_str = "_".join([str(coord) for coord in sh_bbox]) + "_" + date.replace("-", "_")
            split = file_str.split("_")
            d, m, y = split[-3], split[-2], split[-1]
            merged_file = os.path.join(dir_validation_data, "s2_bands_%s.tiff" % file_str)
            detections_file = os.path.join(dir_comparison_detections_boxes, "%s_detections.gpkg" % file_str)
            if os.path.exists(merged_file) or os.path.exists(detections_file):
                detection_files.append(detections_file)
            elif not os.path.exists(merged_file):
                band_stack, folder = sh.get_data(sh_bbox, [date, date], DataCollection.SENTINEL2_L2A,
                                                 ["B04", "B03", "B02", "B08", "CLM"], resolution, dir_validation_data,
                                                 merged_file)
                band_stack = None
            else:
                rf_td = RFTruckDetector()
                band_stack = rf_td.read_bands(merged_file)
                rf_td.preprocess_bands(band_stack[0:4])
                rf_td.train()
                prediction_array = rf_td.predict()
                prediction_boxes = rf_td.extract_objects(prediction_array)
                rf_td.prediction_boxes_to_gpkg(prediction_boxes, detections_file)
        uba_no2_arrays = self.compare_insitu_no2(glob(os.path.join(dir_comparison_detections_boxes, "*.gpkg")))
        for comparison_variable in comparison_variables:
            self.s2_vs_s5p_model_no2(comparison_variable, detection_files, uba_no2_arrays)

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

    def s2_vs_s5p_model_no2(self, raster_variable_name, detection_files, uba_no2_arrays):
        wind_bins_low = np.arange(0, 181, 180, dtype=np.float32)  # wind directions
        wind_bins_up = np.arange(180, 361, 180, dtype=np.float32)
        uba_station_locations_pd = pd.read_csv(uba_stations_locations_file, sep=";", index_col=0)
        var0, var1 = comparison_variables[0], comparison_variables[1]
        for row_idx in range(len(uba_station_locations_pd)):
            row = uba_station_locations_pd.iloc[row_idx]
            station_point = Point([row.lon, row.lat])
            # iterate over dates, get numbers for each date by wind direction
            observation_dict = {}
            for wind_low in wind_bins_low:
                observation_dict[str(wind_low)] = {"comparison": [], "s2": []}
            wind_arrays, s2_arrays, comparison_arrays, dates = [], [], {var0: [], var1: []}, []  # all timestamps
            meta, x_station, y_station = None, None, None
            for date, detections_file in zip(self.dates, detection_files):
                date_compact = date[-4:] + date[3:5] + date[0:2]
                try:
                    comparison_raster_file = glob(os.path.join(
                        dir_comparison_s5p, "test_tropomi_NO2_%s*.nc" % date_compact))[0]
                except IndexError:
                    continue
                else:
                    dates.append(date)
                print("Reading: %s" % comparison_raster_file)
                reference_array = xr.open_dataset(comparison_raster_file)
                lon, lat = reference_array.lon.values, reference_array.lat.values
                # location in array
                x_station = np.argmin(np.abs(lon - station_point.x))
                y_station = np.argmin(np.abs(lat - station_point.y))
                comparison_array = reference_array[raster_variable_name].values
                wind = xr.open_dataset(os.path.join(dir_comparison_wind, "Wind_U_V_%s.nc" % date_compact))
                wind_direction = self.calc_wind_direction(wind)
                wind_direction[np.isnan(comparison_array)] = np.nan
                detections = gpd.read_file(detections_file)
                detections_basename = os.path.basename(detections_file).replace(".gpkg", "")
                detections = detections.to_crs("EPSG:4326")  # because rasters are given as 4326
                detections_raster_file = os.path.join(dir_comparison_detections_rasters, detections_basename + ".tiff")
                if os.path.exists(detections_raster_file):
                    with rio.open(detections_raster_file, "r") as src:
                        meta = src.meta
                        s2_trucks_array = src.read(1)
                else:
                    s2_trucks_array = self.rasterize_s2_detections(
                        detections, reference_array, raster_variable_name, detections_raster_file)
                s2_arrays.append(s2_trucks_array.copy())
                wind_arrays.append(wind_direction.copy())
                comparison_arrays[var0].append(reference_array[var0].values.copy())
                comparison_arrays[var1].append(reference_array[var1].values.copy())
                comparison_array[s2_trucks_array < 1] = np.nan
                s2_trucks_array[np.isnan(comparison_array)] = np.nan
                shape = comparison_array.shape
                ymin, xmin = int(np.clip(y_station - 1, 0, np.inf)), int(np.clip(x_station - 1, 0, np.inf))
                ymax, xmax = int(np.clip(y_station + 2, 0, shape[0])), int(np.clip(x_station + 2, 0, shape[1]))
                comparison_subset = comparison_array[ymin:ymax, xmin:xmax]
                s2_trucks_subset = s2_trucks_array[ymin:ymax, xmin:xmax]
                wind_direction_subset = wind_direction[ymin:ymax, xmin:xmax]
                for wind_low, wind_up in zip(wind_bins_low, wind_bins_up):
                    ys, xs = np.where((wind_direction_subset >= wind_low) * (wind_direction_subset < wind_up))
                    for y, x in zip(ys, xs):
                        values = [comparison_subset[y, x], s2_trucks_subset[y, x]]
                        if any([np.isnan(value) for value in values]):
                            continue
                        else:
                            observation_dict[str(wind_low)]["comparison"].append(values[0])
                            observation_dict[str(wind_low)]["s2"].append(values[1])
            # plot values of all dates at this station by wind direction
            for wind_low, wind_up in zip(wind_bins_low, wind_bins_up):
                x = np.float32(observation_dict[str(wind_low)]["s2"])
                y = np.float32(observation_dict[str(wind_low)]["comparison"])
                self.scatter_plot_by_wind(wind_low, wind_up, x, y, raster_variable_name, row.name)
            # spatial comparison
            if raster_variable_name == comparison_variables[1]:
                correlations = self.compare_spatially(wind_arrays, s2_arrays, comparison_arrays,
                                                      wind_bins_low, wind_bins_up, dates, raster_variable_name)
                meta["count"], meta["dtype"] = correlations.shape[0], np.float32
           #     correlations_file = os.path.join(dir_comparison_plots,
            #                                     "spatial_correlations_%s.tiff" % raster_variable_name)
             #   with rio.open(correlations_file, "w", **meta) as tgt:
              #      for idx in range(meta["count"]):
               #         tgt.write(correlations[idx].astype(np.float32), idx + 1)
            comparison_values_list = list(comparison_arrays.values())
            ymin, ymax = int(np.clip(y_station - 1, 0, np.inf)), int(np.clip(y_station + 2, 0, np.inf))
            xmin, xmax = int(np.clip(x_station - 1, 0, np.inf)), int(np.clip(x_station + 2, 0, np.inf))
            comparison_at_station = np.zeros((2, np.float32(comparison_arrays[var0]).shape[0]))
            for i in range(comparison_at_station.shape[1]):
                comparison_at_station[0, i] = np.nanmean(np.float32(comparison_arrays[var0])[i, ymin:ymax, xmin:xmax])
                comparison_at_station[1, i] = np.nanmean(np.float32(comparison_arrays[var1])[i, ymin:ymax, xmin:xmax])
            uba_values = []
            for date_idx, date in enumerate(self.dates):  # more dates in uba arrays than in other data, sort out
                if date in dates:
                    uba_values.append(uba_no2_arrays[row_idx, date_idx])
            s2_arrays = np.float32(s2_arrays)
            # mean in window at station
            s2_window_mean = [np.nanmean(s2_arrays[i, ymin:ymax, xmin:xmax]) for i in range(s2_arrays.shape[0])]
            wind_mean = [np.nanmean(wind_arrays[i][ymin:ymax, xmin:xmax]) for i in range(len(wind_arrays))]
            self.line_plot_summary(dates, np.float32(s2_window_mean),
                                   comparison_at_station, np.float32(uba_values), np.float32(wind_mean), row.name)

    def compare_insitu_no2(self, detection_files):
        crs = gpd.read_file(detection_files[0]).crs
        uba_station_locations_pd = pd.read_csv(uba_stations_locations_file, sep=";", index_col=0)
        values = np.zeros((len(uba_station_locations_pd), 2, len(detection_files)))
        for row_idx in range(len(uba_station_locations_pd)):
            row = uba_station_locations_pd.iloc[row_idx]
            station_name = row.name
            station_buffer = self.get_uba_station_buffer(row, uba_station_buffer, crs)
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
            y, x = values[row_idx][0], values[row_idx][1]
            y[x == 0] = 0
            x[y == 0] = 0
            regress = linregress(x, y)
            try:
                m, b = np.polyfit(x, y, 1)
            except np.linalg.LinAlgError:  # only zeros (nans)
                continue
            plt.plot(x, m * x + b, color="#2b2b2b")
            plt.scatter(x, y, color="#c404ab")
            plt.xlabel("S2 trucks")
            plt.ylabel("UBA station NO2 (10-11 AM)")
            plt.title("UBA station %s" % station_name, fontsize=12)
            plt.axes().xaxis.set_tick_params(labelsize=8)
            plt.axes().yaxis.set_tick_params(labelsize=8)
            plt.text(np.nanquantile(x, [0.025])[0],
                     np.nanquantile(y, [0.95])[0], "Lin. regression\nrsquared: %s\nslope: %s" % (np.round(regress.rvalue, 2),
                                                                             np.round(regress.slope, 2)),
                     fontsize=8)
            plt.savefig(os.path.join(dir_comparison_plots, station_name + "_vs_sentinel2_trucks_scatter.png"), dpi=200)
            plt.close()
        return values[:, 0, :]

    @staticmethod
    def compare_spatially(wind_directions, s2_values, comparison_values, wind_bins_low, wind_bins_up, dates,
                          variable_name):
        comparison_var0 = np.float32(comparison_values[variable_name])
        comparison_var1_name = comparison_variables[0] if variable_name == comparison_variables[1] else comparison_variables[1]
        comparison_var1 = np.float32(comparison_values[comparison_var1_name])
        dates = np.array(["-".join([d.split("-")[2], d.split("-")[1], d.split("-")[0]]) for d in dates])
        wind_directions, s2_values = np.float32(wind_directions), np.float32(s2_values)
        shape, n_wind = wind_directions[0].shape, len(wind_bins_low)
        correlations = np.zeros((n_wind, shape[0], shape[1]), np.float32)
        for y in range(shape[0]):  # look into correlation for all dates at each cell
            for x in range(shape[1]):
                # differentiated by wind direction
                for idx, wind_low, wind_up in zip(range(n_wind), wind_bins_low, wind_bins_up):
                    wind_all_dates = wind_directions[:, y, x]  # wind of all dates at this cell
                    wind_indices = np.where((wind_all_dates >= wind_low) * (wind_all_dates < wind_up))[0]
                    if len(wind_indices) == 0:
                        correlations[idx, y, x] = 0
                    else:
                        var_s2, var0 = s2_values[wind_indices, y, x], comparison_var0[wind_indices, y, x]
                        if len(var_s2) < 5:
                            rvalue = 0
                        else:
                            rvalue = linregress(var_s2, var0).rvalue
                        correlations[idx, y, x] = rvalue
        # plot at positions where correlation high
        indices = np.where(correlations > 0.75)
        for idx, y, x in zip(indices[0], indices[1], indices[2]):
            wind_low, wind_up = wind_bins_low[idx], wind_bins_up[idx]
            wind_all_dates = wind_directions[:, y, x]  # wind of all dates at this cell
            wind_indices = np.where((wind_all_dates >= wind_low) * (wind_all_dates < wind_up))[0]
            sns.set(rc={"figure.figsize": (8, 4)})
            sns.set_theme(style="white")
            var_s2 = s2_values[wind_indices, y, x].copy()
            var_s2 /= np.nanmax(var_s2)
            var0 = comparison_var0[wind_indices, y, x]
            var0 /= np.nanmax(var0)
            var1 = comparison_var1[wind_indices, y, x]
            rsquared_var1 = np.round(linregress(s2_values[wind_indices, y, x], var1).rvalue, 2)
            var1 /= np.nanmax(var1)
            selected_dates = dates[wind_indices]
            ax = sns.lineplot(selected_dates, var_s2)
            ax = sns.lineplot(selected_dates, var0)
            ax = sns.lineplot(selected_dates, var1)
            plt.ylabel("Values normalized by max", fontsize=10)
            plt.tick_params(labelsize=10)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            plt.subplots_adjust(bottom=0.2, left=0.1, right=0.7)
            plt.legend(["Sentinel-2 trucks"] + comparison_variables, fontsize=10, loc="center right",
                       bbox_to_anchor=(1.5, 0.5))
            plt.text(len(selected_dates) - 0.7, 0.8, "r-squared S2 vs.\n%s=%s\nS2 vs. %s=%s" % (
                variable_name, np.round(correlations[idx, y, x], 2), comparison_var1_name, rsquared_var1),
                     fontsize=8)
            plt.title("S2 trucks vs. %s and %s at position y=%s, x=%s" % (comparison_variables[0],
                                                                          comparison_variables[1], y, x))
            fname = "s2_vs_%s_wind_%s_%s_y%s_x%s_lineplot.png"
            plt.savefig(os.path.join(dir_comparison_plots, fname % (variable_name, wind_low, wind_up, y, x)),
                        dpi=600)
            plt.close()
        return correlations

    @staticmethod
    def scatter_plot_by_wind(wind_low_threshold, wind_up_threshold, x, y, raster_variable_name, station_name):
        sns.set(rc={'figure.figsize': (9, 5)})
        sns.set_theme(style="white")
        scatter_file = os.path.join(
            dir_comparison_plots, raster_variable_name + "_wind_%s_%s_station_%s_scatterplot.png" %
                                  (wind_low_threshold, wind_up_threshold, station_name))
        # scatterplot
        if len(x) == 0 or len(y) == 0:
            return
        try:
            m, b = np.polyfit(x, y, 1)
        except np.linalg.LinAlgError:  # only zeros (nans)
            return
        regress = linregress(x, y)
        plt.plot(x, m * x + b, color="#2b2b2b")
        sns.scatterplot(x, y, color="#c404ab")
        plt.axes().xaxis.set_tick_params(labelsize=8)
        plt.axes().yaxis.set_tick_params(labelsize=8)
        plt.text(np.nanquantile(x, [0.025])[0],
                 np.nanquantile(y, [0.9])[0],
                 "Lin. regression\nrsquared: %s\nslope: %s" % (np.round(regress.rvalue, 2),
                                                               np.round(regress.slope, 2)),
                 fontsize=8)
        plt.ylabel("S2 trucks")
        plt.xlabel(raster_variable_name)
        plt.title("UBA station %s | Wind direction %s-%s" % (station_name, wind_low_threshold, wind_up_threshold))
        plt.savefig(scatter_file, dpi=300)
        plt.close()

    @staticmethod
    def line_plot_summary(dates, s2_values, comparison_arrays, uba_values, wind_values, station_name):
        dates = ["-".join([d.split("-")[2], d.split("-")[1], d.split("-")[0]]) for d in dates]
        colors = ["#5e128a", "#bbe63f", "#5f8000", "#016b05", "#0caab4"]
        names = ["Sentinel-2", "Sentinel-5P NO2 total column", "Model NO2 total column", "UBA NO2"]#, "Wind direction"]
        sns.set(rc={"figure.figsize": (9, 5)})
        sns.set_theme(style="white")
        line_file = os.path.join(dir_comparison_plots, "station_%s_lineplot.png" % station_name)
        not_nan = ~np.isnan(s2_values) * ~np.isnan(comparison_arrays[0])
        s2_no_nan = s2_values[not_nan]
        correlation_with_s5p = np.round(linregress(s2_no_nan, comparison_arrays[0][not_nan]).rvalue, 2)
        correlation_with_model = np.round(linregress(s2_no_nan, comparison_arrays[1][not_nan]).rvalue, 2)
        try:
            correlation_with_uba = np.round(linregress(s2_no_nan, uba_values[not_nan]).rvalue, 2)
        except ValueError:
            correlation_with_uba = np.zeros_like(s2_no_nan)
        for values, c in zip([s2_values, comparison_arrays[0], comparison_arrays[1], uba_values], colors):
            values_copy = values.copy().flatten()
            values_copy[~not_nan] = 0
            values_copy /= np.max(values_copy)
            ax = sns.lineplot(x=np.array(dates)[not_nan], y=values_copy[not_nan], color=c)
        txt = "r-squared S2 vs. S5p=%s\nr-squared S2 vs. Model=%s\nr-squared S2 vs. UBA=%s"
        ax.text(np.count_nonzero(not_nan), 0.8, txt % (correlation_with_s5p, correlation_with_model,
                                                       correlation_with_uba),
                fontsize=10)
        ax.xaxis.set_tick_params(labelsize=10)
        ax.yaxis.set_tick_params(labelsize=10)
        plt.ylabel("Values normalized by max", fontsize=10)
        plt.subplots_adjust(bottom=0.2, right=0.75)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.legend(names, bbox_to_anchor=(1.45, 0.5), fontsize=10, loc="center right")
        plt.title("Series S2 trucks, S5P and Model total NO2 column comparison at station %s" % station_name, fontsize=12)
        plt.tight_layout()
        plt.savefig(line_file, dpi=600)
        plt.close()

    @staticmethod
    def get_uba_station_buffer(station_pd_row, buffer_distance, crs):
        station_point_gpd = gpd.GeoDataFrame({"id": [0], "geometry": [Point([station_pd_row.lon,
                                                                             station_pd_row.lat])]}, crs="EPSG:4326")
        station_point_gpd = station_point_gpd.to_crs(crs)
        return station_point_gpd.buffer(buffer_distance)

    @staticmethod
    def rasterize_s2_detections(detections, reference_array, raster_variable_name, raster_file):
        lat, lon = reference_array.lat.values[::-1], reference_array.lon.values
        lat_resolution = np.mean(lat[1:] - lat[:-1])
        lon_resolution = np.mean(lon[1:] - lon[:-1])
        box_str = "_".join([str(np.min(coord)) + "_" + str(np.max(coord)) for coord in [lat, lon]])
        raster_file = raster_file.replace("BOX_STR", box_str)
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
        # trucks raster to gtiff
        meta = dict(dtype=np.float32, count=1, crs=detections.crs, height=s2_trucks_array.shape[0],
                    width=s2_trucks_array.shape[1], driver="GTiff", nodata=None)
        meta["transform"] = rio.transform.from_bounds(np.min(lon), np.min(lat), np.max(lon), np.max(lat), len(lon),
                                                      len(lat))
        with rio.open(raster_file, "w", **meta) as tgt:
            tgt.write(s2_trucks_array, 1)
        return s2_trucks_array

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
                    direction = -1
                else:
                    direction = np.degrees(np.arctan(np.abs(vector[1]) / np.abs(vector[0]))) + offset
                meteorological_direction = direction - 180 if direction >= 180 else direction + 180
                wind_direction[y, x] = meteorological_direction
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
            validator.validate_with_bast(date, detection_file, validator.station_file, "")
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
  #  comparison.plot_s2_series()
    print("Done")
