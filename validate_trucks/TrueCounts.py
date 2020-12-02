import pandas as pd
import geopandas as gpd
from obspy.geodetics import kilometers2degrees

NAME_DATE = "Datum"
NAME_HOUR = "Stunde"
NAME_TR1 = "Lkw_R1"
NAME_TR2 = "Lkw_R2"
CSV_SEP = ";"


class TrueCounts:
    def __init__(self, true_count_csv, station_xy):
        self.data = pd.read_csv(true_count_csv, sep=CSV_SEP)
        self.station_xy = station_xy
        self.data_subset = None  # pd
        self.counts = None
        self.cars = None
        self.buff = None  # gpd polygon
        self.max_distance = None

    def sub_hour_count(self, date, hour, minutes):
        self.minutes = minutes
        self.data_subset = self.data[self.data[NAME_DATE] == int(date)]
        self.data_subset = self.data_subset[self.data_subset[NAME_HOUR] == hour]
        amount = minutes / 60
        tr1, tr2 = NAME_TR1, NAME_TR2
        self.counts = float(self.data_subset[tr1]) * amount + float(self.data_subset[tr2]) * amount
        self.cars = float(self.data_subset["Pkw_R1"]) * amount + float(self.data_subset["Pkw_R2"]) * amount

    def max_traveled_dist(self, minutes, speed):
        self.max_distance = speed / (60 / minutes)  # km

    def buffer(self, minutes, speed):
        self.max_traveled_dist(minutes, speed)
        self.buff = gpd.GeoDataFrame(geometry=[self.station_xy.buffer(kilometers2degrees(self.max_distance))])
        self.buff.crs = "EPSG:4326"
