
import os
import numpy as np
import geopandas as gpd
import requests

from osm.utils import get_roads

stations = ["Immensen (3489)", "Theeßen (3810)", "Alleringersleben (3837)", "Peine (3306)"]

BAST_URL = "https://www.bast.de/BASt_2017/DE/Verkehrstechnik/Fachthemen/v2-verkehrszaehlung/Aktuell/" \
           "zaehl_aktuell_node.html;jsessionid=63610843F87B77C24C4320BC4EAD6647.live21304"

OSM_BUFFER = 30

NAME_DATE = "Datum"
NAME_HOUR = "Stunde"
NAME_TR1 = "Lkw_R1"
NAME_TR2 = "Lkw_R2"

hour = 10
minutes = 20

from xcube.util.assertions import assert_given

class Validator:
    def __init__(self, truck_points_path, true_counts_path, station_xy):
        self.eo_counts = gpd.read_file(truck_points_path)  # gpd points
        self.truck_points_path = truck_points_path
        self.tc = TrueCounts(true_counts_path, station_xy)
        self.osm_mask = None
        self.bbox_id = int(os.path.basename(truck_points_path).split("_")[-1].split(".")[0][2:])
        self.eo_counts.crs = "EPSG:4326"
        self.eo_counts_subset = None
        self.truck_dist = None
        self.within_dist = None
        self.eo_vs_truth = None

    def subset_to_buffer(self, buffer):
        self.eo_counts_subset = gpd.sjoin(self.eo_counts, buffer, op="within")

    def within_distance(self, osm_mask, station, km_max_distance):
        self.truck_dist = TruckDistance(osm_mask)
        self.within_dist = 0
        for point in self.eo_counts_subset.geometry:
            self.truck_dist.calc_travel_dist(point, station)
            dist = self.truck_dist.dist
            if dist is not None and dist <= km_max_distance:
                self.within_dist += 1

    def validate(self, date, hour, minutes, speed, grid_gadm, dir_not_commit):
        self.tc.sub_hour_count(date[2:4] + date[5:7] + date[8:], hour, minutes)
        self.tc.buffer(minutes, speed)
        print(self.bbox_id)
        osm = gpd.read_file(
            os.path.join(dir_not_commit, "ancillary_data", "roads", str(self.bbox_id) + "_" + "highway.gpkg"))
        self.osm_mask = get_osm_raster(osm, grid_gadm, date, self.truck_points_path)
        osm_in_buffer = gpd.overlay(self.tc.buff, osm)
        # mask osm raster to buffered osm
        bounds = osm_in_buffer.total_bounds
        lat, lon = self.osm_mask.lat.values, self.osm_mask.lon.values
        lat_bounds = (lat >= bounds[1]) * (lat <= bounds[3])
        lon_bounds = (lon >= bounds[0]) * (lon <= bounds[2])
        mesh = np.meshgrid(lon_bounds, lat_bounds)
        mask = mesh[0] * mesh[1]
        self.osm_mask = self.osm_mask.where(mask)
        self.osm_mask.roadmask.values[np.isnan(self.osm_mask.roadmask.values)] = 0.
        self.subset_to_buffer(self.tc.buff)
        self.within_distance(self.osm_mask, self.tc.station_xy, self.tc.max_distance)
        eo_count = self.within_dist / 2  # divide by two in order to include only lanes where trucks are coming from the station
        self.eo_vs_truth = {"date": date, "eo_count": eo_count, "true_count": self.tc.counts,
                            "percentage": (eo_count / self.tc.counts) * 100, "cars": self.tc.cars}





    @staticmethod
    def get_osm_mask(self, bbox, metadata, dir_out):
        osm_file = get_roads(bbox, ["motorway", "trunk", "primary"], OSM_BUFFER,
                             dir_out, str(bbox).replace(", ", "_")[1:-1] + "_osm_roads", str(metadata["crs"]))
        osm_vec = gpd.read_file(osm_file)
        ref_xr = xr.DataArray(data=self.trucks_np, coords={"lat": self.lat, "lon": self.lon}, dims=["lat", "lon"])
        osm_raster = rasterize_osm(osm_vec, ref_xr).astype(np.float32)
        osm_raster[osm_raster != 0] = 1
        osm_raster[osm_raster == 0] = np.nan
        return osm_raster


    @staticmethod
    def get_bbox_id(point, grid_gadm):
        point_in_box = []
        for i, geom in enumerate(grid_gadm.geometry):
            if geom.contains(point):
                point_in_box.append(grid_gadm.bbox_id[i])
        return point_in_box

    @staticmethod
    def get_station_utm(station_name):
        """
        gets UTM coordinates of BAST traffic count station from BAST webpage
        :param station_name: str in the format "Theeßen (3810)"
        :return: dict x and y UTM coordinate
        """
        response = requests.get(BAST_URL)
        page_str = response.text
        stations = page_str.split("addBKGPoempel")[2:]
        for station_row in stations:
            try:
                name = station_row.split(": ")[1].split(",")[0]
            except IndexError:
                continue
            if name == station_name:
                try:
                    coordinates = station_row.split("results, ")[1].split(",")
                    x, y = coordinates[0].split('"')[1], coordinates[1]
                except IndexError:
                    continue
                try:
                    return {"x": float(x), "y": float(y)}
                except ValueError:
                    continue
        print("No station name match")
