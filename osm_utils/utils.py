import os
import geopandas as gpd
import pandas as pd
import numpy as np

from array_utils.points import rasterize
from pkg_utils.package_utils import pip_install
try:
    import OSMPythonTools
except ModuleNotFoundError:
    pip_install("OSMPythonTools")
from OSMPythonTools.overpass import overpassQueryBuilder, Overpass


def buffer_bbox(bbox_osm):
    offset = 0.05  # add a buffer to bbox in order to be sure cube is entirely covered
    bbox_osm[0] -= offset  # min lat
    bbox_osm[1] -= offset  # min lon
    bbox_osm[2] += offset  # max lat
    bbox_osm[3] += offset  # max lon
    return bbox_osm


# bbox List of four coords
# bbox_id Integer processing id of bbox
# osm_value String OSM value
# osm_key String OSM key
# element_type List of String
# returns GeoPandasDataFrame
def get_osm(bbox,
            osm_value="motorway",
            osm_key="highway"):
    element_type = ["way", "relation"]
    bbox_osm = buffer_bbox(bbox)
    quot = '"'
    select = quot + osm_key + quot + '=' + quot + osm_value + quot
    select_link = select.replace(osm_value, osm_value + "_link")  # also get road links
    select_junction = select.replace(osm_value, osm_value + "_junction")
    geoms = []
    for selector in [select, select_link, select_junction]:
        query = overpassQueryBuilder(bbox=bbox_osm,
                                     elementType=element_type,
                                     selector=selector,
                                     out='body',
                                     includeGeometry=True)
        try:
            elements = Overpass().query(query, timeout=120).elements()
        except Exception:
            elements = []
            Warning("Could not download OSM data")
        # create multiline of all elements
        if len(elements) > 0:
            for i in range(len(elements)):
                elem = elements[i]
                try:
                    geoms.append(elem.geometry())
                except Exception:
                    continue
        Warning("Could not retrieve " + select)
    if len(geoms) > 0:
        lines = gpd.GeoDataFrame(crs="EPSG:4326", geometry=geoms)
        n = len(geoms)
        lines["osm_value"] = [osm_value] * n  # add road type
        return lines
    else:
        return []


# buffer Float road buffer distance [m]
# bbox List of four coords
# osm_values List of String OSM values
# roads_buffer Float buffer width
# dir_write
def get_roads(bbox, osm_values, buffer_meters, dir_write, filename, crs):
    fwrite = os.path.join(dir_write, filename + ".gpkg")
    file_tmp = os.path.join(dir_write, "tmp.gpkg")
    if not os.path.exists(fwrite):
        roads = []
        offset = 7.5  # meters
        buffer_dist = "buffer_distance"
        # buffer according to road type
        m, t, p, s, ter = "motorway", "trunk", "primary", "secondary", "tertiary"
        buffers = {m: buffer_meters, t: buffer_meters - offset, p: buffer_meters - (2 * offset),
                   s: buffer_meters - (3 * offset), ter: buffer_meters - (4 * offset)}
        osm_values_int = {m: 1, t: 2, p: 3, s: 4, ter: 5}
        for osm_value in osm_values:
            roads_osm = get_osm(bbox=bbox, osm_value=osm_value)
            if len(roads_osm) > 0:
                roads_osm.to_file(file_tmp, driver="GPKG")
                roads_osm = gpd.read_file(file_tmp)
                roads_osm = roads_osm.to_crs(crs)
                roads_osm[buffer_dist] = [buffers[osm_value]] * len(roads_osm)
                roads_osm["osm_value_int"] = osm_values_int[osm_value]
                roads.append(roads_osm)
        try:
            roads_merge = gpd.GeoDataFrame(pd.concat(roads, ignore_index=True), crs=roads[0].crs)
        except ValueError:
            Warning("No objects")
        else:
            buffered = roads_merge.buffer(distance=roads_merge[buffer_dist])
            roads_merge.geometry = buffered
            roads_merge.to_file(fwrite, driver="GPKG")
            if os.path.exists(file_tmp):
                os.remove(file_tmp)
    return fwrite


# osm geodataframe of polygons
# reference_raster xarray with lat and lon
def rasterize_osm(osm, reference_raster):
    osm_values = list(set(osm["osm_value"]))
    nan_placeholder = 100
    road_rasters = []
    for osm_value in osm_values:
        osm_subset = osm[osm["osm_value"] == osm_value]
        raster = rasterize(osm_subset, reference_raster.lat, reference_raster.lon)
        cond = np.isfinite(raster)
        raster_osm = np.where(cond, list(osm_subset.osm_value_int)[0],
                              nan_placeholder)  # use placeholder instead of nan first
        raster_osm = raster_osm.astype(np.float)
        road_rasters.append(raster_osm)
        # merge road types in one layer
    road_raster_np = np.array(road_rasters).min(axis=0)  # now use the lowest value (highest road level) because some intersect
    road_raster_np[road_raster_np == nan_placeholder] = 0
    return road_raster_np  # 0=no_road 1=motorway, 2=trunk, ...


def osm_values_to_name(values):
    mapping = {1: "Motorway", 2: "Trunk", 3: "Primary", 4: "Secondary", 5: "Tertiary"}
    return [mapping[value] for value in values]
