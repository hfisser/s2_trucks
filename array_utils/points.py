import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr

from rasterio import features
from array_utils.geocoding import transform_lat_lon


# extracts coordinates at value in np array and returns points as GeoDataFrame
# data 2d np array
# match_value Float value in data where point coordinates are extracted
# lon_lat dict of:
# "lon": np array longitude values"
# "lat": np array latitude values"
# crs String EPSG:XXXX
def points_from_np(data, match_value, lon_lat, crs):
    indices = np.argwhere(data == match_value)
    if len(indices) > 0:
        lat_indices = indices[:, [0]]
        lon_indices = indices[:, [1]]
        lat_coords = lon_lat["lat"][lat_indices]
        lon_coords = lon_lat["lon"][lon_indices]
        points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon_coords, lat_coords))
        points.crs = crs
        return points


def raster_to_points(raster, lon_lat, field_name, crs):
    points_list = []
    match_values = np.unique(raster[(raster != 0) * ~np.isnan(raster)])  # by pixel value
    for x in match_values:
        points = points_from_np(raster, x, lon_lat, crs=crs)
        points[field_name] = [x] * len(points)
        points_list.append(points)
    return gpd.GeoDataFrame(pd.concat(points_list, ignore_index=True))


def rasterize(polygons, lat, lon, fill=np.nan):
    transform = transform_lat_lon(lat, lon)
    out_shape = (len(lat), len(lon))
    raster = features.rasterize(polygons.geometry, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float)
    return xr.DataArray(raster, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
