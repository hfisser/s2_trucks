import os
import rasterio
import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
from shapely import geometry
from pyproj import Transformer
from rasterio.transform import Affine
from array_utils.math import rescale
from osm_utils.utils import get_roads, rasterize_osm

dir_main = "F:\\Masterarbeit\\DLR\\project\\1_truck_detection\\data"
dir_osm_roads = os.path.join(dir_main, "osm")
dir_s2 = os.path.join(dir_main, "s2", "raw")
s2_directories = [os.path.join(dir_s2, x) for x in os.listdir(dir_s2)]
dir_write = os.path.join(os.path.dirname(dir_s2), "subsets")
number_subsets = 1
roads_buffer = 40
tiles_pd = pd.read_csv(os.path.join(dir_main, "training", "tiles.csv"))
training_tiles = list(tiles_pd["training_tiles"])  # Germany, Italy, USA, France, Russia, South Africa, India
calibration_tiles = list(tiles_pd["calibration_tiles"])  # Netherlands, Alps, USA, Spain, Ukraine, Kenya, China
tiles = training_tiles + calibration_tiles


def subset(d, n_subs, osm_buffer, dir_osm, dir_out):
    tgt_crs = "EPSG:4326"
    home = os.path.dirname(os.path.dirname(d))
    for this_dir in [dir_out, dir_osm]:
        if not os.path.exists(this_dir):
            os.mkdir(this_dir)
    d1 = os.path.join(d, os.listdir(d)[0], "GRANULE")
    if os.path.exists(d1):
        d2 = os.path.join(os.path.join(d1, os.listdir(d1)[0]), "IMG_DATA", "R10m")
        files = os.listdir(d2)
        band_names = np.array([f.split("_")[2] for f in files])
        bands = ["B04", "B03", "B02", "B08"]
        file_stack = os.path.join(dir_out, os.path.basename(d) + ".tif")
        src = rasterio.open(os.path.join(d2, files[0]))
        kwargs = src.meta.copy()
        src_crs = src.crs
        kwargs.update({"count": len(bands), "dtype": np.uint16, "driver": "GTiff"})
        if not os.path.exists(file_stack):
            with rasterio.open(file_stack, "w", **kwargs) as tgt:
                for i, b in enumerate(bands):
                    fname = files[np.where(band_names == b)[0][0]]
                    with rasterio.open(os.path.join(d2, fname)) as src:
                        tgt.write(src.read(1).astype(np.uint16), i+1)
        a, b = src.transform * [0,0], src.transform * [src.height, src.width]
        src_corners = np.array([a[0], b[1], b[0], a[1]]).flatten()
        transformer = Transformer.from_crs(str(src_crs), tgt_crs)
        y0, x0 = transformer.transform(src_corners[0], src_corners[3])
        y1, x1 = transformer.transform(src_corners[2], src_corners[1])
        bbox_epsg4326 = [y1, x0, y0, x1]
        fname = os.path.basename(d) + "_.tif"
        fname_pure = fname.split(".")[0][0:-1]
        gpkg_out = os.path.join(dir_out, fname_pure + ".gpkg")
        gdf = gpd.GeoDataFrame({"index": [1]},
                               geometry=[geometry.box(src_corners[0], src_corners[1], src_corners[2], src_corners[3])],
                               crs=str(src_crs))
        gdf.to_file(gpkg_out, driver="GPKG")
        # get OSM data and mask data to roads
        osm_file = get_roads(bbox_epsg4326, ["motorway", "trunk", "primary"], osm_buffer,
                             dir_osm, fname_pure + "osm_roads", str(src_crs))
        osm_vec = gpd.read_file(osm_file)
        n = n_subs / 2
        n = 1 if n < 1 else n
        stack = rasterio.open(file_stack, "r")
        n_bands = len(bands)
        h, w = stack.height, stack.width
        data = np.zeros((n_bands, h, w), dtype=np.float32)
        for i in range(n_bands):
            data[i] = stack.read(i+1)
        tgt_pixels_y, tgt_pixels_x = int(h/n), int(w/n)
        src_lat = get_lat(src_corners, h)
        src_lon = get_lon(src_corners, w)
        dim = range(int(n))
        for y in dim:
            for x in dim:
                y1, y2 = int(y*tgt_pixels_y), int((y+1)*tgt_pixels_y)
                x1, x2 = int(x*tgt_pixels_x), int((x+1)*tgt_pixels_x)
                file_out = os.path.join(dir_out, "_".join([fname_pure, "y"+str(y1), "x"+str(x1)])+".tif")
                if os.path.exists(file_out):
                    print("Already exists: %s" % file_out)
                else:
                    tgt_lat = src_lat[y1:y2]
                    tgt_lon = src_lon[x1:x2]
                    ref_xr = xr.DataArray(data=np.zeros((len(tgt_lat), len(tgt_lon))),
                                          coords={"lat": tgt_lat, "lon": tgt_lon},
                                          dims=["lat", "lon"])
                    osm_raster = rasterize_osm(osm_vec, ref_xr)
                    osm_raster[osm_raster != 0] = 1
                    osm_raster[osm_raster == 0] = np.nan
                    t = stack.transform
                    transform = Affine(t[0], t[1], tgt_lon[0], t[3], t[4], tgt_lat[0])
                    kwargs.update({"transform": transform,
                                   "height": tgt_pixels_y, "width": tgt_pixels_x,
                                   "dtype": np.float32})
                    with rasterio.open(file_out, "w", **kwargs) as tgt:
                        for i in range(n_bands):
                            data_band = rescale(data[i, y1:y2, x1:x2].astype(np.float32), 0., 1.)
                            tgt.write((data_band * osm_raster).astype(np.float32), i+1)


def get_lat(bbox, h, step=None):
    if step is None:
        step = (bbox[3] - bbox[1]) / h
    return np.arange(bbox[3], bbox[1], -step)


def get_lon(bbox, w, step=None):
    if step is None:
        step = (bbox[2] - bbox[0]) / w
    return np.arange(bbox[0], bbox[2], step)


def transform_to_bbox(kwargs):
    # W,S,E,N
    this_transform = kwargs["transform"]
    upper_y_deg, lower_x_deg = this_transform[5], this_transform[2]
    res_deg = this_transform[0]
    x_deg = kwargs["width"] * res_deg
    y_deg = kwargs["height"] * res_deg
    return [lower_x_deg, upper_y_deg - y_deg, lower_x_deg + x_deg, upper_y_deg]


if __name__ == "__main__":
    if not os.path.exists(dir_write):
        os.mkdir(dir_write)
    if not os.path.exists(dir_osm_roads):
        os.mkdir(dir_osm_roads)
    for directory in s2_directories:
        try:
            tile = os.path.basename(directory).split("_")[5]
        except IndexError:
            continue
        if tile in tiles:
            print("Processing: " + os.path.basename(directory))
            subset(directory, number_subsets, roads_buffer, dir_osm_roads, dir_write)
