import numpy as np
from pyproj import Transformer
from rasterio.transform import Affine


def lat_from_meta(meta):
    try:
        t, h = meta["transform"], meta["height"]
    except KeyError as e:
        raise e
    return np.arange(t[5], t[5] + (t[4] * h), t[4])


def lon_from_meta(meta):
    try:
        t, w = meta["transform"], meta["width"]
    except KeyError as e:
        raise e
    return np.arange(t[2], t[2] + (t[0] * w), t[0])


# lat, lon
def transform_lat_lon(lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale


def metadata_to_bbox_epsg4326(metadata):
    t, src_crs = metadata["transform"], metadata["crs"]
    tgt_crs = "EPSG:4326"
    a, b = t * [0, 0], t * [metadata["height"], metadata["width"]]
    src_corners = np.array([a[0], b[1], b[0], a[1]]).flatten()
    transformer = Transformer.from_crs(str(src_crs), tgt_crs)
    y0, x0 = transformer.transform(src_corners[0], src_corners[3])
    y1, x1 = transformer.transform(src_corners[2], src_corners[1])
    bbox_epsg4326 = [y1, x0, y0, x1]
    return bbox_epsg4326
