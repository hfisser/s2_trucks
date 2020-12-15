import numpy as np
from pyproj import Transformer
from rasterio.transform import Affine


def lat_from_meta(meta):
    try:
        t, h = meta["transform"], meta["height"]
    except KeyError as e:
        raise e
    lat = np.arange(t[5], t[5] + (t[4] * h), t[4])
    # in rare cases coords may be too short or too long (e.g. due to rounding)
    lat = shorten_coords_array(lat, t[5], t[4], h)  # try several times in order to be sure
    lat = enlarge_coords_array(lat, t[5], t[4], h)
    lat = shorten_coords_array(lat, t[5], t[4], h)
    lat = enlarge_coords_array(lat, t[5], t[4], h)
    lat = shorten_coords_array(lat, t[5], t[4], h)
    lat = enlarge_coords_array(lat, t[5], t[4], h)
    return lat


def lon_from_meta(meta):
    try:
        t, w = meta["transform"], meta["width"]
    except KeyError as e:
        raise e
    lon = np.arange(t[2], t[2] + (t[0] * w), t[0])
    # in rare cases coords may be too short or too long (e.g. due to rounding)
    lon = shorten_coords_array(lon, t[2], t[0], w)  # try several times in order to be sure
    lon = enlarge_coords_array(lon, t[2], t[0], w)
    lon = shorten_coords_array(lon, t[2], t[0], w)
    lon = enlarge_coords_array(lon, t[2], t[0], w)
    lon = shorten_coords_array(lon, t[2], t[0], w)
    lon = enlarge_coords_array(lon, t[2], t[0], w)
    return lon


def shorten_coords_array(coords, min_coord, resolution, length_should_have):
    # may occur (rarely) due to rounding e.g. that coords have not correct length
    digits = 15
    while len(coords) > length_should_have and digits > 0:  # lon too long
        digits -= 1
        # round resolution
        coords = np.arange(min_coord, min_coord + (resolution * length_should_have), round(resolution, digits))
    return coords


def enlarge_coords_array(coords, min_coord, resolution, length_should_have):
    # may occur (rarely) due to rounding e.g. that coords have not correct length
    offset = 5e-10
    offset_initial = 5e-10
    while len(coords) < length_should_have:  # lon too long
        offset += offset_initial
        # decrease step
        coords = np.arange(min_coord, min_coord + (resolution * length_should_have), resolution - offset)
    return coords


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
    x0, y0 = transformer.transform(src_corners[0], src_corners[3])
    x1, y1 = transformer.transform(src_corners[2], src_corners[1])
    bbox_epsg4326 = [y1, x0, y0, x1]
    return bbox_epsg4326
