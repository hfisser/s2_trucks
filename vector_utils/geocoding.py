from pyproj import Transformer
from shapely.geometry import Point


def utm_to_4326(utm_point, src_crs, tgt_crs):
    transformer = Transformer.from_crs(src_crs, tgt_crs)
    transformed = transformer.transform(utm_point[1], utm_point[0])
    return Point(transformed[1], transformed[0])
