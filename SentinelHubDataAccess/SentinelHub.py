import os
import utm
import numpy as np
from shapely.geometry import Polygon, box
from sentinelhub import SHConfig, DataCollection, CRS, BBoxSplitter
from SentinelHubDataAccess.RequestBuilder import RequestBuilder


class SentinelHub:
    def __init__(self):
        self.sh_config = SHConfig()

    def set_credentials(self, sh_credentials_file):
        """
        set the credentials in Sentinel Hub configuration
        :param sh_credentials_file: str file path to user credentials
        :return: tuple of numpy ndarray and str output folder
        """
        with open(sh_credentials_file, "r") as f:
            content = f.read()
            self.sh_config.sh_client_id = content.split("SH_CLIENT_ID = ")[1].split("\n")[0]
            self.sh_config.sh_client_secret = content.split("SH_CLIENT_SECRET = ")[1].split("\n")[0]

    def get_data(self, bbox, period, dataset, bands, resolution, dir_save=None):
        # try if split returns several boxes, then it must be split into several boxes and called separately
        if len(self.split_box(bbox, resolution)) > 1:
            raise ValueError("bbox is larger than 2500 pixels on at least one axis. Call split_box() and retrieve"
                             "data separately")
        sh_request_builder = RequestBuilder(bbox, period, dataset, bands, resolution)
        request = sh_request_builder.request(self.sh_config, dir_save)
        data = request.get_data(save_data=dir_save is not None)
        print("Retrieved data of shape: %s" % str(data[0].shape))
        return data[0], request.data_folder

    @staticmethod
    def split_box(bbox_epsg4326, res):
        """
        splits WGS84 box coordinates into as many boxes as needed for Sentinel Hub
        :param bbox_epsg4326: list or tuple of length 4 with xmin, ymin, xmax, ymax
        :param res: float spatial resolution
        :return: list of Bbox instances
        """
        try:
            upper_left_utm = utm.from_latlon(bbox_epsg4326[3], bbox_epsg4326[0])
        except TypeError:  # is already of class Bbox
            return [bbox_epsg4326]
        lower_right_utm = utm.from_latlon(bbox_epsg4326[1], bbox_epsg4326[2], upper_left_utm[2])  # same zone number
        size_y = abs(int(np.round((upper_left_utm[1] - lower_right_utm[1]) / res, 2)))
        size_x = abs(int(np.round((lower_right_utm[0] - upper_left_utm[0]) / res, 2)))
        if size_y < 2500 and size_x < 2500:
            return [bbox_epsg4326]
        else:
            # split into several boxes
            hemisphere = "N" if bbox_epsg4326[3] >= 0 else "S"
            bbox_utm = Polygon(box(upper_left_utm[0], lower_right_utm[1], lower_right_utm[0], upper_left_utm[1]))
            crs = CRS["UTM_" + str(upper_left_utm[2]) + hemisphere]  # use UTM
            bbox_splitter = BBoxSplitter([bbox_utm], crs, (int(size_x / 2499) + 1, int(size_y / 2499) + 1),
                                         reduce_bbox_sizes=True)
            return bbox_splitter.get_bbox_list()


if __name__ == "__main__":
    creds_file = os.path.join("F:" + os.sep + "sh", "sh.txt")
    aoi = (9.56, 52, 9.56 + 0.014, 52 + 0.009)  # xmin, ymin
    # aoi = (9.56, 52, 12.73, 52 + 0.009)
    time_period = ("2019-07-04", "2019-07-04")
    band_names = ["B08", "B04", "B03", "B02", "CLM"]
    dataset_name = DataCollection.SENTINEL2_L2A
    spatial_resolution = 10
    dir_out = "F:\\Masterarbeit\\DLR\\project\\1_truck_detection\\data\\s2\\sentinel_hub"
    sh = SentinelHub()
    sh.set_credentials(creds_file)
    band_data, dir_data = sh.get_data(aoi, time_period, dataset_name, band_names, spatial_resolution,
                                      dir_out)
