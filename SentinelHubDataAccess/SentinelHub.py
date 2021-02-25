import os
import utm
import time
import shutil
import numpy as np
import rasterio as rio
from glob import glob
from rasterio.merge import merge
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

    def get_data(self, bbox, period, dataset, bands, resolution, dir_save=None, merged_file=None, mosaicking_order=None):
        """
        :param bbox: array-like with four coordinates xmin, ymin, xmax, ymax
        :param period: array-like with two dates
        :param dataset: DataCollection instance
        :param bands: array-like with bands str
        :param resolution: float specifies the resolution
        :param dir_save: str directory
        :param merged_file: str file path of merged file
        :param mosaicking_order: str mosaicking order, e.g. "leastCC"
        :return:
        """
        if len(period[0].split("-")[0]) == 2:
            period = ["-".join([date[-4:], date[3:5], date[0:2]]) for date in period]
        if merged_file is not None and os.path.exists(merged_file):
            with rio.open(merged_file, "r") as src:
                data = np.zeros((src.count, src.height, src.width))
                for band_idx in range(data.shape[0]):
                    data[band_idx] = src.read(band_idx + 1)
            return data, os.path.dirname(merged_file)
        else:
            splitted_box = self.split_box(bbox, resolution)
            if len(splitted_box) > 1:
                return self.get_data_large_aoi(splitted_box, period, dataset, bands, resolution, dir_save, merged_file,
                                               mosaicking_order)
            else:
                sh_request_builder = RequestBuilder(bbox, period, dataset, bands, resolution, mosaicking_order)
                request = sh_request_builder.request(self.sh_config, dir_save)
                data = request.get_data(save_data=dir_save is not None)[0]
                d = request.data_folder
                # change dtype
                folder = list(set(glob(os.path.join(d, "*"))) - set(glob(os.path.join(d, "*.tiff"))))[0]
                self.change_tiff_dtype(glob(os.path.join(folder, "*.tiff"))[0], np.float32)
                print("Retrieved data of shape: %s" % str(data.shape))
                return data, d

    def get_data_large_aoi(self, splitted_box, period, dataset, bands, resolution, dir_save, merged_file,
                           mosaicking_order):
        dir_save_tmp = os.path.join(dir_save, "tmp")
        files = []
        for i, bbox in enumerate(splitted_box):
            if not os.path.exists(dir_save_tmp):
                os.mkdir(dir_save_tmp)
            curr_s2_data_file = os.path.join(dir_save_tmp, "s2_date%s_%s_bbox%s.tiff" % (period[0], period[1], i))
            files.append(curr_s2_data_file)
            if os.path.exists(merged_file):
                with rio.open(merged_file, "r") as src:
                    merged_stack = np.zeros((src.count, src.height, src.width))
                    for band_idx in range(src.count):
                        merged_stack[band_idx] = src.read(band_idx + 1)
                return merged_file, merged_stack
            elif os.path.exists(curr_s2_data_file):
                pass
            else:
                band_stack, dir_out = self.get_data(bbox, period, dataset, bands, resolution, dir_save_tmp, None,
                                                    mosaicking_order)
                folders = list(set(glob(os.path.join(dir_out, "*"))) - set(glob(os.path.join(dir_out, "*.tiff"))))
                if len(folders) > 1:
                    print("Several files, don't know which to read from %s" % dir_save)
                    raise FileNotFoundError
                else:
                    folder = folders[0]
                    copied = shutil.copyfile(glob(os.path.join(folder, "*.tiff"))[0], curr_s2_data_file)
                    if os.path.exists(folder) and os.path.exists(curr_s2_data_file):
                        shutil.rmtree(folder)  # remove original download file
                    else:
                        time.sleep(10)
                        shutil.rmtree(folder)
        with rio.open(files[0], "r") as src:
            meta = src.meta  # get meta
        merged_stack, transform = merge(files)
        if np.count_nonzero(merged_stack[0]) == 0:
            for f in files:
                os.remove(f)
            return merged_stack, os.path.dirname(merged_file)
        meta = dict(transform=transform, height=merged_stack.shape[1], width=merged_stack.shape[2],
                    count=merged_stack.shape[0], driver="GTiff", dtype=merged_stack.dtype,
                    crs=meta["crs"])
        with rio.open(merged_file, "w", **meta) as tgt:
            for i in range(merged_stack.shape[0]):
                tgt.write(merged_stack[i], i + 1)
        if os.path.exists(merged_file):
            for file in files:  # delete the sub aoi files
                os.remove(file)
            shutil.rmtree(dir_save_tmp)
        return merged_stack, os.path.dirname(merged_file)

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
        # split into several boxes
        hemisphere = "N" if bbox_epsg4326[3] >= 0 else "S"
        bbox_utm = Polygon(box(upper_left_utm[0], lower_right_utm[1], lower_right_utm[0], upper_left_utm[1]))
        crs = CRS["UTM_" + str(upper_left_utm[2]) + hemisphere]  # use UTM
        bbox_splitter = BBoxSplitter([bbox_utm], crs, (int(size_x / 2400) + 1, int(size_y / 2400) + 1),
                                     reduce_bbox_sizes=True)
        return bbox_splitter.get_bbox_list()

    @staticmethod
    def change_tiff_dtype(file, dtype):
        with rio.open(file, "r") as src:
            meta, data = src.meta, np.zeros((src.count, src.height, src.width), dtype=dtype)
            for band_idx in range(data.shape[0]):
                data[band_idx] = src.read(band_idx + 1).astype(dtype)
        meta["dtype"] = data.dtype
        with rio.open(file, "w", **meta) as tgt:
            for band_idx in range(meta["count"]):
                tgt.write(data[band_idx], band_idx + 1)

    def data_available(self, request_kwargs):
        kwargs = request_kwargs.copy()
        bbox_copy = kwargs["bbox"].copy()
        lon_extent, lat_extent = np.abs(bbox_copy[0] - bbox_copy[2]), np.abs(bbox_copy[1] - bbox_copy[3])
        # strongly crop bbox to very small aoi in order to only get test pixels
        bbox_copy[0] += lon_extent * 0.49
        bbox_copy[1] += lat_extent * 0.4999
        bbox_copy[2] -= lon_extent * 0.49
        bbox_copy[3] -= lat_extent * 0.4999
        kwargs["bbox"] = self.split_box(bbox_copy, kwargs["resolution"])[0]
        kwargs["bands"] = ["B02"]
        kwargs["merged_file"] = os.path.join(os.path.dirname(kwargs["merged_file"]), "test.tiff")
        test_data, folder = self.get_data(**kwargs)
        try:
            shutil.rmtree(glob(os.path.join(folder, "*"))[0])
        except IndexError:
            pass
        return np.count_nonzero(test_data) > 0


if __name__ == "__main__":
    creds_file = os.path.join("F:" + os.sep + "sh", "sh.txt")
    aoi = (9.56, 52, 9.56 + 0.014, 52 + 0.009)  # xmin, ymin
    # aoi = (9.56, 52, 12.73, 52 + 0.009)
    time_period = ("2019-07-04", "2019-07-04")
    band_names = ["B08", "B04", "B03", "B02", "CLM"]
    dataset_name = DataCollection.SENTINEL2_L2A
    spatial_resolution = 10
    dir_write = "F:\\Masterarbeit\\DLR\\project\\1_truck_detection\\data\\s2\\sentinel_hub"
    sh = SentinelHub()
    sh.set_credentials(creds_file)
    band_data, dir_data = sh.get_data(aoi, time_period, dataset_name, band_names, spatial_resolution,
                                      dir_write)
