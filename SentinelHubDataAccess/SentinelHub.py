import os
from sentinelhub import SHConfig, DataCollection
from SentinelHubDataAccess.RequestBuilder import RequestBuilder

creds_file = os.path.join("F:" + os.sep + "sh", "sh.txt")
aoi = (9.56, 52, 9.56 + 0.014, 52 + 0.009)  # xmin, ymin
#aoi = (9.56, 52, 12.73, 52 + 0.009)
time_period = ("2019-07-03", "2019-07-05")
band_names = ["B08", "B04", "B03", "B02", "CLM"]
dataset_name = DataCollection.SENTINEL2_L2A
spatial_resolution = 10
dir_out = "F:\\Masterarbeit\\DLR\\project\\1_truck_detection\\data\\s2\\sentinel_hub"


#spatial_resolution = 60
#aoi = [46.16, -16.15, 46.51, -15.58]


class SentinelHub:
    def __init__(self, ):
        self.sh_config = SHConfig()

    def set_credentials(self, sh_credentials_file):
        with open(sh_credentials_file, "r") as f:
            content = f.read()
            self.sh_config.sh_client_id = content.split("SH_CLIENT_ID = ")[1].split("\n")[0]
            self.sh_config.sh_client_secret = content.split("SH_CLIENT_SECRET = ")[1].split("\n")[0]

    def get_data(self, bbox, period, dataset, bands, resolution, dir_save=None):
        sh_request_builder = RequestBuilder(bbox, period, dataset, bands, resolution)
        request = sh_request_builder.request(self.sh_config, save_dir)
        data = request.get_data(dir_save)
        print("Retrieved data of shape: %s" % str(data[0].shape))
        return data[0], request.data_folder


if __name__ == "__main__":
    sh = SentinelHub()
    sh.set_credentials(creds_file)
    band_data, save_dir = sh.get_data(aoi, time_period, dataset_name, band_names, spatial_resolution,
                                      dir_out)
