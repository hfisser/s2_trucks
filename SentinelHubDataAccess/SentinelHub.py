import os
from sentinelhub import SHConfig, MimeType, CRS, BBox, SentinelHubRequest, SentinelHubDownloadClient, \
    DataCollection, bbox_to_dimensions, DownloadRequest
from SentinelHubDataAccess.RequestBuilder import RequestBuilder

creds_file = os.path.join("F:" + os.sep + "sh", "sh.txt")
aoi = (9.56, 52, 9.56 + 0.014, 52 + 0.009)  # xmin, ymin
#aoi = (9.56, 52, 12.73, 52 + 0.009)
time_period = ("2019-07-03", "2019-07-05")
band_names = ["B02", "B03"]  #, "B04", "B08"]
dataset_name = DataCollection.SENTINEL2_L2A
spatial_resolution = 10

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

    def get_data(self, bbox, period, dataset, bands, resolution):
        sh_request_builder = RequestBuilder(bbox, period, dataset, bands, resolution)
        request = sh_request_builder.request(self.sh_config)
        data = sh_request_builder.get_data(request)
        print("Retrieved data of shape: %s" % str(data[0].shape))
        return data


if __name__ == "__main__":
    sh = SentinelHub()
    sh.set_credentials(creds_file)
    band_data = sh.get_data(aoi, time_period, dataset_name, band_names, spatial_resolution)


