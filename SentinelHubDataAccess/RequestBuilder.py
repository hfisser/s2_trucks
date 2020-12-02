from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, bbox_to_dimensions, DataCollection


class RequestBuilder:
    def __init__(self, bbox_epsg84, time_period, dataset, bands, resolution):
        self.bbox, self.bbox_size, self.eval_script = None, None, None
        self.dataset = dataset
        self.time_period = time_period
        self.bands = bands
        self.resolution = resolution
        self.setup_bbox(bbox_epsg84)
        self.setup_eval_script()

    def setup_bbox(self, bbox_wgs84):
        self.bbox = BBox(bbox=bbox_wgs84, crs=CRS.WGS84)
        self.bbox_size = bbox_to_dimensions(self.bbox, resolution=self.resolution)

    def setup_eval_script(self):
        eval_script = """
            //VERSION=3

            function setup() {
                return {
                    input: [{
                        bands: band_list
                    }],
                    output: {
                        bands: n_bands
                    }
                };
            }

            function evaluatePixel(sample) {
                return sample_list;
            }
        """
        sample_list = ["sample." + band for band in self.bands]
        eval_script = eval_script.replace("band_list", str(self.bands))
        eval_script = eval_script.replace("n_bands", str(len(self.bands)))
        eval_script = eval_script.replace("'", '"')
        eval_script = eval_script.replace("sample_list", str(sample_list))
        self.eval_script = eval_script.replace("'", "")

    def request(self, config):
        request = SentinelHubRequest(
            evalscript=self.eval_script,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=self.dataset,
                    time_interval=self.time_period,
                )
            ],
            responses=[
                SentinelHubRequest.output_response("default", MimeType.TIFF)
            ],
            bbox=self.bbox,
            size=self.bbox_size,
            config=config
        )
        return request

    @staticmethod
    def get_data(request):
        return request.get_data()
