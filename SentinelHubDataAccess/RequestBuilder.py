import utm
from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, bbox_to_dimensions


class RequestBuilder:
    def __init__(self, bbox, time_period, dataset, bands, resolution):
        self.bbox, self.bbox_size, self.eval_script = None, None, None
        self.dataset = dataset
        self.time_period = time_period
        self.bands = bands
        self.resolution = resolution
        self.setup_bbox(bbox)
        self.setup_eval_script()

    def setup_bbox(self, input_bbox):
        """
        Prepares the bbox for use with Sentinel Hub
        :param input_bbox: list, tuple or sentinelhub.Bbox. If hashable type: xmin, ymin, xmax, ymax
        :return: nothing
        """
        try:
            x = input_bbox.min_x
        except AttributeError:  # it's tuple or list
            x, y = input_bbox[0], input_bbox[3]
            if x <= 180 and y <= 90:  # given as lon lat
                upper_left_utm = utm.from_latlon(y, x)
                lower_right_utm = utm.from_latlon(input_bbox[2], input_bbox[1])
                hemisphere = "N" if y >= 0 else "S"
                crs = CRS["UTM_" + str(upper_left_utm[2]) + hemisphere]
                input_bbox = (upper_left_utm[0], lower_right_utm[1], lower_right_utm[0], upper_left_utm[1])
                self.bbox = BBox(bbox=input_bbox, crs=crs)
            else:
                raise ValueError("bbox not EPSG:4326")
        else:
            self.bbox = input_bbox
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

    def request(self, config, dir_save):
        input_data = [SentinelHubRequest.input_data(data_collection=self.dataset, time_interval=self.time_period)]
        response = [SentinelHubRequest.output_response("default", MimeType.TIFF)]
        kwargs = {"evalscript": self.eval_script, "input_data": input_data, "responses": response, "bbox": self.bbox,
                  "size": self.bbox_size, "config": config}
        if dir_save is not None:
            kwargs["data_folder"] = dir_save
        return SentinelHubRequest(**kwargs)
