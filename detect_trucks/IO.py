import pickle
import numpy as np
import rasterio as rio


class IO:
    def __init__(self, rf_truck_detector):
        self.detector = rf_truck_detector

    def read_bands(self, file_path):
        try:
            with rio.open(file_path, "r") as src:
                self.detector.meta = src.meta
                if src.count < 4:
                    raise TypeError("Need 4 bands but %s given" % src.count)
                band_stack = np.zeros((src.count, src.height, src.width), dtype=np.float32)
                for band_idx in range(src.count):
                    band_stack[band_idx] = src.read(band_idx + 1)
        except rio.errors.RasterioIOError as e:
            print("Could not read from %s" % file_path)
            raise e
        else:
            print("Read %s bands from %s" % (self.detector.meta["count"], file_path))
        return band_stack

    def prediction_boxes_to_gpkg(self, prediction_boxes, file_path):
        self._write_boxes(file_path, prediction_boxes, ".gpkg")

    def prediction_boxes_to_geojson(self, prediction_boxes, file_path):
        self._write_boxes(file_path, prediction_boxes, ".geojson")

    def prediction_raster_to_gtiff(self, prediction_raster, file_path):
        if not any([file_path.endswith(suffix) for suffix in [".tif", ".tiff"]]):
            file_path += ".tiff"
        meta_copy = self.detector.meta.copy()
        meta_copy["count"] = 1
        meta_copy["dtype"] = prediction_raster.dtype
        try:
            with rio.open(file_path, "w", **meta_copy) as tgt:
                tgt.write(prediction_raster, 1)
        except rio.errors.RasterioIOError as e:
            print("Could not write to %s" % file_path)
            raise e
        else:
            print("Wrote to: %s" % file_path)

    @staticmethod
    def read_model(path):
        try:
            model = pickle.load(open(path, "rb"))
        except FileNotFoundError as e:
            raise FileNotFoundError("Model file not found at: %s" % path)
        return model

    @staticmethod
    def write_model(model, path):
        pickle.dump(model, open(path, "wb"))

    @staticmethod
    def _write_boxes(file_path, prediction_boxes, suffix):
        if not file_path.endswith(suffix):
            file_path += suffix
        drivers = {".geojson": "GeoJSON", ".gpkg": "GPKG"}
        try:
            prediction_boxes.to_file(file_path, driver=drivers[suffix])
        except TypeError as e:
            raise e
        else:
            print("Wrote to: %s" % file_path)
