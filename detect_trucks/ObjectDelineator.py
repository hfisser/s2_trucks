import numpy as np
import geopandas as gpd
from shapely.geometry import box, Polygon


class ObjectExtractor:
    def __init__(self, rf_truck_detector):
        self.detector = rf_truck_detector

    def delineate(self, predictions_arr):
        detector = self.detector
        predictions_copy, probabilities_copy = predictions_arr.copy(), detector.probabilities.copy()
        predictions_copy[predictions_copy == 1] = 0
        blue_ys, blue_xs = np.where(predictions_copy == 2)
        out_gpd = gpd.GeoDataFrame(crs=detector.meta["crs"])
        detection_boxes, directions, direction_descriptions, speeds, mean_probs, sub_size = [], [], [], [], [], 9
        for y_blue, x_blue in zip(blue_ys, blue_xs):
            if predictions_copy[y_blue, x_blue] == 0:
                continue
            subset_9 = detector.get_arr_subset(predictions_copy, y_blue, x_blue, sub_size).copy()
            subset_3 = detector.get_arr_subset(predictions_copy, y_blue, x_blue, 3).copy()
            subset_9_probs = detector.get_arr_subset(probabilities_copy, y_blue, x_blue, sub_size).copy()
            half_idx_y = y_blue if subset_9.shape[0] < sub_size else int(subset_9.shape[0] * 0.5)
            half_idx_x = x_blue if subset_9.shape[1] < sub_size else int(subset_9.shape[1] * 0.5)
            try:
                current_value = subset_9[half_idx_y, half_idx_x]
            except IndexError:  # upper array edge
                half_idx_y, half_idx_x = int(sub_size / 2), int(sub_size / 2)  # index from lower edge is ok
                current_value = subset_9[half_idx_y, half_idx_x]
            new_value = 100
            if not all([value in subset_9 for value in [2, 3, 4]]):
                continue
            result_tuple = self._cluster_array(arr=subset_9,
                                               probs=subset_9_probs,
                                               point=[half_idx_y, half_idx_x],
                                               new_value=new_value,
                                               current_value=current_value,
                                               yet_seen_indices=[],
                                               yet_seen_values=[],
                                               joker_played=False)
            cluster = result_tuple[0]
            if np.count_nonzero(cluster == new_value) < 3:
                continue
            else:
                out_gpd, predictions_copy = self._postprocess_cluster(cluster, predictions_copy,
                                                                      subset_3, y_blue, x_blue,
                                                                      half_idx_y, half_idx_x, new_value, out_gpd)
        return out_gpd

    def _cluster_array(self, arr, probs, point, new_value, current_value, yet_seen_indices, yet_seen_values,
                       joker_played):
        """
        looks for non zeros in 3x3 window around point in array and assigns a new value to these non-zeros
        :param arr: np array
        :param point: list of int y, x indices
        :param new_value: int value to assign
        :param current_value: value from np array
        :param yet_seen_indices: list of lists, each list is a point with int y, x indices that has been seen before
        :param yet_seen_values: list of values, each value is a value at the yet_seen_indices
        :return: tuple of np array and list
        """
        detector = self.detector
        joker_played = True
        if len(yet_seen_indices) == 0:
            yet_seen_indices.append(point)
            yet_seen_values.append(current_value)
        arr_modified = arr.copy()
        arr_modified[point[0], point[1]] = 0
        window_3x3 = detector.get_arr_subset(arr_modified.copy(), point[0], point[1], 3)
        y, x, ys, xs, window_idx, offset_y, offset_x = point[0], point[1], [], [], 0, 0, 0
        window_5x5_no_corners = detector.eliminate_array_corners(detector.get_arr_subset(arr_modified.copy(), y, x, 5), 1)
        window_3x3_probs = detector.get_arr_subset(probs, y, x, 3)
        window_5x5_probs_no_corners = detector.eliminate_array_corners(detector.get_arr_subset(probs, y, x, 5), 1)
        # first look for values on horizontal and vertical, if none given try corners
        windows, windows_probs = [window_3x3, window_5x5_no_corners], [window_3x3_probs, window_5x5_probs_no_corners]
        windows = windows[0:1] if current_value == 4 or joker_played else windows
        while len(ys) == 0 and window_idx < len(windows):
            window = windows[window_idx]
            window_probs = windows_probs[window_idx]
            offset_y, offset_x = int(window.shape[0] / 2), int(window.shape[1] / 2)  # offset for window ymin and xmin
            go_next = current_value + 1 in window or current_value == 2
            target_value = current_value + 1 if go_next else current_value
            match = window == target_value
            target_value = current_value if np.count_nonzero(match) == 0 else target_value
            match = window == target_value
            ys, xs = np.where(match)
            if len(ys) > 1:  # look for match with highest probability
                window_probs_target = window_probs[target_value - 1] * match
                max_prob = (window_probs_target == np.max(window_probs_target))
                ys, xs = np.where(max_prob)
            window_idx += 1
        ymin, xmin = int(np.clip(point[0] - offset_y, 0, np.inf)), int(np.clip(point[1] - offset_x, 0, np.inf))
        for y_local, x_local in zip(ys, xs):
            y, x = ymin + y_local, xmin + x_local
            if [y, x] not in yet_seen_indices or len(yet_seen_indices) == 0:
                try:
                    current_value = arr[y, x]
                except IndexError:
                    continue
                if 4 in yet_seen_values and current_value <= 3:  # red yet seen but this is green or blue
                    continue
                arr_modified[y, x] = new_value
                yet_seen_indices.append([y, x])
                yet_seen_values.append(current_value)
                # avoid picking many more reds than blues and greens
                n_picks = [np.count_nonzero(np.array(yet_seen_values) == value) for value in [2, 3, 4]]
                if n_picks[2] > n_picks[0] and n_picks[2] > n_picks[1]:
                    break  # finish clustering in order to avoid picking many reds at the edge of object
              #  if any([n > 5 for n in n_picks]):
               #     return np.zeros_like(arr_modified), yet_seen_indices, yet_seen_values, joker_played
                arr_modified, yet_seen_indices, yet_seen_values, joker_played = self._cluster_array(arr_modified,
                                                                                                    probs,
                                                                                                    [y, x],
                                                                                                    new_value,
                                                                                                    current_value,
                                                                                                    yet_seen_indices,
                                                                                                    yet_seen_values,
                                                                                                    joker_played)
        arr_modified[point[0], point[1]] = new_value
        return arr_modified, yet_seen_indices, yet_seen_values, joker_played

    def _postprocess_cluster(self, cluster, preds_copy, prediction_subset_3, y_blue, x_blue, half_idx_y, half_idx_x,
                             new_value, out_gpd):
        # add neighboring blue in 3x3 window around blue
        ys_blue_additional, xs_blue_additional = np.where(prediction_subset_3 == 2)
        ys_blue_additional += half_idx_y - 1  # get index in subset
        xs_blue_additional += half_idx_x - 1
        for y_blue_add, x_blue_add in zip(ys_blue_additional, xs_blue_additional):
            cluster[int(np.clip(y_blue_add, 0, np.inf)), int(np.clip(x_blue_add, 0, np.inf))] = new_value
        cluster[cluster != new_value] = 0
        cluster_ys, cluster_xs = np.where(cluster == new_value)
        # corner of 9x9 subset
        ymin_subset, xmin_subset = np.clip(y_blue - half_idx_y, 0, np.inf), np.clip(x_blue - half_idx_x, 0, np.inf)
        cluster_ys += ymin_subset.astype(cluster_ys.dtype)
        cluster_xs += xmin_subset.astype(cluster_xs.dtype)
        ymin, xmin = np.min(cluster_ys), np.min(cluster_xs)
        # +1 on index because Polygon has to extent up to upper bound of pixel (array coords at upper left corner)
        ymax, xmax = np.max(cluster_ys) + 1, np.max(cluster_xs) + 1
        # check if blue, green and red are given in box and box is large enough, otherwise drop
        box_preds = preds_copy[ymin:ymax, xmin:xmax].copy()
        box_probs = self.detector.probabilities[1:, ymin:ymax, xmin:xmax].copy()
        #box_probs = self.probabilities[:, ymin:ymax, xmin:xmax].copy()
        max_probs = [np.nanmax(box_probs[value - 2] * (box_preds == value)) for value in (2, 3, 4)]
        mean_max_spectral_probability = np.nanmean(max_probs)
        mean_spectral_probability = np.nanmean(np.nanmax(box_probs, 0))
        all_given = all([value in box_preds for value in [2, 3, 4]])
        large_enough = box_preds.shape[0] > 2 or box_preds.shape[1] > 2
        too_large = box_preds.shape[0] > 5 or box_preds.shape[1] > 5
        too_large += box_preds.shape[0] > 4 and box_preds.shape[1] > 4
       # too_large = False
        if too_large > 0 or not all_given or not large_enough:
            return out_gpd, preds_copy
        # calculate direction
        blue_y, blue_x = np.where(box_preds == 2)
        ry, rx = np.where(box_preds == 4)
        # simply use first index
        blue_indices = np.int8([blue_y[0], blue_x[0]])
        red_indices = np.int8([ry[0], rx[0]])
        direction = self.detector.calc_vector_direction_in_degree(red_indices - blue_indices)
        speed = self.detector.calc_speed(box_preds.shape)
        # create output box
        lat, lon = self.detector.lat, self.detector.lon
        lon_min, lat_min = lon[xmin], lat[ymin]
        try:
            lon_max = lon[xmax]
        except IndexError:  # may happen at edge of array
            # calculate next coordinate beyond array bound -> this is just the upper boundary of the box
            lon_max = lon[-1] + (lon[-1] - lon[-2])
        try:
            lat_max = lat[ymax]
        except IndexError:
            lat_max = lat[-1] + (lat[-1] - lat[-2])
        cluster_box = Polygon(box(lon_min, lat_min, lon_max, lat_max))
        if mean_max_spectral_probability > 0.4:
            # set box cells to zero value in predictions array
            preds_copy[ymin:ymax, xmin:xmax] *= np.zeros_like(box_preds)
            blue_indices = np.where(box_preds == 2)
            for yb, xb in zip(blue_indices[0], blue_indices[1]):  # 3x3 around cell blues to 0
                ymin, ymax = np.clip(yb - 1, 0, preds_copy.shape[0]), np.clip(yb + 2, 0, preds_copy.shape[0])
                xmin, xmax = np.clip(xb - 1, 0, preds_copy.shape[1]), np.clip(xb + 2, 0, preds_copy.shape[1])
                preds_copy[ymin:ymax, xmin:xmax] *= np.int8(preds_copy[ymin:ymax, xmin:xmax] != 2)
            rox_idx = len(out_gpd)
            for key, value in zip(["geometry", "id", "mean_spectral_probability",
                                   "mean_max_spectral_probability", "max_blue_probability",
                                   "max_green_probability", "max_red_probability", "direction_degree",
                                   "direction_description", "speed"],
                                  [cluster_box, rox_idx,
                                   mean_spectral_probability, mean_max_spectral_probability,
                                   max_probs[0], max_probs[1], max_probs[2], direction,
                                   self.detector.direction_degree_to_description(direction), speed]):
                out_gpd.loc[rox_idx, key] = value
        return out_gpd, preds_copy
