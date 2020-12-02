def _sub_window_test(self, indices_dict, arr):
    indices_low, indices_up = indices_dict["indices_low"], indices_dict["indices_up"]
    indices = indices_dict["indices"]
    window = np.zeros((16, len(indices_low), len(indices_up)))
    some_nans = np.repeat([np.nan], window.shape[0])
    for y_idx, y_low, y_up in zip(indices, indices_low, indices_up):
        for x_idx, x_low, x_up in zip(indices, indices_low, indices_up):
            sub_arr = arr[0:3, y_low:y_up, x_low:x_up]
            try:
                max_red = np.nanmax(sub_arr[0])
                max_green = np.nanmax(sub_arr[1])
                max_blue = np.nanmax(sub_arr[2])
            except ValueError:
                window[0:window.shape[0], y_idx, x_idx] = some_nans
                continue
            sub_arr[np.isnan(sub_arr)] = 0
            vector_dict = self._characterize_spatial_spectral(sub_arr)
            spectral_angle, slope = vector_dict["spectral_angle"], np.abs(vector_dict["slope"])
            window[0, y_idx, x_idx] = spectral_angle * 20
            window_smaller_5 = len(indices) < 5
            is_left_corner = window_smaller_5 and y_idx == x_idx == 0 or y_idx == indices[-1] and x_idx == 0
            is_right_corner = window_smaller_5 and y_idx == 0 and x_idx == indices[-1] or x_idx == y_idx == indices[-1]
            thresholds_exceeded = spectral_angle < MIN_R_SQUARED or slope > MAX_SLOPE
            if is_left_corner or is_right_corner or thresholds_exceeded:
                window[0:window.shape[0], y_idx, x_idx] = some_nans
                continue
            window[1, y_idx, x_idx] = vector_dict["spatial_angle"]
            window[2, y_idx, x_idx] = (vector_dict["green_length"] / vector_dict["red_length"]) < 0.66
            window[3, y_idx, x_idx] = np.float32(vector_dict["direction"])
            window[4, y_idx, x_idx] = -slope * 10
            window[5, y_idx, x_idx] = max_red * 10
            window[6, y_idx, x_idx] = max_green * 10
            window[7, y_idx, x_idx] = max_blue * 10
            window[8, y_idx, x_idx] = np.max(np.nanstd(sub_arr, axis=0)) * 100
            window[9, y_idx, x_idx] = np.nanstd(sub_arr) * 100
            window[10, y_idx, x_idx] = np.nanstd(normalized_ratio(sub_arr[0], sub_arr[2])) * 10
            window[11, y_idx, x_idx] = np.nanstd(normalized_ratio(sub_arr[0], sub_arr[1])) * 10
            window[12, y_idx, x_idx] = np.nanstd(normalized_ratio(sub_arr[1], sub_arr[0])) * 10
            window[13, y_idx, x_idx] = np.nanstd(normalized_ratio(sub_arr[1], sub_arr[2])) * 10
            window[14, y_idx, x_idx] = np.nanstd(normalized_ratio(sub_arr[2], sub_arr[0])) * 10
            window[15, y_idx, x_idx] = np.nanstd(normalized_ratio(sub_arr[2], sub_arr[1])) * 10
    return window

@staticmethod
def eliminate_close_reds(sub_arr, ratios, quantiles_sum):
    center = [int(sub_arr.shape[1] / 2), int(sub_arr.shape[2] / 2)]
    ymin, xmin = center[0] - 1, center[1] - 1
    window_3x3 = ratios[:, ymin:center[0]+2, xmin:center[1]+2]
    try:
        max_indices = np.where(window_3x3 == np.nanmax(window_3x3, 0))
    except ValueError:
        return sub_arr
    for i in range(len(max_indices[0])):
        if max_indices[0][i] in [0, 1]:
            sub_arr[:, ymin + max_indices[1][i], xmin + max_indices[2][i]] = np.nan
            ratios[:, ymin + max_indices[1][i], xmin + max_indices[2][i]] = np.nan
            quantiles_sum[ymin + max_indices[1][i], xmin + max_indices[2][i]] = np.nan
    return sub_arr, ratios, quantiles_sum

def _calc_match(self, window_result):
    spectral_angles = window_result[0].copy()
    spatial_angles = window_result[1].copy()
    score, stat_index = self.calc_score(window_result)
    if np.count_nonzero(~np.isnan(score)) == 0:
        return None, None, None, None
    max_score = np.nanmax(score)
    spectral_matches = np.where(score == max_score)
    if len(spectral_matches[0]) == 0:
        return None, None, None, None
    # if several matches select the one with highest window statistics
    window_sums, i = np.zeros(len(spectral_matches[0])), 0
    for y, x in zip(spectral_matches[0], spectral_matches[1]):
        window_sums[i], i = stat_index[y, x], i + 1
    match = np.where(window_sums == window_sums.max())[0][0]
    y, x = spectral_matches[0][match], spectral_matches[1][match]
    return y, x, max_score, spatial_angles[y, x]

def crop_box(self, subset, the_box):
    """
    :param subset: 3d numpy array
    :param the_box: dict with xmin, ymin, xmax, ymax int
    :return:
    """
    #return the_box
    buffer_threshold = 3  # percent
    box_sub_subset = subset[0:3, the_box["ymin"]:the_box["ymax"], the_box["xmin"]:the_box["xmax"]].copy()
    blue, green, red = box_sub_subset[2].copy(), box_sub_subset[1].copy(), box_sub_subset[0].copy()
    blue_ratio, green_ratio, red_ratio = normalized_ratio(blue, red), normalized_ratio(green, blue), normalized_ratio(red, blue)
    a_list = [[]]
    rgb_quantiles = [a_list, a_list, a_list]
    for i, band in enumerate([blue, red, green]):
        q = np.array([0.92])
        while len(rgb_quantiles[i][0]) < 3:
            q -= 0.02
            rgb_quantiles[i] = np.where(band > np.nanquantile(band, q))
    boxes, spectral, vector_dicts = [], [], []
    for y_blue, x_blue in zip(rgb_quantiles[0][0], rgb_quantiles[0][1]):
        for y_red, x_red in zip(rgb_quantiles[1][0], rgb_quantiles[1][1]):
            ys, xs = [y_blue, y_red], [x_blue, x_red]
            another_box = [min(ys), min(xs), max(ys) + 1, max(xs) + 1]
            for green_y, green_x in zip(rgb_quantiles[2][0], rgb_quantiles[2][1]):
                another_box_polygon = box(another_box[1], another_box[0], another_box[3], another_box[2])
            if not another_box_polygon.covers(Point([green_x, green_y])):
                continue
            another_subset = box_sub_subset[:, another_box[0]:another_box[2], another_box[1]:another_box[3]]
            vector_dict = self._characterize_spatial_spectral(another_subset)
            match = self._spatial_spectral_match(vector_dict)
            if match:
                boxes.append(another_box)
                spectral.append(vector_dict["spectral_angle"])
                vector_dicts.append(vector_dict)
    spectral = np.array(spectral)
    try:
        idx = np.where(spectral == spectral.max())[0]
    except ValueError:
        return {}
    # +1 bc may be 0
    sizes = np.array([(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) for i in idx])
    selected_idx = idx[np.where(sizes == sizes.max())[0][0]]
    crop_box = boxes[selected_idx]
    vector_dict = vector_dicts[selected_idx]
    the_box["ymax"] = the_box["ymin"] + crop_box[2]
    the_box["xmax"] = the_box["xmin"] + crop_box[3]
    the_box["ymin"] += crop_box[0]
    the_box["xmin"] += crop_box[1]
    new_subset = subset[0:3, the_box["ymin"]:the_box["ymax"], the_box["xmin"]:the_box["xmax"]].copy()
    new_subset[np.isnan(new_subset)] = 0.
    quantiles_sum = self.quantile_filter(new_subset[0:3], [0.33])
    if quantiles_sum is None:
        return {}
    indices = np.where(np.int8(quantiles_sum > 0))
    if len(indices[0]) < 3:
        return {}
    ymin, ymax, xmin, xmax = np.min(indices[0]), np.max(indices[0]), np.min(indices[1]), np.max(indices[1])
    the_box["ymax"] = the_box["ymin"] + ymax + 1
    the_box["xmax"] = the_box["xmin"] + xmax + 1
    the_box["ymin"] += ymin
    the_box["xmin"] += xmin
    vector_dict = self._characterize_spatial_spectral(subset[0:3, the_box["ymin"]:the_box["ymax"], the_box["xmin"]:the_box["xmax"]])
    match = self._spatial_spectral_match(vector_dict)
    plot_img(subset[0:3, the_box["ymin"]:the_box["ymax"], the_box["xmin"]:the_box["xmax"]])
    if match:
        return {"cropped_box": the_box, "rsquared": vector_dict["spectral_angle"], "slope": vector_dict["slope"],
                "direction": vector_dict["direction"]}
    else:
        return {}

