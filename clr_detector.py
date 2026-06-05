import cv2
import numpy as np
import onnxruntime

from lane_detector import classify_color, visualize

INPUT_WIDTH = 800
INPUT_HEIGHT = 320
DEFAULT_MODEL_PATH = "models/clrnet_culane_r18.onnx"


class CLRNetONNX:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, conf_threshold=0.3, nms_threshold=50, max_lanes=8):
        self.session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_lanes = max_lanes
        self.n_offsets = 72
        self.n_strips = 71
        self.prior_ys = np.linspace(1, 0, self.n_offsets)

    @staticmethod
    def _softmax(values, axis=None):
        values = values - values.max(axis=axis, keepdims=True)
        exp_values = np.exp(values)
        return exp_values / exp_values.sum(axis=axis, keepdims=True)

    def _lane_iou(self, left, right, threshold):
        start_left = int(left[2] * self.n_strips + 0.5)
        start_right = int(right[2] * self.n_strips + 0.5)
        start = max(start_left, start_right)
        end_left = start_left + left[4] - 1 + 0.5 - (1 if (left[4] - 1) < 0 else 0)
        end_right = start_right + right[4] - 1 + 0.5 - (1 if (right[4] - 1) < 0 else 0)
        end = int(min(min(end_left, end_right), self.n_strips))
        if end - start < 0:
            return False
        distance = 0.0
        for index in range(5 + start, 5 + end):
            distance += abs(left[index] - right[index])
        return distance < threshold * (end - start + 1)

    def _nms(self, proposals, scores):
        keep = []
        indices = np.argsort(-scores)
        filtered = np.zeros(len(scores))
        for order_index, proposal_index in enumerate(indices):
            if filtered[order_index] == 1:
                continue
            keep.append(proposal_index)
            if len(keep) >= self.max_lanes or order_index == len(scores) - 1:
                break
            for sub_index, other_index in enumerate(indices[order_index + 1:]):
                if self._lane_iou(proposals[proposal_index], proposals[other_index], self.nms_threshold):
                    filtered[order_index + 1 + sub_index] = 1
        return keep

    def _decode(self, predictions, image_width, image_height):
        lanes = []
        for lane in predictions:
            lane_xs = lane[6:].copy()
            start = min(max(0, int(round(lane[2] * self.n_strips))), self.n_strips)
            length = int(round(lane[5]))
            end = min(start + length - 1, len(self.prior_ys) - 1)
            valid_mask = ~((((lane_xs[:start] >= 0.0) & (lane_xs[:start] <= 1.0))[::-1].cumprod()[::-1]).astype(bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][valid_mask] = -2
            lane_ys = self.prior_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = np.flip(np.double(lane_xs), axis=0)
            lane_ys = np.flip(lane_ys, axis=0)
            if len(lane_xs) <= 1:
                continue
            points = []
            for x_value, y_value in zip(lane_xs, lane_ys):
                x_pixel = x_value * image_width
                y_pixel = y_value * image_height
                if 0 <= x_pixel < image_width and 0 <= y_pixel < image_height:
                    points.append((int(round(x_pixel)), int(round(y_pixel))))
            if len(points) >= 2:
                lanes.append(points)
        return lanes

    def detect_points(self, image_bgr):
        image_height, image_width = image_bgr.shape[:2]
        resized = cv2.resize(image_bgr, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
        normalized = resized.astype(np.float32) / 255.0
        blob = normalized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        output = self.session.run(None, {self.input_name: blob})[0][0]
        scores = self._softmax(output[:, :2], axis=1)[:, 1]
        keep = scores >= self.conf_threshold
        predictions = output[keep]
        scores = scores[keep]
        if predictions.shape[0] == 0:
            return []
        nms_predictions = np.concatenate([predictions[..., :4], predictions[..., 5:]], axis=-1)
        nms_predictions[..., 4] = nms_predictions[..., 4] * self.n_strips
        nms_predictions[..., 5:] = nms_predictions[..., 5:] * (INPUT_WIDTH - 1)
        keep_indices = self._nms(nms_predictions, scores)
        predictions = predictions[keep_indices]
        if predictions.shape[0] == 0:
            return []
        predictions[:, 5] = np.round(predictions[:, 5] * self.n_strips)
        return self._decode(predictions, image_width, image_height)


COLOR_BGR = {"yellow": (0, 215, 255), "white": (255, 255, 255)}


BORDER_LEFT_X = 0.01
BORDER_RIGHT_X = 0.95
BORDER_RIGHT_DUP_X = 0.97


def _lane_geometry(points, image_width, image_height):
    points_array = np.array(points, dtype=np.float32)
    top = points_array[np.argmin(points_array[:, 1])]
    bottom = points_array[np.argmax(points_array[:, 1])]
    return {
        "top_x": float(top[0] / image_width),
        "top_y": float(top[1] / image_height),
        "bottom_x": float(bottom[0] / image_width),
        "bottom_y": float(bottom[1] / image_height),
    }


def _is_border_false_positive(lane, image_width, image_height):
    geom = _lane_geometry(lane["points"], image_width, image_height)
    color = lane["color"]
    if color == "yellow" and geom["bottom_x"] >= BORDER_RIGHT_X and geom["bottom_y"] < 0.85:
        return True
    if color == "white" and geom["bottom_x"] <= BORDER_LEFT_X and 0.70 <= geom["bottom_y"] <= 0.85 and geom["top_y"] >= 0.37:
        return True
    return False


def _drop_duplicate_extreme_right_lanes(lanes, image_width, image_height):
    indexed = []
    for index, lane in enumerate(lanes):
        geom = _lane_geometry(lane["points"], image_width, image_height)
        indexed.append((index, lane, geom))
    drop_indices = set()
    for i, (_, first_lane, first_geom) in enumerate(indexed):
        if first_geom["bottom_x"] < BORDER_RIGHT_DUP_X:
            continue
        for second_index, second_lane, second_geom in indexed[i + 1:]:
            if second_geom["bottom_x"] < BORDER_RIGHT_DUP_X or first_lane["color"] != second_lane["color"]:
                continue
            if abs(first_geom["bottom_x"] - second_geom["bottom_x"]) <= 0.015:
                first_index = indexed[i][0]
                if first_geom["bottom_y"] < second_geom["bottom_y"]:
                    drop_indices.add(first_index)
                else:
                    drop_indices.add(second_index)
    return [lane for index, lane in enumerate(lanes) if index not in drop_indices]


def filter_false_positives(lanes, image_width, image_height):
    filtered = [lane for lane in lanes if not _is_border_false_positive(lane, image_width, image_height)]
    return _drop_duplicate_extreme_right_lanes(filtered, image_width, image_height)


def load_model(model_path=DEFAULT_MODEL_PATH, conf_threshold=0.3, max_lanes=8):
    return CLRNetONNX(model_path=model_path, conf_threshold=conf_threshold, max_lanes=max_lanes)


def detect(model, img_bgr):
    image_height, image_width = img_bgr.shape[:2]
    results = []
    for points in model.detect_points(img_bgr):
        if points:
            results.append({"points": points, "color": classify_color(img_bgr, points)})
    return filter_false_positives(results, image_width, image_height)
