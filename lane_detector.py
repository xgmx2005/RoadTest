import cv2
import numpy as np
import scipy.special
import torch
import torchvision.transforms as transforms

from ufld_model import ParsingNet

GRIDING_NUM = 200
NUM_LANES = 4
CLS_NUM_PER_LANE = 18
CULANE_ROW_ANCHOR = [
    121, 131, 141, 150, 160, 170, 180, 189, 199,
    209, 219, 228, 238, 248, 258, 267, 277, 287,
]
INPUT_SIZE = (288, 800)

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def load_model(weight_path, device="cpu"):
    net = ParsingNet(
        pretrained=False,
        backbone="18",
        cls_dim=(GRIDING_NUM + 1, CLS_NUM_PER_LANE, NUM_LANES),
        use_aux=False,
    ).to(device)
    state_dict = torch.load(weight_path, map_location="cpu")["model"]
    compatible = {}
    for key, value in state_dict.items():
        compatible[key[7:] if key.startswith("module.") else key] = value
    net.load_state_dict(compatible, strict=False)
    net.eval()
    return net


def _preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (INPUT_SIZE[1], INPUT_SIZE[0]), interpolation=cv2.INTER_LINEAR)
    return _TRANSFORM(resized).unsqueeze(0)


def _postprocess(out, img_w, img_h):
    col_sample = np.linspace(0, INPUT_SIZE[1] - 1, GRIDING_NUM)
    col_sample_w = col_sample[1] - col_sample[0]

    out_j = out[0].data.cpu().numpy()
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = (np.arange(GRIDING_NUM) + 1).reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    argmax = np.argmax(out_j, axis=0)
    loc[argmax == GRIDING_NUM] = 0
    out_j = loc

    lanes = []
    for lane_idx in range(out_j.shape[1]):
        points = []
        if np.sum(out_j[:, lane_idx] != 0) > 2:
            for anchor_idx in range(out_j.shape[0]):
                if out_j[anchor_idx, lane_idx] > 0:
                    x = int(out_j[anchor_idx, lane_idx] * col_sample_w * img_w / INPUT_SIZE[1]) - 1
                    y = int(img_h * (CULANE_ROW_ANCHOR[CLS_NUM_PER_LANE - 1 - anchor_idx] / INPUT_SIZE[0])) - 1
                    points.append((x, y))
        lanes.append(points)
    return lanes


def classify_color(img_bgr, points, patch=8):
    if not points:
        return None
    height, width = img_bgr.shape[:2]
    selected = []
    for x, y in points:
        x0, x1 = max(0, x - patch), min(width, x + patch + 1)
        y0, y1 = max(0, y - patch), min(height, y + patch + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        region = img_bgr[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
        if len(region) == 0:
            continue
        value = region.max(axis=1)
        threshold = max(np.percentile(value, 70), 110)
        bright = region[value >= threshold]
        if len(bright) > 0:
            selected.append(bright)
    if not selected:
        return "white"

    pixels = np.concatenate(selected, axis=0)
    blue, green, red = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    ratio = np.median(blue / (np.maximum(red, green) + 1e-6))
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sat = float(np.median(hsv[:, 1]))
    hue = float(np.median(hsv[:, 0]))
    if ratio < 0.66 and sat >= 65 and 8 <= hue <= 45:
        return "yellow"
    return "white"


COLOR_BGR = {"yellow": (0, 215, 255), "white": (255, 255, 255)}


def detect(net, img_bgr, device="cpu"):
    img_h, img_w = img_bgr.shape[:2]
    tensor = _preprocess(img_bgr).to(device)
    with torch.no_grad():
        out = net(tensor)
    lanes = _postprocess(out, img_w, img_h)
    results = []
    for points in lanes:
        if points:
            results.append({"points": points, "color": classify_color(img_bgr, points)})
    return results


def visualize(img_bgr, results):
    vis = img_bgr.copy()
    for lane in results:
        points = lane["points"]
        color = lane["color"]
        draw_color = COLOR_BGR.get(color, (0, 255, 0))
        for x, y in points:
            cv2.circle(vis, (x, y), 6, (0, 0, 0), -1)
            cv2.circle(vis, (x, y), 4, draw_color, -1)
        if len(points) >= 2:
            points_array = np.array(points, dtype=np.int32)
            cv2.polylines(vis, [points_array], False, draw_color, 2, cv2.LINE_AA)
            label_x, label_y = points_array[np.argmin(points_array[:, 1])]
            text_origin = (label_x + 5, max(15, label_y - 5))
            cv2.putText(vis, color, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, color, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 1, cv2.LINE_AA)
    yellow_count = sum(1 for lane in results if lane["color"] == "yellow")
    white_count = sum(1 for lane in results if lane["color"] == "white")
    summary = f"lanes={len(results)} yellow={yellow_count} white={white_count}"
    cv2.putText(vis, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    return vis
