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


# color-classification tuning (see docs/模型改进_对照老师真值.md for derivation)
COLOR_YTOP = 0.30      # skip the top 30% of the line (horizon haze, tiny/noisy)
COLOR_YBOT = 0.90      # skip the bottom 10% (ego-vehicle hood / strong reflection)
COLOR_ROAD_OFFSET = 0.08   # road reference patch offset as fraction of image width
COLOR_ABS_DEFICIT = 0.15   # absolute blue-deficit floor for a yellow marking
COLOR_REL_DEFICIT = 0.06   # marking must exceed adjacent road by this much


def _blue_deficit(pixels):
    """Median (1 - B / max(R, G)); higher means more yellow, ~0 for white/gray."""
    blue, green, red = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    return float(np.median(1.0 - blue / (np.maximum(red, green) + 1e-6)))


def _bright_patch(img_bgr, x, y, patch, pct=65, min_bright=80):
    height, width = img_bgr.shape[:2]
    x0, x1 = max(0, x - patch), min(width, x + patch + 1)
    y0, y1 = max(0, y - patch), min(height, y + patch + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    region = img_bgr[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    if len(region) == 0:
        return region
    return region


def classify_color(img_bgr, points, patch=6):
    """Yellow vs white via road-relative blue-deficit.

    Lane markings (yellow or white) are brighter than the road; the
    discriminative cue is the blue channel. Global warm/sunset lighting tints
    both markings and road, so we measure each marking's blue-deficit *relative*
    to the adjacent road surface, which cancels out the global colour cast.
    Sampling is restricted to the mid-section of the line to avoid horizon haze
    and the ego-vehicle's (yellow) hood at the bottom of the frame.
    """
    if not points or len(points) < 2:
        return "white"
    height, width = img_bgr.shape[:2]
    ordered = sorted(points, key=lambda q: q[1])  # top -> bottom
    band = [(x, y) for (x, y) in ordered if COLOR_YTOP * height <= y <= COLOR_YBOT * height]
    if len(band) < 2:
        band = ordered
    offset = int(COLOR_ROAD_OFFSET * width)

    line_pixels, road_pixels = [], []
    for x, y in band:
        region = _bright_patch(img_bgr, x, y, patch)
        if region is None or len(region) == 0:
            continue
        value = region.max(axis=1)
        threshold = max(np.percentile(value, 65), 80)
        bright = region[value >= threshold]
        if len(bright) == 0:
            continue
        refs = [_bright_patch(img_bgr, x + dx, y, patch) for dx in (-offset, offset)]
        refs = [r for r in refs if r is not None and len(r) > 0]
        if not refs:
            continue
        line_pixels.append(bright)
        road_pixels.append(np.concatenate(refs, axis=0))

    if not line_pixels:
        return "white"
    line_deficit = _blue_deficit(np.concatenate(line_pixels, axis=0))
    road_deficit = _blue_deficit(np.concatenate(road_pixels, axis=0)) if road_pixels else 0.0
    if line_deficit >= COLOR_ABS_DEFICIT and (line_deficit - road_deficit) >= COLOR_REL_DEFICIT:
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
