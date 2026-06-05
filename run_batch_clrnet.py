import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clr_detector as detector

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def list_images(folder):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    return sorted(files)


def load_image_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def save_image_unicode(path, image):
    ext = os.path.splitext(path)[1] or ".jpg"
    success, encoded = cv2.imencode(ext, image)
    if not success:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input image folder")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--weights", default=detector.DEFAULT_MODEL_PATH, help="Path to the CLRNet ONNX model")
    parser.add_argument("--conf-threshold", type=float, default=0.3, help="Lane confidence threshold")
    parser.add_argument("--max-lanes", type=int, default=8, help="Maximum lanes kept after NMS")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    model = detector.load_model(args.weights, conf_threshold=args.conf_threshold, max_lanes=args.max_lanes)
    images = list_images(args.input)
    print(f"Found {len(images)} images")

    if images:
        warmup = load_image_unicode(images[0])
        if warmup is not None:
            detector.detect(model, warmup)

    results = {}
    total_time = 0.0
    total_lanes = 0
    total_yellow = 0

    for index, image_path in enumerate(images, start=1):
        image = load_image_unicode(image_path)
        if image is None:
            continue
        start = time.time()
        lanes = detector.detect(model, image)
        elapsed = time.time() - start
        total_time += elapsed

        visualization = detector.visualize(image, lanes)
        output_name = os.path.splitext(os.path.basename(image_path))[0] + "_vis.jpg"
        save_image_unicode(os.path.join(args.output, output_name), visualization)

        results[os.path.basename(image_path)] = {
            "num_lanes": len(lanes),
            "num_yellow": sum(1 for lane in lanes if lane["color"] == "yellow"),
            "num_white": sum(1 for lane in lanes if lane["color"] == "white"),
            "time_sec": round(elapsed, 4),
            "lanes": [
                {
                    "color": lane["color"],
                    "num_points": len(lane["points"]),
                    "points": lane["points"],
                }
                for lane in lanes
            ],
        }
        total_lanes += len(lanes)
        total_yellow += sum(1 for lane in lanes if lane["color"] == "yellow")
        print(f"[{index}/{len(images)}] {os.path.basename(image_path)}: {len(lanes)} lanes, {elapsed * 1000:.1f} ms")

    count = len(images)
    summary = {
        "model": "CLRNet CULane ResNet18 ONNX",
        "num_images": count,
        "total_lanes_detected": total_lanes,
        "total_yellow_detected": total_yellow,
        "total_white_detected": total_lanes - total_yellow,
        "total_time_sec": round(total_time, 3),
        "avg_time_per_image_sec": round(total_time / count, 4) if count else 0,
        "avg_time_per_image_ms": round(total_time / count * 1000, 1) if count else 0,
        "fps": round(count / total_time, 2) if total_time else 0,
        "conf_threshold": args.conf_threshold,
        "max_lanes": args.max_lanes,
    }
    results["__summary__"] = summary

    with open(os.path.join(args.output, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    print("\n===== SUMMARY =====")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nVisualizations + results.json saved to: {args.output}")


if __name__ == "__main__":
    main()
