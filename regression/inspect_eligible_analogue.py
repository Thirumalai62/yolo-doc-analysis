"""Measure model detections over the reviewed September 11 PAA analogue."""
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "output/legal_notices/khaleejtimes_2026-09-11/rendered/khaleejtimes_2026-09-11/page_0006.png"
REGION = (0.04, 0.69, 0.35, 0.98)


def main():
    print({"image": str(IMAGE.relative_to(ROOT)), "size": Image.open(IMAGE).size, "region_xyxyn": REGION})
    weights = {
        "v6": ROOT / "runs/legal_notice_v6_clslogit_cpu_2_r1/weights/best.pt",
        "r6": ROOT / "runs/legal_notice_v8_protected_head_cpu_r6/weights/candidate.pt",
    }
    for name, path in weights.items():
        result = YOLO(str(path)).predict(str(IMAGE), conf=0.01, imgsz=1280, device="cpu", verbose=False)[0]
        hits = []
        for box in result.boxes:
            cx, cy = (float(value) for value in box.xywhn[0, :2])
            if REGION[0] <= cx <= REGION[2] and REGION[1] <= cy <= REGION[3]:
                hits.append({
                    "confidence": round(float(box.conf), 6),
                    "xyxyn": [round(float(value), 4) for value in box.xyxyn[0]],
                })
        print(name, hits)


if __name__ == "__main__":
    main()
