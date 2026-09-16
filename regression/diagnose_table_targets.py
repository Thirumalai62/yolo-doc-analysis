"""Audit native positive targets and search eligible train PDFs for procurement layouts."""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.loss import v8DetectionLoss

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset_v5_corrected"
WEIGHTS = ROOT / "runs/legal_notice_v6_clslogit_cpu_2_r1/weights/best.pt"


def page_tensor(path, auto=True):
    im = cv2.imread(str(path))
    h, w = im.shape[:2]
    r = min(1280 / h, 1280 / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = 1280 - nw, 1280 - nh
    if auto:
        dw, dh = dw % 32, dh % 32
    left, top = round(dw / 2 - 0.1), round(dh / 2 - 0.1)
    padded = LetterBox(new_shape=(1280, 1280), auto=auto, stride=32)(image=im)
    tensor = torch.from_numpy(np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1))).float()[None] / 255
    return tensor, (w, h, r, left, top)


def labels(path, transform, shape):
    w, h, r, left, top = transform
    boxes = []
    for line in path.read_text().splitlines():
        _, x, y, bw, bh = map(float, line.split())
        boxes.append([(x * w * r + left) / shape[1], (y * h * r + top) / shape[0], bw * w * r / shape[1], bh * h * r / shape[0]])
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)


def main():
    import pypdfium2 as pdfium
    model = YOLO(str(WEIGHTS)).model.eval()
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    criterion = v8DetectionLoss(model)
    records = []
    for stem in ("gulftoday_2026-08-14_page_0013", "gulftoday_2026-08-21_page_0013", "gulftoday_2026-09-02_page_0012", "albayan_2026-08-18_page_0024"):
        for auto in (False, True):
            t, transform = page_tensor(DATA / f"images/train/{stem}.png", auto)
            gt = labels(DATA / f"labels/train/{stem}.txt", transform, t.shape[-2:])
            batch = {"batch_idx": torch.zeros(len(gt)), "cls": torch.zeros(len(gt), 1), "bboxes": gt}
            captured = []
            hook = criterion.assigner.register_forward_hook(lambda m, a, out: captured.append(out))
            with torch.no_grad():
                raw = model(t)[1]["one2many"]
                criterion(raw, batch)
            hook.remove()
            _, _, scores, foreground, indices = captured[0]
            prob = raw["scores"][0, 0].sigmoid()
            targets = scores[0, :, 0]
            objects = []
            for i in range(len(gt)):
                mask = foreground[0] & (indices[0] == i)
                objects.append({"label_line": i + 1, "assigned_anchors": int(mask.sum()),
                                "highest_current_score": float(prob[mask].max()) if mask.any() else None,
                                "highest_target_score": float(targets[mask].max()) if mask.any() else None,
                                "sum_score_gradient": float((prob[mask] - targets[mask]).sum())})
            records.append({"image": stem, "padding": "detect_rectangular" if auto else "previous_training_square", "shape": list(t.shape), "objects": objects})
    issues = sorted({p.stem.rsplit("_page_", 1)[0] for p in (DATA / "images/train").glob("*.png")})
    candidates = []
    for issue in issues:
        pdf = ROOT / "input" / f"{issue}.pdf"
        with pdfium.PdfDocument(str(pdf)) as document:
            for n, page in enumerate(document, 1):
                textpage = page.get_textpage()
                text = " ".join(textpage.get_text_range().lower().split())
                textpage.close()
                if "pakistan airports" in text or ("procurement" in text and "corrigendum" in text):
                    candidates.append({"issue": issue, "page": n, "already_training": (DATA / f"images/train/{issue}_page_{n:04d}.png").exists()})
                page.close()
    output = ROOT / "output/regression_audit/target_assignment_diagnosis.json"
    output.write_text(json.dumps({"target_audit": records, "train_pdf_procurement_candidates": candidates}, indent=2) + "\n")
    print(json.dumps({"target_audit": records, "train_pdf_procurement_candidates": candidates}, indent=2))


if __name__ == "__main__":
    main()
