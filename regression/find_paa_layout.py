"""Locate PAA-logo layout analogues in eligible non-evaluation renders for review."""
from pathlib import Path
import json
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    excluded = {p.stem.rsplit("_page_", 1)[0]
                for split in ("val", "test")
                for p in (ROOT / f"dataset_v5_corrected/images/{split}").glob("*.png")}
    excluded |= {
        "khaleejtimes_2026-09-04",  # held-out issue
        "khaleejtimes_2026-09-09", "khaleejtimes_2026-09-10",  # preservation suite
    }
    manifest = json.loads((ROOT / "output/regression_audit/legal_notice_v8_protected_scores_cpu_r3/july_manifest.json").read_text())
    source = next(p for p in manifest["pages"] if p["issue"] == "khaleejtimes_2026-07-03" and p["page"] == 2)
    im = cv2.imread(str(ROOT / source["image"]), cv2.IMREAD_GRAYSCALE)
    logo = im[3200:3320, 118:308]
    logo = cv2.resize(logo, None, fx=800/im.shape[1], fy=800/im.shape[1])
    output = ROOT / "output/regression_audit/paa_layout_search"
    output.mkdir(exist_ok=True)
    cv2.imwrite(str(output / "query_logo.png"), logo)
    results = []
    for directory in sorted((ROOT / "output/legal_notices").glob("khaleejtimes_2026-*")):
        if directory.name in excluded or "-07-" in directory.name:
            continue
        for path in sorted((directory / "rendered" / directory.name).glob("*.png")):
            page = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            page = cv2.resize(page, (800, round(page.shape[0] * 800 / page.shape[1])))
            best = (0.0, None, None)
            for scale in (0.65, 0.8, 1.0, 1.2, 1.5):
                template = cv2.resize(logo, None, fx=scale, fy=scale)
                scores = cv2.matchTemplate(page, template, cv2.TM_CCOEFF_NORMED)
                _, score, _, location = cv2.minMaxLoc(scores)
                if score > best[0]:
                    best = (score, location, scale)
            results.append({"image": str(path.relative_to(ROOT)), "score": best[0], "location_800px": best[1], "scale": best[2]})
    results.sort(key=lambda r: r["score"], reverse=True)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"pages_searched": len(results), "strongest_candidates": results[:10]}, indent=2))


if __name__ == "__main__":
    main()
