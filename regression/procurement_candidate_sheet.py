"""Make a visual review sheet of eligible non-July Khaleej Times detections."""
from pathlib import Path
import json
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    val_issues = {p.stem.rsplit("_page_", 1)[0] for p in (ROOT / "dataset_v5_corrected/images/val").glob("*.png")}
    output = ROOT / "output/regression_audit/procurement_candidate_review"
    output.mkdir(exist_ok=True)
    entries = []
    for directory in sorted((ROOT / "output/legal_notices").glob("khaleejtimes_2026-08-*")):
        if directory.name in val_issues:
            continue
        report = directory / "detections.json"
        if not report.exists():
            continue
        for page in json.loads(report.read_text()):
            for detection in page["detections"]:
                path = ROOT / detection["crop"]
                if path.exists():
                    entries.append({"id": len(entries), "issue": directory.name, "image": page["page"], **detection})
                    entries[-1]["review_id"] = len(entries) - 1
    for offset in range(0, len(entries), 24):
        sheet = Image.new("RGB", (6 * 220, 4 * 280), "white")
        draw = ImageDraw.Draw(sheet)
        for i, entry in enumerate(entries[offset:offset+24]):
            with Image.open(ROOT / entry["crop"]) as im:
                im.thumbnail((210, 242))
                x, y = (i % 6) * 220, (i // 6) * 280
                sheet.paste(im, (x, y + 30))
                draw.text((x, y), f"{entry['review_id']} {entry['issue'][13:]}\n{Path(entry['image']).stem}", fill="black")
        sheet.save(output / f"sheet_{offset // 24:02d}.jpg")
    (output / "manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
    print(f"{len(entries)} candidates, {(len(entries)+23)//24} review sheets in {output}")


if __name__ == "__main__":
    main()
