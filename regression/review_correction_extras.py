"""Create review crops for unmatched predictions; never change acceptance labels."""
from pathlib import Path
import json
from PIL import Image
import argparse

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="legal_notice_v8_protected_scores_cpu")
    args = parser.parse_args()
    audit = ROOT / "output/regression_audit" / args.run
    output = audit / "extra_review"
    output.mkdir(exist_ok=True)
    records = []
    for suite in ("july", "fixed080"):
        report = json.loads((audit / f"{suite}.json").read_text())
        for page in report["pages"]:
            for number, prediction in enumerate(page["unreviewed_accepted"], 1):
                crop = output / f"{suite}_{page['issue']}_p{page['page']:04d}_{number}.png"
                with Image.open(ROOT / page["image"]) as image:
                    image.crop(tuple(round(v) for v in prediction["xyxy"])).save(crop)
                records.append({"suite": suite, "issue": page["issue"], "page": page["page"], "crop": str(crop), **prediction})
            if suite == "july":
                for excluded in page["excluded"]:
                    if excluded["accepted"]:
                        print("REMAINING EXCLUSION", excluded)
    (output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
