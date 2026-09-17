"""Finalize the v10 correction-data preflight without fitting model weights."""
from __future__ import annotations

import json
from pathlib import Path

from fixed080_acceptance import file_sha256
from protected_correction import ROOT, load, resolve_config


CONFIG_PATH = ROOT / "regression/legal_notice_v10_targeted_correction_config.json"


def main() -> None:
    config = resolve_config(CONFIG_PATH)
    audit = ROOT / "output/regression_audit" / config["run_name"]
    cache = load(audit / "cache_manifest.json")
    overlay_path = ROOT / config["correction_overlay_manifest"]
    overlay = load(overlay_path)
    baseline = load(audit / "baseline_targeted_acceptance.json")

    assert config["training_authorization_required"] is True
    assert config["training_authorized"] is False
    assert cache["config_sha256"] == file_sha256(CONFIG_PATH)
    assert cache["weights_sha256"] == config["starting_weights_sha256"]
    assert file_sha256(overlay_path) == config["correction_overlay_manifest_sha256"]
    assert baseline["weights_sha256"] == config["starting_weights_sha256"]
    assert baseline["manifest_sha256"] == config["correction_overlay_manifest_sha256"]
    assert baseline["totals"]["valid_expected"] == baseline["totals"]["valid_accepted"] == 21
    assert baseline["totals"]["negative_regions"] == 8
    assert baseline["totals"]["unmatched_accepted"] == 0

    overlay_names = {page["image_name"] for page in overlay["pages"]}
    cached_pages = {page["image"]: page for page in cache["pages"]}
    assert overlay_names <= cached_pages.keys()
    correction_targets = []
    for page in overlay["pages"]:
        cached = cached_pages[page["image_name"]]
        assert cached["anchors"]["3"] > 0
        if page["valid_notice_count"]:
            assert cached["anchors"]["1"] > 0
        correction_targets.extend({
            "issue": page["issue"],
            "page": page["page"],
            "category": page["category"],
            "group": target["group"],
            "baseline_probability": target["baseline_probability"],
        } for target in cached["targets"] if "negative_region" in target)
    assert len(correction_targets) == 8

    report = {
        "status": "ready_for_training_authorization",
        "training_started": False,
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": file_sha256(CONFIG_PATH),
        "starting_weights": config["starting_weights"],
        "starting_weights_sha256": config["starting_weights_sha256"],
        "correction_overlay": config["correction_overlay_manifest"],
        "correction_overlay_sha256": config["correction_overlay_manifest_sha256"],
        "data": {
            **overlay["totals"],
            "base_training_pages": 67,
            "inherited_reviewed_extra_pages": 3,
            "total_cached_pages": len(cache["pages"]),
            "correction_groups": cache["correction_groups"],
        },
        "baseline_targeted_gate": {
            "valid_expected": 21,
            "valid_accepted": 21,
            "negative_regions": 8,
            "negative_margin_passed": baseline["totals"]["negative_margin_passed"],
            "excluded_accepted": baseline["totals"]["excluded_accepted"],
            "unmatched_accepted": 0,
            "report": str((audit / "baseline_targeted_acceptance.json").relative_to(ROOT)),
        },
        "correction_targets": correction_targets,
        "trainable_scope": config["trainable_scope"],
        "optimizer": config["optimizer"],
        "promotion_gates": config["promotion_gates"],
        "authorization": {"required": True, "authorized": False},
    }
    output = audit / "preflight_summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "training_started": False,
                      "config_sha256": report["config_sha256"], "output": str(output.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
