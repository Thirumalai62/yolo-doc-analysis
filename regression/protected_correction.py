"""Fit a minimal active-score correction with independent positive/negative guards.

Caches frozen features from TRAIN pages only, with the same rectangular
letterboxing used by detect. Fits only native final cv3 convolutions and saves a
normal single-class YOLO checkpoint. No additional inference component is added.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO, __version__
from torchvision.ops import box_iou

import fixed080_acceptance as fixed
import prepare_corrected_dataset as dataset_tools
from monitored_clslogit_pilot import DEVELOPMENT_ROOTS, run_fixed, run_challenge
from active_head_pilot import evaluate_july, stop_reasons
from diagnose_table_targets import page_tensor, labels

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset_v5_corrected"
OLD_AUDIT = ROOT / "output/regression_audit/legal_notice_v7_active_head_cpu_3_monitor"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def resolve_config(path):
    config = load(path)
    if "base_config" in config:
        config = {**resolve_config(ROOT / config["base_config"]), **config}
    manifest_value = config.get("correction_overlay_manifest")
    if manifest_value:
        manifest_path = ROOT / manifest_value
        assert fixed.file_sha256(manifest_path) == config["correction_overlay_manifest_sha256"]
        negative_regions = {name: list(regions) for name, regions in config["negative_regions_xyxyn"].items()}
        for page in load(manifest_path)["pages"]:
            assert page["image_name"] not in negative_regions
            negative_regions[page["image_name"]] = page["negative_regions_xyxyn"]
        config["negative_regions_xyxyn"] = negative_regions
    return config


def logit(p):
    return math.log(p / (1 - p))


def xywh_to_corners(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), dim=1)


def nonlinear_logits(x, block, final):
    """Native eval-mode 1x1 Conv/BN/SiLU plus final class convolution."""
    z = F.linear(x, block.conv.weight[:, :, 0, 0], block.conv.bias)
    bn = block.bn
    z = (z - bn.running_mean) * torch.rsqrt(bn.running_var + bn.eps)
    z = z * bn.weight + bn.bias
    return F.linear(F.silu(z), final.weight[:, :, 0, 0], final.bias).flatten()


def overlay_train_pages(config) -> list[dict]:
    pages = list(config.get("extra_train_pages", []))
    manifest_value = config.get("correction_overlay_manifest")
    if manifest_value:
        manifest_path = ROOT / manifest_value
        assert fixed.file_sha256(manifest_path) == config["correction_overlay_manifest_sha256"]
        manifest = load(manifest_path)
        assert manifest["status"] == "prepared_not_trained" and manifest["class_name"] == "legal_notice"
        for page in manifest["pages"]:
            pages.append({
                "issue": page["issue"],
                "image_name": page["image_name"],
                "image": page["image"],
                "label": page["label"],
                "split": "train",
                "reviewed_complete_labels": page["valid_notice_count"] > 0,
                "reviewed_no_valid_notices": page["valid_notice_count"] == 0,
                "image_sha256": page["image_sha256"],
                "label_sha256": page["label_sha256"],
            })
    return pages


def verify(config):
    assert __version__ == config["ultralytics_version"]
    assert fixed.file_sha256(ROOT / config["starting_weights"]) == config["starting_weights_sha256"]
    assert dataset_tools.tree_sha256(DATA, DEVELOPMENT_ROOTS) == config["development_sha256"]
    train_files = {p.name for p in (DATA / "images/train").glob("*.png")}
    assert set(config["positive_corrections"]) <= train_files
    extra = overlay_train_pages(config)
    extra_names = {p["image_name"] for p in extra}
    assert not extra_names & train_files and len(extra_names) == len(extra)
    assert set(config["negative_regions_xyxyn"]) <= train_files | extra_names
    excluded = set(load(ROOT / "regression/july_round5_review.json")["issues"])
    excluded |= {p["issue"] for p in load(ROOT / "regression/fixed080_manifest.json")["pages"]}
    assert not {name.rsplit("_page_", 1)[0] for name in train_files} & excluded
    evaluation_issues = {p.stem.rsplit("_page_", 1)[0]
                         for split in ("val", "test")
                         for p in (DATA / f"images/{split}").glob("*.png")}
    explicitly_eligible = set(config.get("eligible_extra_train_issues", []))
    for page in extra:
        assert page["split"] == "train"
        assert page.get("reviewed_no_valid_notices") is True or page.get("reviewed_complete_labels") is True
        assert not (page.get("reviewed_no_valid_notices") and page.get("reviewed_complete_labels"))
        assert page["issue"] not in excluded
        assert page["issue"] not in evaluation_issues or page["issue"] in explicitly_eligible
        assert page["issue"].startswith("khaleejtimes_2026-08-") or page["issue"] in explicitly_eligible
        assert page["image_name"].startswith(page["issue"] + "_page_")
        image_path = ROOT / page["image"]
        assert image_path.is_file()
        if page.get("image_sha256"):
            assert fixed.file_sha256(image_path) == page["image_sha256"]
        label_value = page.get("label")
        if page.get("reviewed_complete_labels"):
            assert label_value
        if label_value:
            label_path = ROOT / label_value
            assert label_path.is_file()
            if page.get("label_sha256"):
                assert fixed.file_sha256(label_path) == page["label_sha256"]
    return YOLO(str(ROOT / config["starting_weights"]))


def prepare(config, config_path):
    torch.set_num_threads(4)
    wrapper = verify(config)
    model = wrapper.model.eval()
    assert not model.end2end
    for p in model.parameters():
        p.requires_grad_(False)
    head = model.model[-1]
    captured = {}
    cache_pointwise = config.get("feature_layer") == "pre_pointwise"
    handles = [(head.cv3[i][1][1] if cache_pointwise else head.cv3[i][-1]).register_forward_pre_hook(
        lambda module, args, index=i: captured.__setitem__(index, args[0].detach())) for i in range(3)]
    root = ROOT / "output/regression_audit" / config["run_name"]
    assert not root.exists(), "Use a new run name; do not overwrite a frozen cache"
    root.mkdir(parents=True)
    write(root / "config.json", config)
    buckets = [{"x": [], "teacher": [], "kind": [], "group": []} for _ in range(3)]
    audit = []
    group_id = 0
    page_hashes = []
    sources = [(p.name, p, DATA / "labels/train" / p.with_suffix(".txt").name)
               for p in sorted((DATA / "images/train").glob("*.png"))]
    sources += [(p["image_name"], ROOT / p["image"], ROOT / p["label"] if p.get("label") else None)
                for p in overlay_train_pages(config)]
    for image_name, path, label_path in sources:
        t, transform = page_tensor(path, auto=True)
        gt = labels(label_path, transform, t.shape[-2:]) if label_path is not None else torch.zeros(0, 4)
        size = torch.tensor([t.shape[-1], t.shape[-2], t.shape[-1], t.shape[-2]])
        gt_boxes = xywh_to_corners(gt * size)
        with torch.no_grad():
            out = model(t)
            raw = out[1]["one2many"]
            decoded = xywh_to_corners(out[0][0, :4].T)
            z0 = raw["scores"][0, 0]
            probability = z0.sigmoid()
            features = torch.cat([captured[i][0].flatten(1).T for i in range(3)])
        overlaps = box_iou(decoded, gt_boxes) if len(gt) else torch.zeros(len(decoded), 0)
        positive_overlap = overlaps.max(dim=1).values if len(gt) else torch.zeros(len(decoded))
        kind = torch.zeros(len(decoded), dtype=torch.int64)  # 0=background trust, 1=valid trust, 2=table+, 3=negative
        groups = torch.full((len(decoded),), -1, dtype=torch.int64)
        positive_anchors = (positive_overlap >= 0.5) & (probability >= 0.05)
        kind[positive_anchors] = 1
        correction_mask = torch.zeros(len(decoded), dtype=torch.bool)
        page_targets = []
        for line in config["positive_corrections"].get(image_name, []):
            assert 1 <= line <= len(gt)
            suitable = overlaps[:, line - 1] >= 0.85
            assert suitable.any(), f"No complete frozen box for positive {path.name}:{line}"
            # Teach a handful of complete, existing high-score anchors per panel.
            ranked = torch.where(suitable)[0]
            ranked = ranked[probability[ranked].argsort(descending=True)[:4]]
            correction_mask |= overlaps[:, line - 1] >= 0.2
            kind[ranked] = 2
            groups[ranked] = group_id
            page_targets.append({"label_line": line, "group": group_id, "baseline_probability": float(probability[ranked].max())})
            group_id += 1
        w, h, ratio, left, top = transform
        for region in config["negative_regions_xyxyn"].get(image_name, []):
            region = torch.tensor(region) * torch.tensor([w, h, w, h]) * ratio + torch.tensor([left, top, left, top])
            if len(gt_boxes):
                assert not (box_iou(region[None], gt_boxes) > 0.01).any(), "Negative region overlaps a legal notice"
            intersection = (torch.minimum(decoded[:, 2:], region[2:]) - torch.maximum(decoded[:, :2], region[:2])).clamp(min=0).prod(1)
            area = (decoded[:, 2:] - decoded[:, :2]).clamp(min=0).prod(1).clamp(min=1)
            selected = (intersection / area >= 0.5) & (probability >= 0.01) & (positive_overlap < 0.1)
            assert selected.any(), f"No candidate in reviewed negative {path.name}"
            kind[selected] = 3
            groups[selected] = group_id
            page_targets.append({"negative_region": region.tolist(), "group": group_id, "baseline_probability": float(probability[selected].max())})
            group_id += 1
        # Keep all noticeable predictions and a deterministic sample of low-score
        # anchors at every scale. Exclude ambiguous shoulders of corrected tables
        # from background guards so they cannot fight the positive correction.
        keep = (probability >= 0.01) | (kind > 0)
        random = torch.Generator().manual_seed(0)
        keep[torch.randperm(len(keep), generator=random)[:256]] = True
        keep &= ~((kind < 2) & correction_mask)
        offset = 0
        counts = {}
        for scale in range(3):
            n = captured[scale].shape[-2] * captured[scale].shape[-1]
            index = torch.where(keep[offset:offset+n])[0] + offset
            x = torch.cat((features[index], torch.ones(len(index), 1)), dim=1).float()
            layer = head.cv3[scale][-1]
            weight = torch.cat((layer.weight.detach().flatten(), layer.bias.detach()))
            reconstructed = nonlinear_logits(x[:, :-1], head.cv3[scale][1][1], layer) if cache_pointwise else x @ weight
            assert torch.allclose(reconstructed, z0[index], atol=1e-4, rtol=1e-5), "Feature cache does not reproduce active logits"
            for key, value in (("x", x), ("teacher", z0[index]), ("kind", kind[index]), ("group", groups[index])):
                buckets[scale][key].append(value.clone())
            offset += n
        for k in range(4):
            counts[str(k)] = int(((kind == k) & keep).sum())
        audit.append({"image": image_name, "anchors": counts, "targets": page_targets})
        page_hashes.append({"image": image_name, "source": fixed.relative_path(path), "image_sha256": fixed.file_sha256(path),
                            "label_sha256": fixed.file_sha256(label_path) if label_path else None,
                            "valid_label_count": len(gt), "reviewed_empty_label": len(gt) == 0})
        print(f"cached {image_name}: {int(keep.sum())} anchors", flush=True)
    for handle in handles:
        handle.remove()
    cache = [{key: torch.cat(value) for key, value in bucket.items()} for bucket in buckets]
    torch.save(cache, root / "train_features.pt")
    record = {"config_sha256": fixed.file_sha256(config_path), "weights_sha256": config["starting_weights_sha256"],
              "cache_sha256": fixed.file_sha256(root / "train_features.pt"), "pages": audit, "source_hashes": page_hashes,
              "correction_groups": group_id, "padding": "same rectangular LetterBox as detect", "device": "cpu"}
    write(root / "cache_manifest.json", record)
    print(f"Prepared {group_id} reviewed correction groups from train only")


def objective(delta, cache, config, override_logits=None, ridge_override=None):
    positive, background, correction = [], [], []
    for scale, bucket in enumerate(cache):
        z = override_logits[scale] if override_logits is not None else bucket["teacher"] + bucket["x"] @ delta[scale]
        teacher, kind = bucket["teacher"], bucket["kind"]
        pos = kind == 1
        if pos.any():
            # Preserve high-score positives rather than lowering them toward an
            # IoU-quality target. Small numerical drift is allowed, not collapse.
            if config.get("guard_mode") == "decision_margin":
                lower = torch.where(teacher[pos] >= logit(0.8), torch.full_like(teacher[pos], logit(0.85)), teacher[pos])
                upper = teacher[pos] + 2.0
            else:
                lower = teacher[pos] - config["positive_logit_tolerance"]
                lower = torch.where(teacher[pos] >= logit(0.8), lower.clamp(min=logit(0.805)), lower)
                upper = teacher[pos] + config["positive_logit_tolerance"]
            positive.append(torch.relu(lower - z[pos]).square() + torch.relu(z[pos] - upper).square())
        bg = kind == 0
        if bg.any():
            if config.get("guard_mode") == "decision_margin":
                ceiling = torch.where(teacher[bg] < logit(0.8), torch.full_like(teacher[bg], logit(0.75)), teacher[bg])
                background.append(torch.relu(z[bg] - ceiling).square())
            else:
                background.append(torch.relu((z[bg] - teacher[bg]).abs() - config["background_logit_tolerance"]).square())
        for group in bucket["group"].unique():
            if group < 0:
                continue
            select = bucket["group"] == group
            if (kind[select] == 2).all():
                errors = torch.relu(logit(config["target_table_probability"]) - z[select]).square()
            else:
                errors = torch.relu(z[select] - logit(config["target_negative_probability"])).square()
            correction.append(errors.max() if (kind[select] == 3).all() and config.get("negative_pooling") == "max" else errors.mean())
    zero = delta.sum() * 0
    pos_loss = torch.cat(positive).mean() if positive else zero
    bg_loss = torch.cat(background).mean() if background else zero
    correct_loss = torch.stack(correction).mean() if correction else zero
    ridge = ridge_override if ridge_override is not None else delta.square().sum()
    total = correct_loss + config["guard_weight"] * (pos_loss + bg_loss) + config["ridge_weight"] * ridge
    return total, {"correction": float(correct_loss.detach()), "positive_guard": float(pos_loss.detach()),
                   "background_guard": float(bg_loss.detach()), "ridge": float(ridge.detach())}


def fit(config, config_path):
    torch.set_num_threads(4)
    assert config.get("feature_layer") != "pre_pointwise", "Use protected_nonlinear_correction.py for this cache"
    model = verify(config)
    root = ROOT / "output/regression_audit" / config["run_name"]
    cache_root = ROOT / "output/regression_audit" / config.get("feature_cache_run", config["run_name"])
    manifest = load(cache_root / "cache_manifest.json")
    if cache_root == root:
        assert manifest["config_sha256"] == fixed.file_sha256(config_path)
    else:
        source_config = load(cache_root / "config.json")
        for key in ("starting_weights_sha256", "development_sha256", "positive_corrections", "negative_regions_xyxyn",
                    "extra_train_pages", "correction_overlay_manifest", "correction_overlay_manifest_sha256"):
            assert source_config.get(key) == config.get(key), f"Cache mismatch: {key}"
        if root.exists():
            assert load(root / "config.json") == config and not (root / "fit_report.json").exists(), "Output already contains a fit"
        else:
            root.mkdir(parents=True)
        write(root / "config.json", config)
        write(root / "cache_reference.json", {"source": str(cache_root.relative_to(ROOT)), "manifest_sha256": fixed.file_sha256(cache_root / "cache_manifest.json")})
    assert manifest["cache_sha256"] == fixed.file_sha256(cache_root / "train_features.pt")
    cache = torch.load(cache_root / "train_features.pt", map_location="cpu", weights_only=True)
    run = ROOT / "runs" / config["run_name"]
    assert not run.exists(), "Run exists; refusing to overwrite"
    (run / "weights").mkdir(parents=True)
    delta = torch.nn.Parameter(torch.zeros(3, 257))
    optimizer = torch.optim.LBFGS([delta], lr=1.0, max_iter=config["max_iterations"], history_size=30, line_search_fn="strong_wolfe")
    history = []

    def closure():
        optimizer.zero_grad()
        loss, parts = objective(delta, cache, config)
        assert torch.isfinite(loss)
        loss.backward()
        history.append({"loss": float(loss.detach()), **parts})
        return loss

    optimizer.step(closure)
    assert torch.isfinite(delta).all() and delta.abs().max() > 0
    target_scores = []
    for scale, bucket in enumerate(cache):
        z = bucket["teacher"] + bucket["x"] @ delta[scale].detach()
        for group in bucket["group"].unique():
            if group >= 0:
                select = bucket["group"] == group
                target_scores.append({"scale": scale, "group": int(group), "kind": int(bucket["kind"][select][0]),
                                      "before_max": float(bucket["teacher"][select].sigmoid().max()),
                                      "after_max": float(z[select].sigmoid().max())})
    original = {n: p.clone() for n, p in model.model.state_dict().items()}
    with torch.no_grad():
        for scale in range(3):
            conv = model.model.model[-1].cv3[scale][-1]
            conv.weight.add_(delta[scale, :-1].reshape_as(conv.weight))
            conv.bias.add_(delta[scale, -1])
    changed = [n for n, p in model.model.state_dict().items() if not torch.equal(original[n], p)]
    allowed = {f"model.23.cv3.{scale}.2.{suffix}" for scale in range(3) for suffix in ("weight", "bias")}
    assert set(changed) <= allowed and changed
    model.model.end2end = False
    model.model.criterion = None
    # Keep correction tensors at float32 to avoid rounding away a small learned update.
    checkpoint = {"model": deepcopy(model.model).float().eval(), "ema": None,
                  "train_args": {**model.overrides, "device": "cpu"}, "epoch": -1,
                  "correction": {"config": config, "cache_manifest_sha256": fixed.file_sha256(cache_root / "cache_manifest.json")}}
    torch.save(checkpoint, run / "weights/candidate.pt")
    reloaded = YOLO(str(run / "weights/candidate.pt"))
    assert not reloaded.model.end2end
    assert all(torch.equal(model.model.state_dict()[n], reloaded.model.state_dict()[n]) for n in original)
    write(root / "fit_report.json", {"status": "trained_not_yet_accepted", "history": history, "target_scores": target_scores,
                                    "changed_tensors": changed, "candidate_sha256": fixed.file_sha256(run / "weights/candidate.pt"),
                                    "config_sha256": fixed.file_sha256(config_path)})
    print(json.dumps({"first_loss": history[0], "last_loss": history[-1], "target_scores": target_scores}, indent=2))


def evaluate(config):
    verify(config)
    root = ROOT / "output/regression_audit" / config["run_name"]
    weights = ROOT / "runs" / config["run_name"] / "weights/candidate.pt"
    assert fixed.file_sha256(weights) == load(root / "fit_report.json")["candidate_sha256"]
    baseline_root = ROOT / config.get("preservation_baseline_audit", str(OLD_AUDIT.relative_to(ROOT)))
    baseline_july_path = baseline_root / ("july.json" if config.get("preservation_baseline_audit") else "baseline_july.json")
    baseline_july = load(baseline_july_path)
    assert baseline_july["weights_sha256"] == config["starting_weights_sha256"]
    july_manifest = load(baseline_root / "july_manifest.json")
    # A user may clear a detect output folder between reviews. Recover missing
    # renders into this audit only, and require exact equality with frozen hashes.
    import sys
    sys.path.insert(0, str(ROOT))
    from main import render_pdf
    for page in july_manifest["pages"]:
        image = ROOT / page["image"]
        if not image.exists():
            destination = root / "recovered_rendered" / page["issue"]
            recovered = destination / f"page_{page['page']:04d}.png"
            if not recovered.exists():
                render_pdf(ROOT / "input" / (page["issue"] + ".pdf"), destination, 200, {page["page"]})
            assert fixed.file_sha256(recovered) == page["image_sha256"], "Recovered render differs from frozen review"
            page["image"] = fixed.relative_path(recovered)
        else:
            assert fixed.file_sha256(image) == page["image_sha256"]
    write(root / "july_manifest.json", july_manifest)
    fixed_manifest = load(ROOT / "regression/fixed080_manifest.json")
    missing_by_issue = {}
    for page in fixed_manifest["pages"]:
        image = ROOT / page["image"]
        if image.exists():
            assert fixed.file_sha256(image) == page["image_sha256"], "Existing fixed-suite render differs from frozen review"
        else:
            missing_by_issue.setdefault(page["issue"], {"pages": set(), "destination": image.parent})["pages"].add(page["page"])
    for issue, missing in missing_by_issue.items():
        render_pdf(ROOT / "input" / (issue + ".pdf"), missing["destination"], 200, missing["pages"])
    for page in fixed_manifest["pages"]:
        assert fixed.file_sha256(ROOT / page["image"]) == page["image_sha256"], "Recovered fixed-suite render differs from frozen review"
    # Run targeted July recovery first, then every preservation check. Selection
    # is never based on aggregate mAP or on corrected errors alone.
    j = evaluate_july(weights, july_manifest, fixed.file_sha256(root / "july_manifest.json"), root / "july.json")
    f = run_fixed(weights, ROOT / "regression/fixed080_manifest.json", root / "fixed080.json")
    c = run_challenge(weights, ROOT / "regression/full_category_challenge_manifest.json", root / "challenge.json")
    failures = stop_reasons(f, c, j, baseline_july)
    if config.get("preservation_baseline_audit"):
        baseline_fixed = load(baseline_root / "fixed080.json")
        baseline_challenge = load(baseline_root / "challenge.json")
        if f["totals"]["excluded_accepted"] > baseline_fixed["totals"]["excluded_accepted"]:
            failures.append("fixed_exclusions_worse_than_r7")
        if c["totals"]["excluded_accepted_regions"] > baseline_challenge["totals"]["excluded_accepted_regions"]:
            failures.append("challenge_exclusions_worse_than_r7")
        if j["totals"]["excluded_accepted_regions"] > baseline_july["totals"]["excluded_accepted_regions"]:
            failures.append("july_exclusions_worse_than_r7")
    targeted_pass = not failures and j["passed"] and all(t["complete_accepted"] for t in j["recovery_targets"])
    report = {"targeted_pass": targeted_pass, "full_promotion_pass": targeted_pass and f["passed"] and c["passed"],
              "failures": failures, "july": j["totals"], "targets": j["recovery_targets"], "fixed": f["totals"], "challenge": c["totals"]}
    write(root / "acceptance.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "fit", "evaluate"))
    parser.add_argument("--config", default="regression/protected_correction_config.json")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = resolve_config(config_path)
    if args.command == "prepare":
        prepare(config, config_path)
    elif args.command == "fit":
        fit(config, config_path)
    else:
        evaluate(config)


if __name__ == "__main__":
    main()
