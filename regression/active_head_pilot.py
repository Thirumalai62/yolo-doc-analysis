"""Train the inference-active head and reject regressions on three evaluation suites.

The one-to-many branch is selected explicitly. Neither source checkpoints nor
the dataset is modified. All July examples remain evaluation-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys

import torch
from ultralytics import YOLO, __version__
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss

import fixed080_acceptance as fixed
import full_category_challenge as challenge
import prepare_corrected_dataset as dataset_tools
from monitored_clslogit_pilot import DEVELOPMENT_ROOTS, run_fixed, run_challenge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class ActiveHeadLoss:
    """Use native one-to-many assignment/loss, including its logged loss items."""
    def __init__(self, model):
        self.native = v8DetectionLoss(model, tal_topk=10)

    def __call__(self, predictions, batch):
        predictions = self.native.parse_output(predictions)
        return self.native(predictions["one2many"], batch)


class ActiveHeadTrainer(DetectionTrainer):
    def _model_train(self):
        super()._model_train()
        self.model.end2end = False
        # Keep all running statistics stable; active BN affine parameters may learn.
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

    def optimizer_step(self):
        if not getattr(self, "verified_update", False):
            active = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
            gradients = [p.grad for p in active.values() if p.grad is not None]
            require(gradients and all(torch.isfinite(g).all() for g in gradients), "Missing/nonfinite active gradients")
            gradient_norm = sum(float(g.detach().square().sum()) for g in gradients) ** 0.5
            require(gradient_norm > 0, "Active head gradients are zero")
            require(all(p.grad is None for p in self.model.parameters() if not p.requires_grad), "Frozen gradient detected")
            before = {n: p.detach().clone() for n, p in active.items()}
            super().optimizer_step()
            changed = [n for n, p in active.items() if not torch.equal(before[n], p.detach())]
            after_scores = probe_scores(self.model, self.probe)
            delta = float((after_scores - self.initial_probe_scores).abs().max())
            require(changed and delta > 0, "Optimizer update did not affect active inference scores")
            self.update_evidence = {"gradient_norm": gradient_norm, "changed_tensors": changed, "raw_score_max_delta": delta}
            self.verified_update = True
            self._model_train()
            write(self.audit_root / "first_update_verification.json", self.update_evidence)
            print(f"ACTIVE HEAD VERIFIED: {len(changed)} tensors updated; raw-score delta={delta:.8f}")
        else:
            super().optimizer_step()


def probe_scores(model, image):
    model.eval()
    with torch.no_grad():
        result = model(image)
        return result[1]["one2many"]["scores"].detach().clone()


def preflight(config):
    require(__version__ == config["ultralytics_version"], "Ultralytics version changed")
    for key in ("starting_weights", "fixed_manifest", "challenge_manifest"):
        require(fixed.file_sha256(ROOT / config[key]) == config[key + "_sha256"], f"Changed {key}")
    root = (ROOT / config["dataset"]).parent
    require(dataset_tools.tree_sha256(root, DEVELOPMENT_ROOTS) == config["development_sha256"], "Development dataset changed")
    fixed_manifest = load(ROOT / config["fixed_manifest"])
    fixed.validate_manifest(fixed_manifest)
    challenge.validate_manifest(ROOT / config["challenge_manifest"], load(ROOT / config["challenge_manifest"]))
    review = load(ROOT / "regression/july_round5_review.json")
    split_issues = {}
    train_hashes = set()
    for split in ("train", "val"):
        images = sorted((root / "images" / split).glob("*.png"))
        split_issues[split] = {p.stem.rsplit("_page_", 1)[0] for p in images}
        require(not split_issues[split] & (set(review["issues"]) | challenge.FORBIDDEN_ISSUES), "Evaluation issue in development data")
        require(not any(i.endswith(("2026-09-09", "2026-09-10")) for i in split_issues[split]), "Preservation issue leakage")
        if split == "train":
            train_hashes = {fixed.file_sha256(p) for p in images}
    require(not split_issues["train"] & split_issues["val"], "Issue split overlap")
    require(not train_hashes & {p["image_sha256"] for p in fixed_manifest["pages"]}, "Preservation image overlap")
    model = YOLO(str(ROOT / config["starting_weights"]))
    require(model.names == {0: "legal_notice"} and not model.model.end2end, "Wrong model class/inference mode")
    active = {n: p for n, p in model.model.named_parameters() if n.startswith(config["trainable_prefix"])}
    require(len(active) == config["trainable_tensors"] and sum(p.numel() for p in active.values()) == config["trainable_parameters"], "Wrong active head architecture")
    return model, train_hashes


def build_july(train_hashes):
    from PIL import Image
    review_path = ROOT / "regression/july_round5_review.json"
    review = load(review_path)
    overrides = {(x["issue"], x["page"], x["detection_id"]): x for x in review["exclusions"]}
    pages, sources, found = [], [], set()
    for issue in review["issues"]:
        source = ROOT / "output/legal_notices" / issue / "detections.json"
        sources.append({"path": fixed.relative_path(source), "sha256": fixed.file_sha256(source)})
        for entry in load(source):
            image_path = ROOT / entry["page"]
            number = int(image_path.stem.split("_")[-1])
            image_hash = fixed.file_sha256(image_path)
            require(image_hash not in train_hashes, "July image overlaps training")
            with Image.open(image_path) as image:
                width, height = image.size
            page = {"issue": issue, "page": number, "newspaper": issue.split("_")[0],
                    "image": fixed.relative_path(image_path), "image_sha256": image_hash,
                    "width": width, "height": height, "valid": [], "excluded": []}
            for detection in entry["detections"]:
                key = (issue, number, detection["id"])
                excluded = key in overrides
                if excluded:
                    found.add(key)
                box = detection["bbox_xyxy"]
                reference = {"id": f"{issue}_p{number:04d}_review_{detection['id']:03d}",
                             "xyxy": box, "xywhn": challenge.xyxy_to_xywhn(box, width, height),
                             "category": overrides[key]["category"] if excluded else "valid_legal_notice",
                             "classification": "excluded" if excluded else "valid"}
                page["excluded" if excluded else "valid"].append(reference)
            for target in review["missed_valid"]:
                if target["issue"] == issue and target["page"] == number:
                    page["valid"].append({**target, "classification": "valid", "recovery_target": True,
                                          "xywhn": challenge.xyxy_to_xywhn(target["xyxy"], width, height)})
            pages.append(page)
    require(found == set(overrides), "Missing reviewed procurement references")
    return {"suite": "july_round5_preservation_and_recovery", "pages": pages,
            "review": fixed.relative_path(review_path), "review_sha256": fixed.file_sha256(review_path),
            "sources": sources, "notes": review["notes"], "acceptance_confidence": 0.8, "image_size": 1280,
            "valid_match_iou": 0.5, "excluded_prediction_overlap": 0.5,
            "strict_gates": {"valid_missed": 0, "excluded_accepted_regions": 0, "unreviewed_accepted": 0}}


def evaluate_july(weights, manifest, manifest_hash, output):
    model = YOLO(str(weights))
    require(not model.model.end2end, "Saved checkpoint changed inference mode")
    weight_hash = fixed.file_sha256(weights)
    records, targets = [], []
    for page in manifest["pages"]:
        image = ROOT / page["image"]
        require(fixed.file_sha256(image) == page["image_sha256"], "July image changed")
        result = model.predict(str(image), conf=0.8, imgsz=1280, device="cpu", verbose=False, save=False)[0]
        records.append({"image": page["image"], "predictions": [fixed.prediction_record(b) for b in result.boxes]})
        if any(r.get("recovery_target") for r in page["valid"]):
            diagnostic = model.predict(str(image), conf=0.05, imgsz=1280, device="cpu", verbose=False, save=False)[0]
            boxes = [fixed.prediction_record(b) for b in diagnostic.boxes]
            for target in (r for r in page["valid"] if r.get("recovery_target")):
                matches = [(challenge.iou(target["xyxy"], b["xyxy"]), b) for b in boxes]
                overlap, best = max(matches, key=lambda pair: pair[0], default=(0.0, None))
                confidence = best["confidence"] if best and overlap >= 0.5 else None
                targets.append({"id": target["id"], "iou": overlap, "confidence": confidence,
                                "complete_accepted": confidence is not None and confidence >= 0.8 and overlap >= 0.85})
    require(not model.predictor.model.end2end, "Normal prediction selected the wrong head")
    require(fixed.file_sha256(weights) == weight_hash, "Checkpoint changed during evaluation")
    predictions = {"pages": records, "weights_sha256": weight_hash, "manifest_sha256": manifest_hash,
                   "minimum_confidence": 0.8, "device": "cpu", "image_size": 1280, "end2end": False}
    write(output.with_suffix(".predictions.json"), predictions)
    report = {**challenge.score_predictions(manifest, predictions), "recovery_targets": targets,
              "weights_sha256": weight_hash, "manifest_sha256": manifest_hash,
              "predictions_sha256": fixed.file_sha256(output.with_suffix(".predictions.json"))}
    write(output, report)
    del model
    gc.collect()
    return report


def stop_reasons(fixed_report, challenge_report, july, baseline_july):
    reasons = []
    for name, report in (("fixed", fixed_report), ("challenge", challenge_report)):
        for metric in ("valid_missed", "unreviewed_accepted"):
            if report["totals"][metric]:
                reasons.append(f"{name}_{metric}")
    if not fixed_report["gate_results"]["prediction_provenance_verified"]:
        reasons.append("fixed_provenance")
    if fixed_report["totals"]["critical_boundary_failures"]:
        reasons.append("critical_boundary_failure")
    if fixed_report["totals"]["excluded_accepted"] > 17 or challenge_report["totals"]["excluded_accepted_regions"] > 7:
        reasons.append("exclusions_worse_than_start")
    baseline_ids = {r["reference_id"] for p in baseline_july["pages"] for r in p["valid_matches"]}
    actual_ids = {r["reference_id"] for p in july["pages"] for r in p["valid_matches"]}
    if baseline_ids - actual_ids:
        reasons.append("july_previously_valid_lost")
    if july["totals"]["unreviewed_accepted"]:
        reasons.append("july_unreviewed_accepted")
    return reasons


def launch(config_path, config, model, train_hashes):
    import cv2
    import numpy as np
    from ultralytics.data.augment import LetterBox
    run_dir = ROOT / "runs" / config["run_name"]
    audit = ROOT / "output/regression_audit" / (config["run_name"] + "_monitor")
    require(not run_dir.exists() and not audit.exists(), "Output exists; use a new run name")
    audit.mkdir(parents=True)
    write(audit / "config.json", config)
    manifest = build_july(train_hashes)
    write(audit / "july_manifest.json", manifest)
    manifest_hash = fixed.file_sha256(audit / "july_manifest.json")
    starting = ROOT / config["starting_weights"]
    baseline_july = evaluate_july(starting, manifest, manifest_hash, audit / "baseline_july.json")
    baseline_fixed = run_fixed(starting, ROOT / config["fixed_manifest"], audit / "baseline_fixed080.json")
    baseline_challenge = run_challenge(starting, ROOT / config["challenge_manifest"], audit / "baseline_challenge.json")
    require(not stop_reasons(baseline_fixed, baseline_challenge, baseline_july, baseline_july), "Starting checkpoint fails preservation preflight")
    initial_state = {n: p.detach().cpu().clone() for n, p in model.model.state_dict().items()}
    image = (ROOT / config["dataset"]).parent / "images/train" / config["probe_train_image"]
    im = LetterBox(new_shape=(1280, 1280), auto=False)(image=cv2.imread(str(image)))
    probe = torch.from_numpy(np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))).float()[None] / 255
    summary = {"status": "initializing", "config_sha256": fixed.file_sha256(config_path),
               "script_sha256": fixed.file_sha256(Path(__file__)), "epochs": [], "device": "cpu",
               "started_at_utc": datetime.now(timezone.utc).isoformat()}
    write(audit / "summary.json", summary)

    def setup(trainer):
        trainer.model.end2end = False
        trainer.ema.ema.end2end = False
        trainer.model.criterion = ActiveHeadLoss(trainer.model)
        # Validation must log the same active-branch loss as training.
        trainer.ema.ema.criterion = ActiveHeadLoss(trainer.ema.ema)
        actual = {n: p for n, p in trainer.model.named_parameters() if p.requires_grad}
        expected = {n for n, p in trainer.model.named_parameters() if n.startswith(config["trainable_prefix"])}
        require(set(actual) == expected and len(actual) == config["trainable_tensors"], "Wrong trainable branch")
        require(sum(p.numel() for p in actual.values()) == config["trainable_parameters"], "Wrong parameter count")
        require(trainer.device.type == "cpu" and trainer.accumulate == 1, "Device or accumulation mismatch")
        require(all(torch.equal(trainer.model.state_dict()[n].cpu(), p.float()) for n, p in initial_state.items()), "Initialization differs from chosen best.pt")
        trainer.audit_root, trainer.probe = audit, probe
        trainer.initial_probe_scores = probe_scores(trainer.model, probe)
        summary["trainer_preflight"] = {"trainable_names": sorted(actual), "parameters": sum(p.numel() for p in actual.values()), "end2end": False, "loss": "one2many_only"}
        summary["status"] = "training"
        write(audit / "summary.json", summary)

    def saved(trainer):
        number = int(trainer.epoch) + 1
        checkpoint = Path(trainer.wdir) / f"epoch{number - 1}.pt"
        require(getattr(trainer, "verified_update", False), "No verified active update")
        current = YOLO(str(checkpoint)).model
        require(not current.end2end, "Checkpoint mode mismatch")
        changed = [n for n, p in initial_state.items() if not torch.equal(current.state_dict()[n].cpu().half(), p.half())]
        require(changed and all(n.startswith(config["trainable_prefix"]) for n in changed), "Unexpected or absent checkpoint updates")
        delta = float((probe_scores(current, probe) - trainer.initial_probe_scores).abs().max())
        require(delta > 0, "Saved checkpoint does not affect active prediction scores")
        del current
        f = run_fixed(checkpoint, ROOT / config["fixed_manifest"], audit / f"epoch_{number:02d}_fixed080.json")
        c = run_challenge(checkpoint, ROOT / config["challenge_manifest"], audit / f"epoch_{number:02d}_challenge.json")
        j = evaluate_july(checkpoint, manifest, manifest_hash, audit / f"epoch_{number:02d}_july.json")
        reasons = stop_reasons(f, c, j, baseline_july)
        promoted = f["passed"] and c["passed"] and j["passed"] and all(t["complete_accepted"] for t in j["recovery_targets"])
        summary["epochs"].append({"epoch": number, "checkpoint": fixed.relative_path(checkpoint),
                                  "checkpoint_sha256": fixed.file_sha256(checkpoint), "changed_tensors": changed,
                                  "raw_score_max_delta": delta, "fixed": f["totals"], "challenge": c["totals"],
                                  "july": j["totals"], "recovery_targets": j["recovery_targets"],
                                  "stop_reasons": reasons, "promotion_passed": promoted})
        if reasons or promoted:
            trainer.stop = True
            summary["status"] = "stopped_preservation_failure" if reasons else "passed_all_gates"
        write(audit / "summary.json", summary)
        print(json.dumps(summary["epochs"][-1], indent=2))
        gc.collect()

    model.add_callback("on_pretrain_routine_end", setup)
    model.add_callback("on_model_save", saved)
    try:
        model.train(trainer=ActiveHeadTrainer, data=str(ROOT / config["dataset"]), project=str(ROOT / "runs"),
                    name=config["run_name"], exist_ok=False, **config["training"])
        if summary["status"] == "training":
            summary["status"] = "completed_no_passing_checkpoint"
        require(dataset_tools.tree_sha256((ROOT / config["dataset"]).parent, DEVELOPMENT_ROOTS) == config["development_sha256"], "Development data changed during training")
        require(fixed.file_sha256(starting) == config["starting_weights_sha256"], "Starting checkpoint changed")
    except BaseException as error:
        summary["status"] = "error"
        summary["error"] = repr(error)
        raise
    finally:
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(audit / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="regression/active_head_pilot_config.json")
    parser.add_argument("--start-training", action="store_true")
    args = parser.parse_args()
    path = ROOT / args.config
    config = load(path)
    model, hashes = preflight(config)
    if args.start_training:
        launch(path, config, model, hashes)
    else:
        july = build_july(hashes)
        print(json.dumps({"status": "preflight_passed", "end2end": model.model.end2end,
                          "trainable_parameters": config["trainable_parameters"], "july_pages": len(july["pages"]),
                          "july_valid": sum(len(p["valid"]) for p in july["pages"]),
                          "july_excluded": sum(len(p["excluded"]) for p in july["pages"])}, indent=2))


if __name__ == "__main__":
    main()
