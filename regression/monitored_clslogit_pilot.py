"""Run the CPU-only classification-logit pilot with fixed-threshold monitoring."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys

import fixed080_acceptance as fixed_acceptance
import full_category_challenge as challenge
import prepare_corrected_dataset as dataset_tools


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEVELOPMENT_ROOTS = ("data.yaml", "images/train", "labels/train", "images/val", "labels/val")


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Required file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_file(path_value: str, expected_sha256: str, description: str) -> Path:
    path = project_path(path_value)
    if not path.is_file() or fixed_acceptance.file_sha256(path) != expected_sha256:
        raise SystemExit(f"{description} is missing or changed: {path}")
    return path


def fixed_minimum_valid_confidence(report: dict) -> float:
    confidences = [
        reference["accepted_match"]["confidence"]
        for page in report["pages"]
        for reference in page["references"]
        if reference["classification"] == "valid" and reference["accepted_match"] is not None
    ]
    return min(confidences) if confidences else 0.0


def validate_baselines(config: dict, fixed_manifest_path: Path, challenge_manifest_path: Path) -> None:
    fixed_report_path = verify_file(
        config["fixed_baseline_report"],
        config["fixed_baseline_report_sha256"],
        "Fixed v3 baseline report",
    )
    fixed_report = load_json(fixed_report_path)
    if fixed_report.get("manifest_sha256") != fixed_acceptance.file_sha256(fixed_manifest_path):
        raise SystemExit("Fixed v3 baseline uses a different preservation manifest.")
    fixed_candidate = fixed_report.get("candidate", {})
    if (
        fixed_candidate.get("weights_sha256") != config["starting_weights_sha256"]
        or not fixed_candidate.get("prediction_provenance_verified")
    ):
        raise SystemExit("Fixed v3 baseline checkpoint provenance is invalid.")
    fixed_totals = fixed_report.get("totals", {})
    stopping = config["stopping"]
    expected_fixed = {
        "valid_preserved": stopping["fixed_valid_required"],
        "valid_missed": 0,
        "excluded_accepted": stopping["fixed_baseline_excluded_accepted"],
        "critical_boundaries": stopping["critical_boundaries_required"],
        "critical_boundary_failures": 0,
        "unreviewed_accepted": 0,
    }
    for key, expected in expected_fixed.items():
        if fixed_totals.get(key) != expected:
            raise SystemExit(f"Unexpected fixed v3 baseline {key}: {fixed_totals.get(key)}")

    challenge_report_path = verify_file(
        config["challenge_baseline_report"],
        config["challenge_baseline_report_sha256"],
        "Challenge v3 baseline report",
    )
    challenge_report = load_json(challenge_report_path)
    if challenge_report.get("manifest_sha256") != fixed_acceptance.file_sha256(challenge_manifest_path):
        raise SystemExit("Challenge v3 baseline uses a different challenge manifest.")
    if challenge_report.get("candidate", {}).get("weights_sha256") != config["starting_weights_sha256"]:
        raise SystemExit("Challenge v3 baseline checkpoint provenance is invalid.")
    challenge_totals = challenge_report.get("totals", {})
    expected_challenge = {
        "valid_preserved": stopping["challenge_valid_required"],
        "valid_missed": 0,
        "excluded_accepted_regions": stopping["challenge_baseline_excluded_accepted"],
        "unreviewed_accepted": 0,
    }
    for key, expected in expected_challenge.items():
        if challenge_totals.get(key) != expected:
            raise SystemExit(f"Unexpected challenge v3 baseline {key}: {challenge_totals.get(key)}")


def expected_trainable_names(model, frozen_layers: list) -> list[str]:
    patterns = [f"model.{layer}." for layer in frozen_layers] + [".dfl"]
    return [name for name, _ in model.named_parameters() if not any(pattern in name for pattern in patterns)]


def preflight(config_path: Path, require_empty_outputs: bool = True) -> tuple[dict, dict]:
    config = load_json(config_path)
    if config.get("training_authorized") is not True:
        raise SystemExit("Training is not authorized by this pilot config.")

    import ultralytics
    import main as application

    if ultralytics.__version__ != config["ultralytics_version"]:
        raise SystemExit(
            f"Ultralytics must be {config['ultralytics_version']}; found {ultralytics.__version__}."
        )
    training = config["training"]
    required_training = {
        "epochs": 2,
        "image_size": 1280,
        "batch": 1,
        "device": "cpu",
        "workers": 0,
        "amp": False,
        "optimizer": "SGD",
        "learning_rate": 0.0001,
        "final_learning_rate_fraction": 1.0,
        "weight_decay": 0.0,
        "nominal_batch_size": 1,
        "warmup_epochs": 0.0,
        "save_period": 1,
    }
    for key, expected in required_training.items():
        if training.get(key) != expected:
            raise SystemExit(f"Training setting {key} must be {expected!r}; found {training.get(key)!r}.")

    data_path = verify_file(config["dataset"], config["data_yaml_sha256"], "Dataset YAML")
    dataset_root = data_path.parent
    development_hash = dataset_tools.tree_sha256(dataset_root, DEVELOPMENT_ROOTS)
    if development_hash != config["development_split_sha256"]:
        raise SystemExit("Frozen train/validation content has changed.")
    dataset_release = load_json(project_path(config["dataset_release"]))
    if dataset_release.get("dataset_sha256") != config["dataset_sha256"]:
        raise SystemExit("Dataset release does not attest to the configured corrected dataset.")

    weights_path = verify_file(
        config["starting_weights"], config["starting_weights_sha256"], "Starting v3 checkpoint"
    )
    fixed_manifest_path = verify_file(
        config["fixed_manifest"], config["fixed_manifest_sha256"], "Fixed preservation manifest"
    )
    fixed_manifest = load_json(fixed_manifest_path)
    fixed_acceptance.validate_manifest(fixed_manifest)
    challenge_manifest_path = verify_file(
        config["challenge_manifest"], config["challenge_manifest_sha256"], "Challenge manifest"
    )
    challenge_manifest = load_json(challenge_manifest_path)
    challenge.validate_manifest(challenge_manifest_path, challenge_manifest)
    validate_baselines(config, fixed_manifest_path, challenge_manifest_path)

    development_image_hashes = {
        fixed_acceptance.file_sha256(path)
        for split in ("train", "val")
        for path in (dataset_root / "images" / split).rglob("*")
        if path.is_file()
    }
    fixed_image_hashes = {page["image_sha256"] for page in fixed_manifest["pages"]}
    if development_image_hashes & fixed_image_hashes:
        raise SystemExit("September preservation images overlap the development dataset.")

    model = application.load_yolo(weights_path)
    application.require_legal_notice_model(model)
    trainable_names = expected_trainable_names(model.model, training["frozen_layers"])
    head = model.model.model[-1]
    active_prefix = f"model.{head.i}.{'one2one_cv3' if model.model.end2end else 'cv3'}."
    if not any(name.startswith(active_prefix) for name in trainable_names):
        raise SystemExit(
            f"Freeze plan does not train the inference-active head ({active_prefix}). "
            "Use the corrected regression/active_head_pilot.py trainer."
        )
    if trainable_names != config["expected_trainable_parameters"]:
        raise SystemExit(f"Freeze plan resolves to unexpected trainable parameters: {trainable_names}")
    parameter_count = sum(
        parameter.numel()
        for name, parameter in model.model.named_parameters()
        if name in set(trainable_names)
    )
    if parameter_count != config["expected_trainable_parameter_count"]:
        raise SystemExit(f"Expected 771 trainable parameters; resolved {parameter_count}.")
    del model
    gc.collect()

    run_dir = ROOT / "runs" / config["run_name"]
    monitor_root = project_path(config["monitor_output"])
    if require_empty_outputs and (run_dir.exists() or monitor_root.exists()):
        existing = run_dir if run_dir.exists() else monitor_root
        raise SystemExit(f"Pilot output already exists: {existing}. Refusing to overwrite it.")

    summary = {
        "status": "preflight_passed",
        "training_started": False,
        "config": relative_path(config_path),
        "config_sha256": fixed_acceptance.file_sha256(config_path),
        "dataset": relative_path(data_path),
        "development_split_sha256": development_hash,
        "starting_weights": relative_path(weights_path),
        "fixed_manifest": relative_path(fixed_manifest_path),
        "challenge_manifest": relative_path(challenge_manifest_path),
        "run_directory": relative_path(run_dir),
        "monitor_output": relative_path(monitor_root),
        "device": training["device"],
        "trainable_parameters": trainable_names,
        "trainable_parameter_count": parameter_count,
        "training": training,
    }
    return config, summary


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_fixed(checkpoint: Path, manifest_path: Path, output_path: Path, before_page=None) -> dict:
    manifest = load_json(manifest_path)
    predictions = fixed_acceptance.run_candidate(checkpoint, manifest, before_page=before_page)
    prediction_path = output_path.with_suffix(".predictions.json")
    write_json(prediction_path, predictions)
    artifact = {
        "path": relative_path(prediction_path),
        "sha256": fixed_acceptance.file_sha256(prediction_path),
    }
    return fixed_acceptance.check_candidate(
        manifest_path,
        predictions,
        artifact,
        output_path,
        enforce=False,
        allow_unverified=False,
        verify_images=False,
    )


def run_challenge(checkpoint: Path, manifest_path: Path, output_path: Path, before_page=None) -> dict:
    manifest = load_json(manifest_path)
    predictions = challenge.run_predictions(checkpoint, manifest, before_page=before_page)
    predictions["manifest_sha256"] = fixed_acceptance.file_sha256(manifest_path)
    prediction_path = output_path.with_suffix(".predictions.json")
    write_json(prediction_path, predictions)
    scored = challenge.score_predictions(manifest, predictions)
    report = {
        "suite": manifest["suite"],
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": relative_path(manifest_path),
        "manifest_sha256": fixed_acceptance.file_sha256(manifest_path),
        "candidate": {
            "weights": predictions["weights"],
            "weights_sha256": predictions["weights_sha256"],
            "predictions": relative_path(prediction_path),
            "predictions_sha256": fixed_acceptance.file_sha256(prediction_path),
        },
        "acceptance_confidence": manifest["acceptance_confidence"],
        "image_size": manifest["image_size"],
        "unavailable_categories": manifest["unavailable_categories"],
        **scored,
    }
    write_json(output_path, report)
    challenge.write_markdown(report, output_path.with_suffix(".md"))
    return report


def epoch_status(config: dict, fixed_report: dict, challenge_report: dict, previous: dict | None) -> dict:
    stopping = config["stopping"]
    fixed_totals = fixed_report["totals"]
    challenge_totals = challenge_report["totals"]
    minimum_confidence = fixed_minimum_valid_confidence(fixed_report)
    failures = []
    if fixed_totals["valid_preserved"] != stopping["fixed_valid_required"]:
        failures.append("fixed_valid_preservation")
    if fixed_totals["critical_boundary_failures"] != 0:
        failures.append("critical_boundary_preservation")
    if fixed_totals["unreviewed_accepted"] != 0:
        failures.append("fixed_unreviewed_accepted")
    if minimum_confidence < stopping["minimum_fixed_valid_confidence"]:
        failures.append("minimum_fixed_valid_confidence")
    if challenge_totals["valid_preserved"] != stopping["challenge_valid_required"]:
        failures.append("challenge_valid_preservation")
    if challenge_totals["unreviewed_accepted"] != 0:
        failures.append("challenge_unreviewed_accepted")
    fixed_exclusions = fixed_totals["excluded_accepted"]
    challenge_exclusions = challenge_totals["excluded_accepted_regions"]
    if fixed_exclusions > stopping["fixed_baseline_excluded_accepted"]:
        failures.append("fixed_exclusions_worse_than_v3")
    if challenge_exclusions > stopping["challenge_baseline_excluded_accepted"]:
        failures.append("challenge_exclusions_worse_than_v3")

    fixed_improved = fixed_exclusions < stopping["fixed_baseline_excluded_accepted"]
    challenge_improved = challenge_exclusions < stopping["challenge_baseline_excluded_accepted"]
    if failures:
        decision = "stop_preservation_failure"
    elif previous is None and fixed_improved and challenge_improved:
        decision = "stop_early_success"
    elif previous is None and not fixed_improved and not challenge_improved:
        decision = "stop_futility"
    elif previous is None:
        decision = "continue_one_suite_improved"
    elif (
        fixed_improved
        and challenge_improved
        and fixed_exclusions <= previous["fixed_excluded_accepted"]
        and challenge_exclusions <= previous["challenge_excluded_accepted"]
    ):
        decision = "completed_success"
    else:
        decision = "completed_no_joint_improvement"
    return {
        "decision": decision,
        "stop_reasons": failures,
        "minimum_fixed_valid_confidence": minimum_confidence,
        "fixed_valid_preserved": fixed_totals["valid_preserved"],
        "fixed_excluded_accepted": fixed_exclusions,
        "fixed_critical_boundaries_preserved": (
            fixed_totals["critical_boundaries"] - fixed_totals["critical_boundary_failures"]
        ),
        "fixed_unreviewed_accepted": fixed_totals["unreviewed_accepted"],
        "challenge_valid_preserved": challenge_totals["valid_preserved"],
        "challenge_excluded_accepted": challenge_exclusions,
        "challenge_unreviewed_accepted": challenge_totals["unreviewed_accepted"],
    }


def launch_training(config_path: Path, config: dict, preflight_summary: dict) -> None:
    import main as application

    training = config["training"]
    data_path = project_path(config["dataset"])
    weights_path = project_path(config["starting_weights"])
    fixed_manifest_path = project_path(config["fixed_manifest"])
    challenge_manifest_path = project_path(config["challenge_manifest"])
    monitor_root = project_path(config["monitor_output"])
    monitor_root.mkdir(parents=True)
    summary_path = monitor_root / "monitor_summary.json"
    monitor = {
        **preflight_summary,
        "status": "training",
        "training_started": True,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "epochs": [],
    }
    write_json(summary_path, monitor)

    def on_pretrain_routine_end(trainer) -> None:
        trainable = [name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad]
        count = sum(parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad)
        optimizer_parameters = {
            id(parameter)
            for group in trainer.optimizer.param_groups
            for parameter in group["params"]
        }
        trainable_parameters = {
            id(parameter) for parameter in trainer.model.parameters() if parameter.requires_grad
        }
        if trainable != config["expected_trainable_parameters"]:
            raise RuntimeError(f"Trainer resolved unexpected trainable parameters: {trainable}")
        if count != config["expected_trainable_parameter_count"]:
            raise RuntimeError(f"Trainer resolved {count} trainable parameters instead of 771.")
        if not trainable_parameters <= optimizer_parameters:
            raise RuntimeError("Optimizer is missing one or more trainable parameters.")
        if trainer.accumulate != 1:
            raise RuntimeError(f"Optimizer accumulation must be 1; found {trainer.accumulate}.")
        if trainer.device.type != "cpu":
            raise RuntimeError(f"Pilot must run on CPU; found {trainer.device.type}.")
        monitor["trainer_preflight"] = {
            "device": trainer.device.type,
            "optimizer": type(trainer.optimizer).__name__,
            "optimizer_includes_frozen_parameters": optimizer_parameters != trainable_parameters,
            "accumulate": trainer.accumulate,
            "trainable_parameters": trainable,
            "trainable_parameter_count": count,
        }
        write_json(summary_path, monitor)

    def on_model_save(trainer) -> None:
        epoch_index = int(trainer.epoch)
        epoch_number = epoch_index + 1
        checkpoint = Path(trainer.wdir) / f"epoch{epoch_index}.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"Expected checkpoint was not saved: {checkpoint}")
        fixed_output = monitor_root / f"epoch_{epoch_number:02d}_fixed080.json"
        challenge_output = monitor_root / f"epoch_{epoch_number:02d}_challenge.json"
        fixed_report = run_fixed(checkpoint, fixed_manifest_path, fixed_output)
        challenge_report = run_challenge(checkpoint, challenge_manifest_path, challenge_output)
        previous = monitor["epochs"][-1] if monitor["epochs"] else None
        status = epoch_status(config, fixed_report, challenge_report, previous)
        metrics = {
            key: float(value)
            for key, value in trainer.metrics.items()
            if isinstance(value, (int, float)) or hasattr(value, "item")
        }
        epoch_record = {
            "epoch": epoch_number,
            "checkpoint": relative_path(checkpoint),
            "checkpoint_sha256": fixed_acceptance.file_sha256(checkpoint),
            "fixed_report": relative_path(fixed_output),
            "challenge_report": relative_path(challenge_output),
            "validation_metrics": metrics,
            **status,
        }
        monitor["epochs"].append(epoch_record)
        monitor["status"] = status["decision"]
        if status["decision"] != "continue_one_suite_improved":
            trainer.stop = True
        write_json(summary_path, monitor)
        gc.collect()

    try:
        model = application.load_yolo(weights_path)
        application.require_legal_notice_model(model)
        model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
        model.add_callback("on_model_save", on_model_save)
        results = model.train(
            data=str(data_path),
            epochs=training["epochs"],
            imgsz=training["image_size"],
            batch=training["batch"],
            device=training["device"],
            workers=training["workers"],
            cache=training["cache"],
            pretrained=training["pretrained"],
            resume=training["resume"],
            amp=training["amp"],
            compile=training["compile"],
            fraction=training["fraction"],
            single_cls=training["single_class"],
            freeze=training["frozen_layers"],
            plots=training["plots"],
            project=str(ROOT / "runs"),
            name=config["run_name"],
            exist_ok=False,
            optimizer=training["optimizer"],
            lr0=training["learning_rate"],
            lrf=training["final_learning_rate_fraction"],
            momentum=training["momentum"],
            weight_decay=training["weight_decay"],
            nbs=training["nominal_batch_size"],
            warmup_epochs=training["warmup_epochs"],
            warmup_momentum=training["warmup_momentum"],
            warmup_bias_lr=training["warmup_bias_lr"],
            mosaic=training["mosaic"],
            scale=training["scale"],
            translate=training["translate"],
            fliplr=training["horizontal_flip"],
            flipud=training["vertical_flip"],
            degrees=training["degrees"],
            shear=training["shear"],
            perspective=training["perspective"],
            mixup=training["mixup"],
            cutmix=training["cutmix"],
            copy_paste=training["copy_paste"],
            bgr=training["bgr"],
            hsv_h=training["hsv_h"],
            hsv_s=training["hsv_s"],
            hsv_v=training["hsv_v"],
            box=training["box_loss_gain"],
            cls=training["class_loss_gain"],
            dfl=training["dfl_loss_gain"],
            dropout=training["dropout"],
            rect=training["rectangular_batches"],
            cos_lr=training["cosine_learning_rate"],
            multi_scale=training["multi_scale"],
            close_mosaic=training["close_mosaic"],
            save=True,
            save_period=training["save_period"],
            val=True,
            split="val",
            iou=training["validation_iou"],
            max_det=training["max_detections"],
            patience=training["patience"],
            seed=training["seed"],
            deterministic=training["deterministic"],
        )
    except BaseException as error:
        monitor["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        monitor["status"] = "training_error"
        monitor["error"] = {"type": type(error).__name__, "message": str(error)}
        write_json(summary_path, monitor)
        raise
    monitor["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    monitor["training_results"] = relative_path(Path(results.save_dir))
    if monitor["status"] == "training":
        monitor["status"] = "completed_without_epoch_monitor"
    write_json(summary_path, monitor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="regression/clslogit_pilot_config.json")
    parser.add_argument("--start-training", action="store_true")
    args = parser.parse_args()
    config_path = project_path(args.config)
    config, summary = preflight(config_path)
    if args.start_training:
        launch_training(config_path, config, summary)
    else:
        print(json.dumps(summary, indent=2))
        print("Preflight passed. Training was not started.")


if __name__ == "__main__":
    main()
