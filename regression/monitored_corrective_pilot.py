"""Preflight or launch the fixed-0.80 monitored corrective training pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys

import fixed080_acceptance as acceptance
import prepare_corrected_dataset as dataset_tools


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REQUIRED_EARLY_STOP_GATES = {
    "valid_missed",
    "critical_boundary_failures",
    "unreviewed_accepted",
}


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


def preservation_status(report: dict, early_stop_gates: set[str]) -> dict:
    failures = [gate for gate in sorted(early_stop_gates) if report["totals"].get(gate) != 0]
    if not report["gate_results"].get("prediction_provenance_verified"):
        failures.append("prediction_provenance_verified")
    return {
        "preservation_passed": not failures,
        "stop_reasons": failures,
        "promotion_passed": report["passed"],
    }


def validate_baseline_report(report: dict, config: dict, manifest_path: Path) -> None:
    expected_weights_hash = config["starting_weights_sha256"]
    if report.get("manifest_sha256") != acceptance.file_sha256(manifest_path):
        raise SystemExit("Fresh v3 baseline report does not use the current regression manifest.")
    candidate = report.get("candidate", {})
    if candidate.get("weights_sha256") != expected_weights_hash:
        raise SystemExit("Fresh v3 baseline report does not use the configured starting checkpoint.")
    if not candidate.get("prediction_provenance_verified"):
        raise SystemExit("Fresh v3 baseline prediction provenance is not verified.")
    prediction_artifact = candidate.get("predictions", {})
    prediction_path = project_path(prediction_artifact.get("path", ""))
    if (
        not prediction_path.is_file()
        or acceptance.file_sha256(prediction_path) != prediction_artifact.get("sha256")
    ):
        raise SystemExit("Fresh v3 baseline prediction artifact is missing or changed.")
    totals = report.get("totals", {})
    if totals.get("valid_missed") != 0 or totals.get("critical_boundary_failures") != 0:
        raise SystemExit("Starting v3 checkpoint does not pass preservation and boundary preflight.")
    if totals.get("unreviewed_accepted") != 0:
        raise SystemExit("Starting v3 checkpoint has unreviewed accepted detections.")


def preflight(config_path: Path, require_empty_outputs: bool = True) -> tuple[dict, dict]:
    config = load_json(config_path)
    import ultralytics

    if ultralytics.__version__ != config["ultralytics_version"]:
        raise SystemExit(
            f"Ultralytics version must be {config['ultralytics_version']}; "
            f"found {ultralytics.__version__}."
        )
    training = config.get("training", {})
    if training.get("epochs") != 3 or training.get("save_period") != 1:
        raise SystemExit("The corrective pilot must run three epochs and save every epoch.")
    if training.get("learning_rate") != 0.00005:
        raise SystemExit("The corrective pilot learning rate must remain fixed at 0.00005.")
    early_stop_gates = set(config.get("early_stop_gates", []))
    if early_stop_gates != REQUIRED_EARLY_STOP_GATES:
        raise SystemExit(f"Early-stop gates must be exactly: {sorted(REQUIRED_EARLY_STOP_GATES)}")

    data_path = project_path(config["dataset"])
    dataset_root = data_path.parent
    if acceptance.file_sha256(data_path) != config["data_yaml_sha256"]:
        raise SystemExit("Corrected dataset data.yaml has changed.")
    if dataset_tools.tree_sha256(dataset_root) != config["dataset_sha256"]:
        raise SystemExit("Corrected dataset content does not match the pinned pilot dataset.")
    if (
        dataset_tools.tree_sha256(dataset_root, ("images/test", "labels/test"))
        != config["test_split_sha256"]
    ):
        raise SystemExit("Corrected dataset test split has changed.")

    weights_path = project_path(config["starting_weights"])
    if acceptance.file_sha256(weights_path) != config["starting_weights_sha256"]:
        raise SystemExit("Starting v3 checkpoint has changed.")
    manifest_path = project_path(config["regression_manifest"])
    if acceptance.file_sha256(manifest_path) != config["regression_manifest_sha256"]:
        raise SystemExit("Regression manifest has changed; review and update the pilot config explicitly.")
    manifest = load_json(manifest_path)
    acceptance.validate_manifest(manifest)
    regression_image_hashes = {page["image_sha256"] for page in manifest["pages"]}
    dataset_image_hashes = {
        acceptance.file_sha256(path)
        for path in (dataset_root / "images").rglob("*")
        if path.is_file()
    }
    overlap = regression_image_hashes & dataset_image_hashes
    if overlap:
        raise SystemExit(f"Regression suite overlaps the training dataset by {len(overlap)} image(s).")

    baseline_report_path = project_path(config["baseline_report"])
    if acceptance.file_sha256(baseline_report_path) != config["baseline_report_sha256"]:
        raise SystemExit("Fresh v3 baseline report has changed.")
    validate_baseline_report(load_json(baseline_report_path), config, manifest_path)

    run_dir = ROOT / "runs" / config["run_name"]
    monitor_root = project_path(config["monitor_output"])
    if require_empty_outputs and (run_dir.exists() or monitor_root.exists()):
        raise SystemExit(
            f"Pilot output already exists: {run_dir if run_dir.exists() else monitor_root}. "
            "Use a new reviewed config instead of overwriting it."
        )

    summary = {
        "status": "preflight_passed",
        "training_started": False,
        "config": relative_path(config_path),
        "dataset": relative_path(data_path),
        "starting_weights": relative_path(weights_path),
        "regression_manifest": relative_path(manifest_path),
        "run_directory": relative_path(run_dir),
        "monitor_output": relative_path(monitor_root),
        "acceptance_confidence": manifest["acceptance_confidence"],
        "regression_dataset_overlap_images": len(overlap),
        "early_stop_gates": sorted(early_stop_gates),
        "training": training,
    }
    return config, summary


def write_monitor_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def launch_training(config_path: Path, config: dict, preflight_summary: dict) -> None:
    import main as application

    training = config["training"]
    data_path = project_path(config["dataset"])
    weights_path = project_path(config["starting_weights"])
    manifest_path = project_path(config["regression_manifest"])
    monitor_root = project_path(config["monitor_output"])
    monitor_root.mkdir(parents=True)
    summary_path = monitor_root / "monitor_summary.json"
    monitor_summary = {
        **preflight_summary,
        "status": "training",
        "training_started": True,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "epochs": [],
        "stopped_early": False,
        "preservation_failed": False,
    }
    write_monitor_summary(summary_path, monitor_summary)

    def on_model_save(trainer) -> None:
        epoch_index = int(trainer.epoch)
        epoch_number = epoch_index + 1
        checkpoint = Path(trainer.wdir) / f"epoch{epoch_index}.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"Expected per-epoch checkpoint was not saved: {checkpoint}")
        report_path = monitor_root / f"epoch_{epoch_number:02d}_fixed080.json"
        candidate = acceptance.run_candidate(checkpoint, load_json(manifest_path))
        prediction_path = report_path.with_suffix(".predictions.json")
        prediction_path.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
        candidate_artifact = {
            "path": relative_path(prediction_path),
            "sha256": acceptance.file_sha256(prediction_path),
        }
        report = acceptance.check_candidate(
            manifest_path,
            candidate,
            candidate_artifact,
            report_path,
            enforce=False,
            allow_unverified=False,
        )
        status = preservation_status(report, set(config["early_stop_gates"]))
        monitor_summary["epochs"].append({
            "epoch": epoch_number,
            "checkpoint": relative_path(checkpoint),
            "checkpoint_sha256": acceptance.file_sha256(checkpoint),
            "report": relative_path(report_path),
            **status,
            "valid_preserved": report["totals"]["valid_preserved"],
            "valid_missed": report["totals"]["valid_missed"],
            "excluded_accepted": report["totals"]["excluded_accepted"],
            "critical_boundary_failures": report["totals"]["critical_boundary_failures"],
            "unreviewed_accepted": report["totals"]["unreviewed_accepted"],
        })
        if not status["preservation_passed"]:
            trainer.stop = True
            monitor_summary["status"] = "preservation_failed"
            monitor_summary["preservation_failed"] = True
            monitor_summary["stopped_early"] = epoch_number < training["epochs"]
        write_monitor_summary(summary_path, monitor_summary)
        gc.collect()

    try:
        model = application.load_yolo(weights_path)
        application.require_legal_notice_model(model)
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
            amp=training["amp"],
            compile=training["compile"],
            fraction=training["fraction"],
            single_cls=training["single_class"],
            freeze=training["frozen_layers"],
            distill_model=training["distillation_model"],
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
            auto_augment=training["auto_augment"],
            erasing=training["erasing"],
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
        monitor_summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if not monitor_summary["preservation_failed"]:
            monitor_summary["status"] = "training_error"
        monitor_summary["error"] = {"type": type(error).__name__, "message": str(error)}
        write_monitor_summary(summary_path, monitor_summary)
        raise
    monitor_summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    monitor_summary["training_results"] = relative_path(Path(results.save_dir))
    passing_epochs = [epoch["epoch"] for epoch in monitor_summary["epochs"] if epoch["promotion_passed"]]
    monitor_summary["passing_epochs"] = passing_epochs
    if monitor_summary["status"] == "training":
        monitor_summary["status"] = "completed_with_passing_epoch" if passing_epochs else "completed_no_passing_epoch"
    write_monitor_summary(summary_path, monitor_summary)
    if monitor_summary["preservation_failed"]:
        raise SystemExit("Training stopped after a fixed-0.80 preservation failure.")
    if not passing_epochs:
        raise SystemExit("Training completed, but no epoch passed all fixed-0.80 promotion gates.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="regression/corrective_pilot_config.json")
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--start-training",
        action="store_true",
        help="Explicitly authorize the three-epoch monitored training run",
    )
    action.add_argument(
        "--simulate-report",
        help="Show whether an existing acceptance report would trigger early stopping",
    )
    args = parser.parse_args()

    config_path = project_path(args.config)
    config, summary = preflight(config_path, require_empty_outputs=not args.simulate_report)
    if args.simulate_report:
        report = load_json(project_path(args.simulate_report))
        print(json.dumps(preservation_status(report, set(config["early_stop_gates"])), indent=2))
    elif args.start_training:
        launch_training(config_path, config, summary)
    else:
        print(json.dumps(summary, indent=2))
        print("Preflight passed. Training was not started; use --start-training only after explicit approval.")


if __name__ == "__main__":
    main()
