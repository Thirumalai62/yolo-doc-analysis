"""Run all v11 monitored gates for an existing epoch checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from active_head_pilot import evaluate_july
import evaluation_cache
import fixed080_acceptance as fixed
from fixed080_acceptance import file_sha256
from monitored_clslogit_pilot import run_challenge, run_fixed
from monitored_feature_finetune import (
    evaluate_target,
    load,
    preservation_failures,
    project_path,
    validate_stopping_config,
    verify_report_checkpoint_sha,
    write,
)


ROOT = Path(__file__).resolve().parents[1]


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def report_artifact(path: Path, report: dict) -> dict:
    return {"path": relative(path), "sha256": file_sha256(path), "totals": report["totals"]}


def evaluate(config_path: Path, checkpoint: Path, output: Path, fixed_predictions: Path | None = None) -> dict:
    if not checkpoint.is_file() or output.exists():
        raise RuntimeError("Checkpoint must exist and output directory must not exist")
    config = load(config_path)
    checkpoint_sha256 = file_sha256(checkpoint)
    output.mkdir(parents=True)
    summary = {
        "status": "evaluation_running",
        "config": relative(config_path),
        "config_sha256": file_sha256(config_path),
        "checkpoint": relative(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "reports": {},
    }
    write(output / "summary.json", summary)

    try:
        validate_stopping_config(config)
        july_path = project_path(config["july_manifest"])
        july_manifest = load(july_path)
        baseline_reports = {}
        for suite, artifact in config["baseline_reports"].items():
            path = project_path(artifact["path"])
            if not path.is_file() or file_sha256(path) != artifact["sha256"]:
                raise RuntimeError(f"Changed {suite} baseline report")
            baseline_reports[suite] = load(path)
            verify_report_checkpoint_sha(suite, baseline_reports[suite], config["starting_weights_sha256"])

        def checkpoint_unchanged() -> None:
            if file_sha256(checkpoint) != checkpoint_sha256:
                raise RuntimeError("Checkpoint changed during evaluation")

        def cached_evaluation(manifest_key: str, callback):
            checkpoint_unchanged()
            cache = evaluation_cache.IssueScopedCache(config, manifest_key, output / "evaluation_cache")
            try:
                return callback(cache.activate)
            finally:
                cache.close()

        def accept_report(suite: str, path: Path, report: dict) -> None:
            summary["reports"][suite] = report_artifact(path, report)
            verify_report_checkpoint_sha(suite, report, checkpoint_sha256)
            checkpoint_unchanged()

        fixed_path = output / "fixed080.json"
        if fixed_predictions:
            checkpoint_unchanged()
            predictions = load(fixed_predictions)
            if predictions.get("weights_sha256") != checkpoint_sha256:
                raise RuntimeError("Cached fixed predictions do not belong to the pinned checkpoint")
            fixed_report = fixed.check_candidate(
                project_path(config["fixed_manifest"]),
                predictions,
                {"path": relative(fixed_predictions), "sha256": file_sha256(fixed_predictions)},
                fixed_path,
                enforce=False,
                allow_unverified=False,
                verify_images=False,
            )
        else:
            fixed_report = cached_evaluation(
                "fixed_manifest",
                lambda before_page: run_fixed(checkpoint, project_path(config["fixed_manifest"]), fixed_path, before_page),
            )
        accept_report("fixed", fixed_path, fixed_report)

        challenge_path = output / "challenge.json"
        challenge_report = cached_evaluation(
            "challenge_manifest",
            lambda before_page: run_challenge(checkpoint, project_path(config["challenge_manifest"]), challenge_path, before_page),
        )
        accept_report("challenge", challenge_path, challenge_report)

        july_output = output / "july.json"
        july_report = cached_evaluation(
            "july_manifest",
            lambda before_page: evaluate_july(checkpoint, july_manifest, file_sha256(july_path), july_output, before_page),
        )
        accept_report("july", july_output, july_report)

        target_path = output / "target.json"
        target_report = evaluate_target(checkpoint, config, target_path, checkpoint_sha256)
        accept_report("target", target_path, target_report)

        failures, confidence = preservation_failures(config, fixed_report, challenge_report, july_report, baseline_reports)
        promoted = not failures and target_report["passed"]
        summary.update({
            "status": "promotion_passed" if promoted else "promotion_failed",
            "preservation_failures": failures,
            "confidence_preservation": confidence,
            "target_passed": target_report["passed"],
            "target": target_report["totals"],
            "promotion_passed": promoted,
        })
    except BaseException as error:
        summary["status"] = "evaluation_error"
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
        write(output / "summary.json", summary)
        raise
    write(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="regression/legal_notice_v11_feature_finetune_config.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixed-predictions", help="Reuse predictions generated with per-page hash verification")
    args = parser.parse_args()
    summary = evaluate(
        project_path(args.config),
        project_path(args.checkpoint),
        project_path(args.output),
        project_path(args.fixed_predictions) if args.fixed_predictions else None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
