"""Fit the final native nonlinear score block using a frozen train-only cache."""
import argparse
from copy import deepcopy
from pathlib import Path
import torch
from ultralytics import YOLO

from protected_correction import ROOT, load, write, verify, nonlinear_logits, objective, resolve_config
import fixed080_acceptance as fixed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="regression/protected_correction_r4_config.json")
    parser.add_argument("--recover-report", action="store_true", help="Rebuild an interrupted run's report without training")
    parser.add_argument("--start-training", action="store_true", help="Explicitly authorize the frozen training run")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = resolve_config(config_path)
    if not args.recover_report and config.get("training_authorization_required") and not args.start_training:
        raise SystemExit("Training is not authorized; review the frozen config and rerun with --start-training.")
    assert config["feature_layer"] == "pre_pointwise"
    torch.set_num_threads(4)
    wrapper = verify(config)
    model = wrapper.model.eval()
    head = model.model[-1]
    audit = ROOT / "output/regression_audit" / config["run_name"]
    cache_root = ROOT / "output/regression_audit" / config.get("feature_cache_run", config["run_name"])
    manifest = load(cache_root / "cache_manifest.json")
    if cache_root == audit:
        assert fixed.file_sha256(config_path) == manifest["config_sha256"]
    else:
        source_config = load(cache_root / "config.json")
        for key in ("feature_layer", "starting_weights_sha256", "development_sha256", "positive_corrections",
                    "negative_regions_xyxyn", "extra_train_pages", "correction_overlay_manifest",
                    "correction_overlay_manifest_sha256"):
            assert source_config.get(key) == config.get(key)
        if not audit.exists():
            audit.mkdir(parents=True)
            write(audit / "config.json", config)
            write(audit / "cache_reference.json", {"path": str(cache_root.relative_to(ROOT)), "manifest_sha256": fixed.file_sha256(cache_root / "cache_manifest.json")})
        else:
            assert args.recover_report and load(audit / "config.json") == config
    assert fixed.file_sha256(cache_root / "train_features.pt") == manifest["cache_sha256"]
    cache = torch.load(cache_root / "train_features.pt", weights_only=True, map_location="cpu")
    run = ROOT / "runs" / config["run_name"]
    if args.recover_report:
        candidate = run / "weights/candidate.pt"
        assert candidate.is_file() and not (audit / "fit_report.json").exists()
        recovered = YOLO(str(candidate)).model.eval()
        recovered_head = recovered.model[-1]
        changed = [n for n, p in recovered.state_dict().items() if not torch.equal(model.state_dict()[n], p)]
        allowed = {f"model.23.cv3.{scale}.{suffix}" for scale in range(3) for suffix in (
            "1.1.conv.weight", "1.1.bn.weight", "1.1.bn.bias", "2.weight", "2.bias")}
        assert changed and set(changed) <= allowed
        with torch.no_grad():
            logits = [nonlinear_logits(bucket["x"][:, :-1], recovered_head.cv3[i][1][1], recovered_head.cv3[i][-1])
                      for i, bucket in enumerate(cache)]
            penalty = sum((recovered.state_dict()[n] - model.state_dict()[n]).square().sum() for n in changed)
            loss, parts = objective(torch.zeros(3, 257), cache, config, logits, penalty)
            target_scores = []
            for scale, (bucket, z) in enumerate(zip(cache, logits)):
                for group in bucket["group"].unique():
                    if group >= 0:
                        mask = bucket["group"] == group
                        target_scores.append({"scale": scale, "group": int(group), "kind": int(bucket["kind"][mask][0]),
                                              "before_max": float(bucket["teacher"][mask].sigmoid().max()),
                                              "after_max": float(z[mask].sigmoid().max())})
        write(audit / "fit_report.json", {"status": "recovered_after_interruption", "history_available": False,
                                          "final_loss": {"loss": float(loss), **parts}, "target_scores": target_scores,
                                          "changed_tensors": changed, "candidate_sha256": fixed.file_sha256(candidate)})
        print({"status": "report_recovered", "final_loss": float(loss), "candidate": str(candidate)})
        return
    assert not run.exists()
    (run / "weights").mkdir(parents=True)
    original = {n: p.detach().clone() for n, p in model.state_dict().items()}
    active = {}
    for name, parameter in model.named_parameters():
        learn = any(name.startswith(f"model.23.cv3.{scale}.{suffix}") for scale in range(3) for suffix in ("1.1.", "2."))
        parameter.requires_grad_(learn)
        if learn:
            active[name] = parameter
    assert len(active) == 15 and sum(p.numel() for p in active.values()) == 198915
    if config.get("warm_start_run"):
        warm_path = ROOT / "runs" / config["warm_start_run"] / "weights/candidate.pt"
        report = load(ROOT / "output/regression_audit" / config["warm_start_run"] / "fit_report.json")
        assert fixed.file_sha256(warm_path) == report["candidate_sha256"]
        warm = YOLO(str(warm_path)).model.state_dict()
        assert all(torch.equal(warm[n], p) for n, p in original.items() if n not in active)
        with torch.no_grad():
            for name, p in active.items():
                p.copy_(warm[name])
    optimizer = torch.optim.LBFGS(
        list(active.values()), lr=1, max_iter=config["max_iterations"],
        max_eval=config.get("max_evaluations", config["max_iterations"]),
        history_size=10, line_search_fn="strong_wolfe",
    )
    history = []

    def scores():
        return [nonlinear_logits(bucket["x"][:, :-1], head.cv3[i][1][1], head.cv3[i][-1]) for i, bucket in enumerate(cache)]

    def closure():
        optimizer.zero_grad()
        penalty = sum((p - original[n]).square().sum() for n, p in active.items())
        loss, parts = objective(torch.zeros(3, 257), cache, config, scores(), penalty)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active.values())
        history.append({"loss": float(loss.detach()), **parts})
        if len(history) % 20 == 0:
            print(f"closure {len(history)} loss={float(loss.detach()):.6f}", flush=True)
        return loss

    optimizer.step(closure)
    assert len(history) <= config.get("max_evaluations", config["max_iterations"]), "Optimizer exceeded evaluation budget"
    target_scores = []
    with torch.no_grad():
        for scale, (bucket, z) in enumerate(zip(cache, scores())):
            for group in bucket["group"].unique():
                if group >= 0:
                    mask = bucket["group"] == group
                    target_scores.append({"scale": scale, "group": int(group), "kind": int(bucket["kind"][mask][0]),
                                          "before_max": float(bucket["teacher"][mask].sigmoid().max()), "after_max": float(z[mask].sigmoid().max())})
    changed = [n for n, p in model.state_dict().items() if not torch.equal(original[n], p)]
    assert changed and set(changed) <= set(active)
    model.end2end = False
    model.criterion = None
    torch.save({"model": deepcopy(model).float().eval(), "ema": None, "epoch": -1,
                "train_args": {**wrapper.overrides, "device": "cpu"},
                "correction": {"config": config, "cache_manifest_sha256": fixed.file_sha256(cache_root / "cache_manifest.json")}}, run / "weights/candidate.pt")
    reloaded = YOLO(str(run / "weights/candidate.pt")).model
    assert not reloaded.end2end and all(torch.equal(model.state_dict()[n], reloaded.state_dict()[n]) for n in original)
    optimizer_state = optimizer.state[next(iter(active.values()))]
    write(audit / "fit_report.json", {"history": history, "target_scores": target_scores, "changed_tensors": changed,
                                     "optimizer_invocations": 1, "optimizer_iterations": optimizer_state.get("n_iter"),
                                     "closure_evaluations": len(history),
                                     "candidate_sha256": fixed.file_sha256(run / "weights/candidate.pt")})
    print({"initial_loss": history[0], "final_loss": history[-1], "target_scores": target_scores})


if __name__ == "__main__":
    main()
