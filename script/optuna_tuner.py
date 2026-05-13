#!/usr/bin/env python3
"""Optuna hyperparameter tuner for CSTG training.

Three modes:
  default              single-objective: maximize training-window PSNR
                       (cheap, prunes weak trials early via MedianPruner)
  --multi_objective    Pareto: maximize training-window PSNR vs minimize Gaussian count
                       (compactness trade-off)
  --quality_objective  Pareto on held-out test cameras: PSNR + SSIM + LPIPS
                       (3 objectives, no early pruning, runs test.py per trial)

The quality_objective mode is the right one when you actually care about
generalization quality and don't want PSNR-only to mislead — train.py only
writes train-view PSNR/SSIM, so we run test.py post-training to get held-out
PSNR/SSIM/LPIPS-Alex.

Usage:
    python script/optuna_tuner.py \
        --source_path data/tree_tech/colmap_0 \
        --base_config configs/techni_custom/tree_tech.json \
        --n_trials 20 \
        --study_name tree_tech_quality \
        --output_dir log/optuna_runs \
        --quality_objective
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Map training loader name -> matching test.py --valloader name.
_VALLOADER = {
    "colmap": "colmapvalid",
    "technicolor": "technicolorvalid",
    "immersive": "immersivevalid",
}

# Iterations at which we report intermediate PSNR to Optuna for pruning
REPORT_ITERS = [1000, 3000, 5000, 8000, 12000]
POLL_INTERVAL = 60  # seconds between metrics.json checks


def suggest_params(trial, base_config):
    """Suggest hyperparameters, returning a full config dict."""
    config = dict(base_config)

    # Core learning rates
    config["scaling_lr"] = trial.suggest_float("scaling_lr", 0.0003, 0.005, log=True)
    config["opacity_lr"] = trial.suggest_float("opacity_lr", 0.01, 0.1, log=True)
    config["mask_lr"] = trial.suggest_float("mask_lr", 0.003, 0.02, log=True)

    # Densification
    config["densify_grad_threshold"] = trial.suggest_float("densify_grad_threshold", 0.0001, 0.0005, log=True)
    config["gnumlimit"] = trial.suggest_int("gnumlimit", 300000, 2000000, step=100000)
    config["densify_until_iter"] = trial.suggest_int("densify_until_iter", 7000, 15000, step=1000)
    config["desicnt"] = trial.suggest_int("desicnt", 3, 12, step=3)

    # Pruning and masking
    config["lambda_mask"] = trial.suggest_float("lambda_mask", 0.0002, 0.005, log=True)
    config["mask_prune_iter"] = trial.suggest_int("mask_prune_iter", 500, 2000, step=250)
    config["lambda_dssim"] = trial.suggest_float("lambda_dssim", 0.1, 0.4)

    # Alpha (silhouette) loss weight — only meaningful when the scene has masks.
    # If lambda_alpha is not in the base config (non-masked scene), this overrides
    # to a non-zero value but the loss in train.py silently skips when
    # gt_alpha_mask is None, so it's safe to always sweep.
    config["lambda_alpha"] = trial.suggest_float("lambda_alpha", 0.1, 1.0)

    # EMS (error-guided multiview sampling)
    config["emsstart"] = trial.suggest_int("emsstart", 1000, 10000, step=1000)

    # Model capacity
    config["max_hashmap"] = trial.suggest_int("max_hashmap", 14, 17)
    config["rvq_size_geo"] = trial.suggest_categorical("rvq_size_geo", [256, 512, 1024])
    config["rvq_size_temp"] = trial.suggest_categorical("rvq_size_temp", [256, 512, 1024])
    config["rvq_num_geo"] = trial.suggest_int("rvq_num_geo", 3, 5)
    config["rvq_num_temp"] = trial.suggest_int("rvq_num_temp", 3, 5)

    # Training duration
    config["iterations"] = trial.suggest_int("iterations", 20000, 45000, step=5000)

    # Derived params (must stay consistent)
    config["rvq_iter"] = max(config["iterations"] - 6000, config["densify_until_iter"] + 1000)
    n_steps = max(4, config["iterations"] // 5000)
    config["net_lr_step"] = [int(config["iterations"] * (i + 1) / (n_steps + 1)) for i in range(n_steps)]
    # test.py reads test_iteration from the config and looks for
    # point_cloud/iteration_<N>/point_cloud.ply — must match the swept iters.
    config["test_iteration"] = config["iterations"]

    return config


def read_metrics(metrics_path):
    """Read metrics.json, returning list of metric entries."""
    if not os.path.exists(metrics_path):
        return []
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        return data.get("metrics", [])
    except (json.JSONDecodeError, KeyError):
        return []


def read_final_metrics(metrics_path):
    """Read final PSNR and gaussian count from completed training."""
    if not os.path.exists(metrics_path):
        return None, None
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        if "final" in data:
            return data["final"]["avg_psnr"], data["final"]["num_gaussians"]
        if data.get("metrics"):
            last = data["metrics"][-1]
            return last["avg_psnr"], last["num_gaussians"]
    except (json.JSONDecodeError, KeyError):
        pass
    return None, None


def read_last_line(file_path):
    """Read the last line of a file efficiently, handling carriage returns from tqdm."""
    if not os.path.exists(file_path):
        return ""
    try:
        with open(file_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            curr_pos = f.tell()
            if curr_pos == 0:
                return ""
            
            # Read a chunk from the end to find the latest update
            read_size = min(curr_pos, 2048)
            f.seek(curr_pos - read_size)
            chunk = f.read(read_size).decode(errors='ignore')
            
            # tqdm uses \r to update the same line; log files preserve these.
            # We want the text after the very last \r or \n.
            import re
            lines = re.split(r'[\r\n]', chunk)
            # Filter out empty strings and return the last non-empty one
            for line in reversed(lines):
                if line.strip():
                    return line.strip()
            return ""
    except Exception:
        return ""


def run_test_and_read_metrics(trial_dir, source_path, config_path, base_config):
    """Run test.py on a freshly-trained model, then read held-out PSNR/SSIM/LPIPS.

    Returns (psnr, ssim, lpips_alex) or None on failure.
    """
    loader = base_config.get("loader", "colmap")
    valloader = _VALLOADER.get(loader)
    if valloader is None:
        raise ValueError(f"Unknown loader '{loader}'; cannot pick valloader for test.py")

    cmd = [
        sys.executable, "test.py",
        "--quiet", "--eval", "--skip_train",
        "--valloader", valloader,
        "--configpath", config_path,
        "--model_path", trial_dir,
        "--source_path", source_path,
        "--test_iteration", "-1",  # auto-resolve to latest saved iteration
    ]
    test_log = os.path.join(trial_dir, "test_stdout.log")
    with open(test_log, "w") as lf:
        ret = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if ret != 0:
        print(f"  test.py failed (exit {ret}); see {test_log}")
        return None

    # test.py writes <model>/<iteration>_runtimeresults.json — pick the latest iter.
    candidates = sorted(
        glob.glob(os.path.join(trial_dir, "*_runtimeresults.json")),
        key=lambda p: int(os.path.basename(p).split("_")[0]),
    )
    if not candidates:
        print(f"  no *_runtimeresults.json found in {trial_dir}")
        return None
    with open(candidates[-1]) as f:
        data = json.load(f)
    # Structure: { model_path: { iteration_str: { "PSNR":..., "SSIM":..., "LPIPS":..., ... } } }
    try:
        inner = next(iter(data.values()))
        iter_block = next(iter(inner.values()))
        return (
            float(iter_block["PSNR"]),
            float(iter_block["SSIM"]),
            float(iter_block["LPIPS"]),
        )
    except (StopIteration, KeyError, ValueError) as e:
        print(f"  could not parse test metrics from {candidates[-1]}: {e}")
        return None


def objective(trial, args, base_config, multi_objective=False, quality_objective=False):
    trial_dir = os.path.join(args.output_dir, args.study_name, f"trial_{trial.number:03d}")
    os.makedirs(trial_dir, exist_ok=True)

    # Suggest params and write config
    config = suggest_params(trial, base_config)
    config_path = os.path.join(trial_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Launch training subprocess
    cmd = [
        sys.executable, "train.py",
        "--quiet", "--eval",
        "--configpath", config_path,
        "--model_path", trial_dir,
        "--source_path", args.source_path,
        "--comp", "--store_npz",
        "--data_device", "cpu",
    ]
    stdout_path = os.path.join(trial_dir, "stdout.log")
    stdout_file = open(stdout_path, 'w')
    process = subprocess.Popen(cmd, stdout=stdout_file, stderr=subprocess.STDOUT)

    metrics_path = os.path.join(trial_dir, "metrics.json")
    report_step = 0
    last_psnr = None

    try:
        while True:
            retcode = process.poll()

            # Read intermediate metrics
            metrics = read_metrics(metrics_path)
            if metrics:
                last_m = metrics[-1]
                trial.set_user_attr("cur_iter", last_m["iteration"])
                trial.set_user_attr("cur_psnr", round(last_m["avg_psnr"], 4))
                trial.set_user_attr("cur_gaussians", last_m.get("num_gaussians", 0))

            # Update dashboard with last log line
            last_log = read_last_line(stdout_path)
            if last_log:
                trial.set_user_attr("last_log", last_log)

            # Report at checkpoint iterations (single-objective only)
            psnr_by_iter = {m["iteration"]: m["avg_psnr"] for m in metrics}
            while report_step < len(REPORT_ITERS):
                target_iter = REPORT_ITERS[report_step]
                if target_iter in psnr_by_iter:
                    psnr_val = psnr_by_iter[target_iter]
                    last_psnr = psnr_val
                    report_step += 1

                    if not multi_objective and not quality_objective:
                        trial.report(psnr_val, step=report_step - 1)
                        if trial.should_prune():
                            process.terminate()
                            try:
                                process.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                process.kill()
                            stdout_file.close()
                            raise optuna.TrialPruned(
                                f"Pruned at iter {target_iter} with PSNR {psnr_val:.2f}"
                            )
                else:
                    break

            if retcode is not None:
                stdout_file.close()
                if retcode != 0:
                    raise optuna.TrialPruned(f"Training crashed with exit code {retcode}")
                break

            time.sleep(POLL_INTERVAL)

    except optuna.TrialPruned:
        raise
    except Exception:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        stdout_file.close()
        raise

    # Quality-objective mode: run test.py for held-out PSNR/SSIM/LPIPS.
    if quality_objective:
        result = run_test_and_read_metrics(trial_dir, args.source_path, config_path, base_config)
        if result is None:
            raise optuna.TrialPruned("test.py failed or produced no metrics")
        psnr_t, ssim_t, lpips_t = result
        trial.set_user_attr("test_psnr", round(psnr_t, 4))
        trial.set_user_attr("test_ssim", round(ssim_t, 6))
        trial.set_user_attr("test_lpips", round(lpips_t, 6))
        return psnr_t, ssim_t, lpips_t

    # Read final metrics (training-window)
    final_psnr, final_gaussians = read_final_metrics(metrics_path)
    if final_psnr is not None:
        if multi_objective:
            return final_psnr, final_gaussians
        return final_psnr
    if last_psnr is not None:
        if multi_objective:
            # Try to get last gaussian count from metrics
            try:
                with open(metrics_path) as f:
                    data = json.load(f)
                if data.get("metrics"):
                    return last_psnr, data["metrics"][-1]["num_gaussians"]
            except (json.JSONDecodeError, KeyError):
                pass
            return last_psnr, 0
        return last_psnr
    raise optuna.TrialPruned("No metrics available after training")


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuner for CSTG")
    parser.add_argument("--source_path", required=True, help="Path to colmap data (e.g. data/leopard_run/colmap_0)")
    parser.add_argument("--base_config", required=True, help="Path to base config JSON to start from")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials to run")
    parser.add_argument("--study_name", default="cstg_tuning", help="Optuna study name (used for DB filename)")
    parser.add_argument("--output_dir", default="log/optuna_runs", help="Directory for trial outputs")
    parser.add_argument("--timeout", type=int, default=None, help="Total study timeout in seconds")
    parser.add_argument("--multi_objective", action="store_true", help="Maximize PSNR while minimizing gaussian count (compactness Pareto)")
    parser.add_argument("--quality_objective", action="store_true",
                        help="Pareto over held-out test metrics: maximize PSNR + SSIM, minimize LPIPS-Alex. "
                             "Runs test.py per trial, no early pruning.")
    args = parser.parse_args()

    if args.multi_objective and args.quality_objective:
        parser.error("--multi_objective and --quality_objective are mutually exclusive")

    with open(args.base_config) as f:
        base_config = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    storage = f"sqlite:///{os.path.join(os.path.abspath(args.output_dir), args.study_name + '.db')}"

    sampler = TPESampler(seed=42, n_startup_trials=5)

    if args.quality_objective:
        # Held-out PSNR ↑, SSIM ↑, LPIPS ↓. Pareto front, no pruner.
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            load_if_exists=True,
            directions=["maximize", "maximize", "minimize"],
            sampler=sampler,
        )
    elif args.multi_objective:
        # Multi-objective: no pruner (not supported), Pareto front
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            load_if_exists=True,
            directions=["maximize", "minimize"],  # PSNR up, gaussians down
            sampler=sampler,
        )
    else:
        pruner = MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=2,
        )
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            pruner=pruner,
            sampler=sampler,
        )

    if args.quality_objective:
        mode = "quality multi-objective (test PSNR + SSIM + LPIPS)"
    elif args.multi_objective:
        mode = "multi-objective (PSNR + gaussian count)"
    else:
        mode = "single-objective (PSNR)"
    print(f"Starting Optuna study: {args.study_name}")
    print(f"  Mode: {mode}")
    print(f"  Base config: {args.base_config}")
    print(f"  Source path: {args.source_path}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Output: {args.output_dir}")
    print(f"  Storage: {storage}")
    print()

    study.optimize(
        lambda trial: objective(
            trial, args, base_config,
            multi_objective=args.multi_objective,
            quality_objective=args.quality_objective,
        ),
        n_trials=args.n_trials,
        timeout=args.timeout,
        catch=(Exception,),
    )

    # Print results
    print("\n" + "=" * 60)
    print("STUDY COMPLETE")
    print("=" * 60)
    print(f"Total trials: {len(study.trials)}")
    print(f"  Completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"  Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"  Failed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")

    if args.quality_objective:
        print(f"\nPareto front ({len(study.best_trials)} trials):")
        for t in study.best_trials:
            print(f"  Trial #{t.number}: PSNR={t.values[0]:.4f}  SSIM={t.values[1]:.4f}  LPIPS={t.values[2]:.4f}")
    elif args.multi_objective:
        print(f"\nPareto front ({len(study.best_trials)} trials):")
        for t in study.best_trials:
            print(f"  Trial #{t.number}: PSNR={t.values[0]:.4f}, Gaussians={int(t.values[1]):,}")
    else:
        print(f"\nBest trial: #{study.best_trial.number}")
        print(f"Best PSNR: {study.best_value:.4f}")
    print(f"Best params:")
    best_trial = (study.best_trials[0]
                  if (args.multi_objective or args.quality_objective)
                  else study.best_trial)
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")

    # Write best config
    best_config = dict(base_config)
    best_config.update(best_trial.params)
    best_config["rvq_iter"] = max(best_config["iterations"] - 2000, best_config["densify_until_iter"] + 1000)
    n_steps = max(4, best_config["iterations"] // 5000)
    best_config["net_lr_step"] = [int(best_config["iterations"] * (i + 1) / (n_steps + 1)) for i in range(n_steps)]

    best_config_path = os.path.join(args.output_dir, "best_config.json")
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"\nBest config written to: {best_config_path}")


if __name__ == "__main__":
    main()
