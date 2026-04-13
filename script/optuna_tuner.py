#!/usr/bin/env python3
"""Optuna hyperparameter tuner for CSTG training.

Runs Bayesian optimization over training config parameters, using PSNR as the
objective. Reads intermediate metrics from metrics.json (written by train.py)
and prunes underperforming trials early.

Usage:
    python script/optuna_tuner.py \
        --source_path data/leopard_run/colmap_0 \
        --base_config configs/n3d_ours/leopard_ours_v33.json \
        --n_trials 20 \
        --study_name leopard_tuning \
        --output_dir log/optuna_runs
"""

import argparse
import json
import os
import subprocess
import sys
import time

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

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
    config["densify_grad_threshold"] = trial.suggest_float("densify_grad_threshold", 0.00003, 0.0005, log=True)
    config["gnumlimit"] = trial.suggest_int("gnumlimit", 300000, 3000000, step=100000)
    config["densify_until_iter"] = trial.suggest_int("densify_until_iter", 10000, 30000, step=1000)
    config["desicnt"] = trial.suggest_int("desicnt", 6, 18, step=3)

    # Pruning and masking
    config["lambda_mask"] = trial.suggest_float("lambda_mask", 0.0002, 0.005, log=True)
    config["mask_prune_iter"] = trial.suggest_int("mask_prune_iter", 500, 2000, step=500)
    config["lambda_dssim"] = trial.suggest_float("lambda_dssim", 0.1, 0.4)

    # EMS (error-guided multiview sampling)
    config["emsstart"] = trial.suggest_int("emsstart", 6000, 20000, step=1000)

    # Model capacity
    config["max_hashmap"] = trial.suggest_int("max_hashmap", 14, 18)
    config["rvq_size_geo"] = trial.suggest_categorical("rvq_size_geo", [256, 512, 1024])
    config["rvq_size_temp"] = trial.suggest_categorical("rvq_size_temp", [256, 512, 1024])
    config["rvq_num_geo"] = trial.suggest_int("rvq_num_geo", 3, 5)
    config["rvq_num_temp"] = trial.suggest_int("rvq_num_temp", 3, 5)

    # Training duration
    config["iterations"] = trial.suggest_int("iterations", 20000, 45000, step=5000)

    # Derived params (must stay consistent)
    config["rvq_iter"] = max(config["iterations"] - 2000, config["densify_until_iter"] + 1000)
    n_steps = max(4, config["iterations"] // 5000)
    config["net_lr_step"] = [int(config["iterations"] * (i + 1) / (n_steps + 1)) for i in range(n_steps)]

    return config


def read_metrics(metrics_path):
    """Read metrics.json, returning dict of {iteration: avg_psnr}."""
    if not os.path.exists(metrics_path):
        return {}
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        return {e["iteration"]: e["avg_psnr"] for e in data.get("metrics", [])}
    except (json.JSONDecodeError, KeyError):
        return {}


def read_final_psnr(metrics_path):
    """Read final PSNR from completed training."""
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        if "final" in data:
            return data["final"]["avg_psnr"]
        if data.get("metrics"):
            return data["metrics"][-1]["avg_psnr"]
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def objective(trial, args, base_config):
    trial_dir = os.path.join(args.output_dir, f"trial_{trial.number:03d}")
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
            psnr_by_iter = read_metrics(metrics_path)

            # Report at checkpoint iterations
            while report_step < len(REPORT_ITERS):
                target_iter = REPORT_ITERS[report_step]
                if target_iter in psnr_by_iter:
                    psnr_val = psnr_by_iter[target_iter]
                    trial.report(psnr_val, step=report_step)
                    last_psnr = psnr_val
                    report_step += 1

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

    # Read final PSNR
    final_psnr = read_final_psnr(metrics_path)
    if final_psnr is not None:
        return final_psnr
    if last_psnr is not None:
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
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_config = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    storage = f"sqlite:///{os.path.join(os.path.abspath(args.output_dir), args.study_name + '.db')}"

    pruner = MedianPruner(
        n_startup_trials=3,   # first 3 trials run to completion
        n_warmup_steps=2,     # don't prune before step 2 (iter 5000)
    )
    sampler = TPESampler(seed=42, n_startup_trials=5)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        pruner=pruner,
        sampler=sampler,
    )

    print(f"Starting Optuna study: {args.study_name}")
    print(f"  Base config: {args.base_config}")
    print(f"  Source path: {args.source_path}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Output: {args.output_dir}")
    print(f"  Storage: {storage}")
    print()

    study.optimize(
        lambda trial: objective(trial, args, base_config),
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
    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best PSNR: {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Write best config
    best_config = dict(base_config)
    best_config.update(study.best_params)
    best_config["rvq_iter"] = max(best_config["iterations"] - 2000, best_config["densify_until_iter"] + 1000)
    n_steps = max(4, best_config["iterations"] // 5000)
    best_config["net_lr_step"] = [int(best_config["iterations"] * (i + 1) / (n_steps + 1)) for i in range(n_steps)]

    best_config_path = os.path.join(args.output_dir, "best_config.json")
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"\nBest config written to: {best_config_path}")


if __name__ == "__main__":
    main()
