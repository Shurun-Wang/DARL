import argparse
import warnings
from pathlib import Path

import torch
import yaml

from data.get_data import get_hpo_dataload
from optimizer.darl import darl_optimizer
from optimizer.runtime import start_timer, stop_timer, reset_complexity_cache, complexity_cache_info
from utils import seed_everything, get_dict


# ------------------------------ Args ------------------------------
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='adhd', type=str, choices=['adhd', 'idd', 'mdd', 'sch'])
    parser.add_argument('--fold_idx', default=1, type=int)
    parser.add_argument('--model_name', default='DeprNet', type=str, choices=['DeprNet', 'MBSzEEGNet', 'OhCNN', 'SzHNN', 'STGEFormer'])

    parser.add_argument('--sampling_rate', default=200, type=int)
    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--algorithm', default='darl', type=str, choices=['darl'])
    parser.add_argument('--search_trials', default=3000, type=int)
    parser.add_argument('--search_epochs', default=3, type=int)
    parser.add_argument('--fixed_noise_std', type=float, default=0.01)
    parser.add_argument('--schedule_noise_start', type=float, default=0, help='linearly increasing noise start; default grad_noise_min')
    parser.add_argument('--schedule_noise_end', type=float, default=0.01, help='linearly increasing noise end; default grad_noise_max')
    parser.add_argument('--run_tag', default='', help='suffix distinguishing repeated ablation settings')

    parser.add_argument('--grad_noise_min', default=0.0, type=float)
    parser.add_argument('--grad_noise_max', default=0.01, type=float)
    parser.add_argument('--ppo_buffer_capacity', default=4, type=int)

    parser.add_argument('--device', default='mps', help='device')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='0 recommended for RAM cache; try 2-4 for disk reads')
    parser.add_argument('--data_cache', choices=['ram', 'none'], default='none')
    parser.add_argument('--prefetch_factor', default=2, type=int)
    parser.add_argument('--seed', default=10, type=int)

    args = parser.parse_args()
    return args



def main(args):
    warnings.filterwarnings("ignore")
    seed_everything(args.seed)
    device = torch.device(args.device)
    total_started = start_timer(device)
    train_loader, val_loader, ch_num, cache_bytes = get_hpo_dataload(args)
    data_setup_seconds = stop_timer(total_started, device)
    reset_complexity_cache()
    search_dict = get_dict(args.model_name)

    # Use the same single seed for every outer fold, as specified in the protocol.
    seed_everything(args.seed)
    fold_key = f"fold_{args.fold_idx}"

    print(f"\n========== Starting {fold_key} ==========")
    fold_started = start_timer(device)
    history, statistics = darl_optimizer(args, ch_num, device, train_loader, val_loader, search_dict)
    search_elapsed = stop_timer(fold_started, device)
    fold_elapsed = stop_timer(total_started, device)

    total_trials = int(statistics.get("total_trials", len(history)))
    valid_count = int(statistics.get("success_trials", 0))
    invalid_count = int(statistics.get("failed_trials", total_trials - valid_count))
    trial_times = [float(item["trial_time_seconds"]) for item in history if "trial_time_seconds" in item]
    valid_scores = [float(item["val_balanced_accuracy"]) for item in history if item.get("valid", True) and "val_balanced_accuracy" in item]

    statistics.update({
        "outer_fold": args.fold_idx,
        "seed": args.seed,
        "device": str(device),
        "algorithm": args.algorithm,
        "run_tag": args.run_tag,
        "batch_size": args.batch_size,
        "data_cache": args.data_cache,
        "data_cache_bytes": cache_bytes,
        "num_workers": args.num_workers,
        "data_setup_seconds": data_setup_seconds,
        "search_time_seconds": search_elapsed,
        "complexity_cache": statistics.get('complexity_cache', complexity_cache_info()),
        "timing_scope": "selected-fold data setup plus HPO; single-device allocation wall time",
        "wall_time_seconds": float(fold_elapsed),
        "wall_time_hours": float(fold_elapsed / 3600.0),
        "gpu_hours": float(fold_elapsed / 3600.0) if device.type in {"cuda", "mps"} else 0.0,
        "average_trial_time_seconds": (
            float(sum(trial_times) / len(trial_times)) if trial_times else 0.0),
        "amortized_trial_time_seconds": float(fold_elapsed / total_trials) if total_trials else 0.0,
        "recorded_trial_time_seconds": float(sum(trial_times)),
        "valid_configurations": valid_count,
        "invalid_configurations": invalid_count,
        "complexity_profile_failures": sum(
            1 for item in history
            if item.get("valid", False) and not item.get("complexity_profiled", False)
        ),

        "best_val_balanced_accuracy": max(valid_scores) if valid_scores else None,
        "sensitivity_parameters": {
            "grad_noise_min": args.grad_noise_min,
            "grad_noise_max": args.grad_noise_max,
            "ppo_buffer_capacity": args.ppo_buffer_capacity,
            "search_trials": args.search_trials,
        },
    })

    # Raw per-trial accuracy/FLOPs/timing in history supports later Pareto and
    # accuracy-gain-per-GPU-hour analyses without rerunning HPO.
    out_dir = Path("scripts") / str(args.fold_idx) /args.model_name
    if args.algorithm not in ('darl'):
        out_dir = out_dir / 'ablation' / args.algorithm
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.algorithm}_{args.dataset}_{str(args.grad_noise_max)}_{str(args.ppo_buffer_capacity)}_{str(args.seed)}"
    with open(out_dir / f"{stem}_history.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(history, f, sort_keys=False, allow_unicode=True)
    with open(out_dir / f"{stem}_statistics.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(statistics, f, sort_keys=False, allow_unicode=True)
    print(f"Saved results to {out_dir}")


if __name__ == '__main__':
    opts = get_args()
    main(opts)
