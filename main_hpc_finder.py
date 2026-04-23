import argparse
import warnings
from pathlib import Path
import yaml
from optimizer.darl import darl_optimizer
from utils import seed_everything, get_dict
from data.get_data import get_dataload
import torch

# ------------------------------ Args ------------------------------
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='sch', type=str, choices=['adhd', 'idd', 'mdd', 'sch'])
    parser.add_argument('--model_name', default='SzHNN', type=str,
                        choices=['DeprNet', 'MBSzEEGNet', 'OhCNN', 'SzHNN', 'STGEFormer'])

    parser.add_argument('--sampling_rate', default=200, type=int)
    parser.add_argument('--batch_size', default=64, type=int)

    # Search controls
    parser.add_argument('--algorithm', default='darl', type=str, help='darl')
    parser.add_argument('--search_trials', default=3000, type=int, help='number of TPE trials')
    parser.add_argument('--search_epochs', default=3, type=int, help='epochs per trial')

    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=10, type=int)

    return parser.parse_args()


def main(args):
    warnings.filterwarnings("ignore")
    seed_everything(args.seed)
    device = torch.device(args.device)

    dataloader_train_folds, dataloader_val_folds, _, ch_num, _ = get_dataload(args)
    train_loader = dataloader_train_folds['fold_1']
    val_loader = dataloader_val_folds['fold_1']

    search_dict = get_dict(args.model_name)
    history, statistics = darl_optimizer(args, ch_num, device, train_loader, val_loader, search_dict)

    # -------------------- Save best hp --------------------
    out_dir = Path("scripts") / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    hist_path = out_dir / f"{args.algorithm}_{args.dataset}_history.yaml"
    with open(hist_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(history, f, sort_keys=False, allow_unicode=True)

    stati_path = out_dir / f"{args.algorithm}_{args.dataset}_statistics.yaml"
    with open(stati_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(statistics, f, sort_keys=False, allow_unicode=True)


if __name__ == '__main__':
    opts = get_args()
    main(opts)
