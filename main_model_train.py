import argparse
import random
import warnings
import numpy as np
import os
from data.get_data import get_dataload
import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_config, get_model_complexity_info, compute_fold_attributions
from models.get_model import get_model, get_opt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


# ----------------------------------------------Parameters------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='mdd', type=str, choices=['adhd', 'mdd', 'sch', 'idd'])
    parser.add_argument('--model_name', default='OhCNN', type=str,
                        choices=['OhCNN', 'DeprNet', 'SzHNN', 'MBSzEEGNet', 'STGEFormer'])

    parser.add_argument('--model_setting', default='original.yaml', type=str)
    parser.add_argument('--sampling_rate', default=200, type=int)
    parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
    parser.add_argument('--epochs', default=100, type=int, help='epochs for training')
    parser.add_argument('--patience', default=20, type=int, help='early stopping')
    parser.add_argument('--lr_patience', default=10, type=int, help='learning rate decay')
    parser.add_argument('--lr_decay_factor', default=0.5, type=float, help='learning rate decay factor')

    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=10, type=int, help='10,20,30,40,50')

    known_args, _ = parser.parse_known_args()

    return parser.parse_args()
# -------------------------------------------------------------------------------------------------------------


def seed_everything(seed=6718):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed(seed)  # PyTorch, for CUDA
    torch.backends.cudnn.deterministic = True  # PyTorch, for deterministic algorithm
    torch.backends.cudnn.benchmark = False  # PyTorch, to disable dynamic algorithms

# -------------------------------------------------------------------------------------------------------------


def train_one_epoch(model, device, data_loader, criterion, optimizer):
    model.train()
    correct = 0
    total = 0
    for samples, targets in data_loader:
        samples, targets = samples.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(samples)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
    accuracy = correct / total
    return accuracy


def evaluate(model, device, data_loader):
    model.eval()
    all_targets = []
    all_probabilities = []
    all_predictions = []

    with torch.no_grad():
        for samples, targets in data_loader:
            samples, targets = samples.to(device), targets.to(device)
            outputs = model(samples)

            _, predicted = torch.max(outputs, 1)

            probabilities = torch.softmax(outputs, dim=1)
            positive_probs = probabilities[:, 1]

            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(positive_probs.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, zero_division=0)
    recall = recall_score(all_targets, all_predictions, zero_division=0)
    f1 = f1_score(all_targets, all_predictions, zero_division=0)
    try:
        auc_roc = roc_auc_score(all_targets, all_probabilities)
    except ValueError as e:
        auc_roc = 0.5

    return accuracy, f1, precision, recall, auc_roc


def train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs, patience=20, lr_patience=10, lr_decay_factor=0.5):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    epoch_no_improvement = 0

    for epoch in range(1, num_epochs+1):
        train_one_epoch(model, device, train_loader, criterion, optimizer)
        val_acc, _, _, _, _ = evaluate(model, device, val_loader)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
            epoch_no_improvement = 0
        else:
            epoch_no_improvement += 1
        if epoch_no_improvement >= patience:
            break
        elif epoch_no_improvement >= lr_patience:
            for param_group in optimizer.param_groups:
                new_lr = param_group['lr'] * lr_decay_factor
                param_group['lr'] = new_lr

    model.load_state_dict(best_model_wts)
    return model


def main(args):
    seed_everything(args.seed)
    device = torch.device(args.device)

    dataloader_train_folds, dataloader_val_folds, dataloader_test_folds, ch_num, ch_names = get_dataload(args)
    cfg = load_config("scripts/" + args.model_name + "/" + args.model_setting)

    test_acc_list = []
    test_pre_list = []
    test_rec_list = []
    test_f1_list = []
    test_auc_list = []

    # 💡 新增：用于在全局累加 5 折产生的 attributions
    global_hc_attributions = []
    global_pt_attributions = []

    for fold_idx, (data_loader_train, data_loader_val, data_loader_test) in enumerate(zip(dataloader_train_folds.values(),
                                                                    dataloader_val_folds.values(),
                                                                    dataloader_test_folds.values())):
        current_fold_seed = args.seed + fold_idx
        seed_everything(current_fold_seed)

        model = get_model(args.model_name, ch_num, cfg)
        model.to(device)
        param, flop = get_model_complexity_info(model, (1, ch_num, 2000), device)
        print(param, flop)
        criterion = nn.CrossEntropyLoss()

        optimizer = get_opt(model.parameters(), cfg)

        best_model = train_model(model, device, data_loader_train, data_loader_val, criterion, optimizer, args.epochs,
                                 args.patience, args.lr_patience, args.lr_decay_factor)

        test_acc, test_f1, test_pre, test_rec, test_auc = evaluate(best_model, device, data_loader_test)

        hc_fold, pt_fold = compute_fold_attributions(best_model, device, data_loader_test)
        global_hc_attributions.extend(hc_fold)
        global_pt_attributions.extend(pt_fold)

        test_acc_list.append(test_acc)
        test_f1_list.append(test_f1)
        test_pre_list.append(test_pre)
        test_rec_list.append(test_rec)
        test_auc_list.append(test_auc)

    #

    acc_str = '\t'.join([f"{x:.4f}" for x in test_acc_list])
    pre_str = '\t'.join([f"{x:.4f}" for x in test_pre_list])
    rec_str = '\t'.join([f"{x:.4f}" for x in test_rec_list])
    f1_str = '\t'.join([f"{x:.4f}" for x in test_f1_list])
    auc_str = '\t'.join([f"{x:.4f}" for x in test_auc_list])

    print(f'test_acc:\t{acc_str}')
    print(f'test_pre:\t{pre_str}')
    print(f'test_rec:\t{rec_str}')
    print(f'test_f1:\t{f1_str}')
    print(f'test_auc:\t{auc_str}')

    arch_name = args.model_setting.replace('.yaml', '')
    save_dir = os.path.join("results", args.model_name)
    os.makedirs(save_dir, exist_ok=True)

    hc_arr = np.array(global_hc_attributions)
    pt_arr = np.array(global_pt_attributions)

    data_filename = f"{args.dataset}_{arch_name}_seed_{args.seed}.npz"
    data_save_path = os.path.join(save_dir, data_filename)

    np.savez(data_save_path, hc=hc_arr, pt=pt_arr)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    opts = get_args()
    main(opts)

