import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import warnings
from uuid import uuid4
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (accuracy_score, f1_score, precision_score, roc_auc_score, matthews_corrcoef, confusion_matrix)

ROOT = Path(__file__).resolve().parent


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='adhd', choices=['adhd', 'mdd', 'sch', 'idd'])
    parser.add_argument('--model_name', default='DeprNet', choices=['OhCNN', 'DeprNet', 'SzHNN', 'MBSzEEGNet', 'STGEFormer'])
    parser.add_argument('--model_setting', default='/Users/wangsr/PycharmProjects/NAI_tnsre_major/scripts/DeprNet/darl_adhd.yaml', help='legacy scripts/<model_name>/ filename')
    parser.add_argument('--config_path', '--model_config_path', default=None, help='YAML configuration file or directory, not a .pth weight file')
    parser.add_argument('--config_rank', type=int, default=None, choices=[1], help='optional exported rank filter')
    parser.add_argument('--output_dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--split_root', default=None)
    parser.add_argument('--sampling_rate', default=200, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--lr_patience', default=10, type=int)
    parser.add_argument('--lr_decay_factor', default=.5, type=float)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--fold', default=1, type=int, choices=range(1, 6))
    parser.add_argument('--save_attributions', default=False, action='store_true', help='optional expensive Integrated Gradients')
    parser.add_argument('--run_layout', choices=['timestamp', 'stable'], default=None,
                        help='save_attributions defaults to stable names; other training keeps timestamp layout')
    parser.add_argument('--ig_steps', type=int, default=64)
    parser.add_argument('--ig_max_steps', type=int, default=256)
    parser.add_argument('--ig_batch_size', type=int, default=8)
    parser.add_argument('--ig_internal_batch_size', type=int, default=32)
    parser.add_argument('--ig_atol', type=float, default=.001)
    parser.add_argument('--ig_rtol', type=float, default=.05)
    parser.add_argument('--skip_complexity', default=False, help='skip parameter/FLOPs profiling')
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.lr_patience, args.batch_size) < 1:
        parser.error('epochs/patience/lr_patience/batch_size must be positive')
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_training_configs(args):
    source = Path(args.config_path).expanduser() if args.config_path else ROOT / 'scripts' / args.model_name / args.model_setting
    paths = sorted(source.rglob('*.yaml')) if source.is_dir() else [source]
    selected = []
    for path in paths:
        payload = yaml.safe_load(path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            raise ValueError(f'Invalid configuration: {path}')
        metadata = payload.get('metadata', {})
        hp = payload.get('hp', payload)
        if not isinstance(hp, dict) or not {'opt', 'lr'} <= hp.keys():
            raise ValueError(f'Missing hp/opt/lr: {path}')
        if metadata:
            if metadata['dataset'] != args.dataset or metadata['model_name'] != args.model_name:
                if source.is_dir():
                    continue
                raise ValueError(f'Dataset/model mismatch: {path}')
            if int(metadata['outer_fold']) != args.fold:
                raise ValueError(f'Fold mismatch: {path} was searched on fold {metadata["outer_fold"]}, not {args.fold}')
            if args.config_rank is not None and metadata['rank'] != args.config_rank:
                continue
        else:
            if args.config_rank is not None:
                raise ValueError('Legacy YAML has no rank metadata')
            warnings.warn(f'{path}: no fold provenance; verify the configuration did not use this test fold')
        selected.append((path, hp, metadata))
    if not selected:
        raise ValueError(f'No matching configurations: {source}')
    return selected


def binary_metrics(targets, probabilities):
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    if not len(targets) or probabilities.shape != (len(targets), 2):
        raise ValueError('Expected nonempty targets and N x 2 probabilities')
    if not np.isfinite(probabilities).all() or not set(np.unique(targets)) <= {0, 1}:
        raise ValueError('Invalid probabilities or nonbinary labels')
    predictions = probabilities.argmax(axis=1)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    sensitivity = float(tp/(tp+fn)) if tp+fn else None
    specificity = float(tn/(tn+fp)) if tn+fp else None
    ba = (sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None
    return dict(accuracy=float(accuracy_score(targets, predictions)), sensitivity=sensitivity,
                specificity=specificity, balanced_accuracy=ba,
                precision=float(precision_score(targets, predictions, zero_division=0)),
                recall=sensitivity, f1=float(f1_score(targets, predictions, zero_division=0)),
                mcc=float(matthews_corrcoef(targets, predictions)),
                auc=float(roc_auc_score(targets, probabilities[:, 1])) if len(np.unique(targets)) == 2 else None,
                confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
                confusion_matrix_labels=[0, 1], positive_label=1, n_subjects=len(targets))


def evaluate(model, device, data_loader):
    model.eval()
    subjects = {}
    with torch.inference_mode():
        for batch in data_loader:
            if len(batch) != 3:
                raise ValueError('Subject-level evaluation requires subject IDs')
            samples, targets, ids = batch
            probabilities = torch.softmax(model(samples.to(device)), dim=1).cpu().numpy()
            for sid, target, probability in zip(ids, targets.tolist(), probabilities):
                sid = str(sid)
                if sid not in subjects:
                    subjects[sid] = [int(target), probability.astype(np.float64), 1]
                else:
                    if subjects[sid][0] != int(target):
                        raise ValueError(f'Inconsistent label for subject {sid}')
                    subjects[sid][1] += probability
                    subjects[sid][2] += 1
    records = [dict(subject_id=sid, target=label, prediction=int(np.argmax(total/count)),
                    probability_0=float(total[0]/count), probability_1=float(total[1]/count), n_windows=count)
               for sid, (label, total, count) in sorted(subjects.items())]
    metrics = binary_metrics([r['target'] for r in records],
                             [[r['probability_0'], r['probability_1']] for r in records])
    return metrics, records


def train_one_epoch(model, device, data_loader, criterion, optimizer):
    model.train()
    for samples, targets in data_loader:
        samples, targets = samples.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(samples), targets)
        loss.backward()
        optimizer.step()


def train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs,
                patience=20, lr_patience=10, lr_decay_factor=.5):
    best_state, best_metrics, best_epoch = None, None, None
    best_score, stale = -float('inf'), 0
    history = []
    for epoch in range(1, num_epochs+1):
        train_one_epoch(model, device, train_loader, criterion, optimizer)
        metrics, _ = evaluate(model, device, val_loader)
        score = metrics['balanced_accuracy']
        if score is None:
            raise ValueError('Validation requires both classes for balanced accuracy')
        history.append(dict(epoch=epoch, **metrics))
        if score > best_score:
            best_score, best_epoch, best_metrics = score, epoch, metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(f'epoch={epoch} validation_subject_BA={score:.6f} best={best_score:.6f}')
        if stale >= patience:
            break
        if stale and stale % lr_patience == 0:
            for group in optimizer.param_groups:
                group['lr'] *= lr_decay_factor
    if best_state is None:
        raise ValueError('No completed training epoch')
    model.load_state_dict(best_state)
    return model, best_metrics, best_epoch, history


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def stable_layout(args):
    layout = getattr(args, 'run_layout', None)
    return layout == 'stable' or (layout is None and getattr(args, 'save_attributions', False))


def make_run_dir(args, config_name=None):
    if stable_layout(args):
        if not config_name:
            raise ValueError('Stable output requires one named configuration')
        path = (Path(args.output_dir) / args.model_name / args.dataset / config_name /
                f'fold_{args.fold}' / f'seed_{args.seed}')
        path.mkdir(parents=True, exist_ok=False)  # 明确拒绝覆盖同配置、折、种子的结果。
        return path
    # Explicit seed partition plus unique run ID: reruns never overwrite prior results.
    path = (Path(args.output_dir) / args.model_name / args.dataset / f'fold_{args.fold}' /
            f'seed_{args.seed}' / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid4().hex[:8]))
    path.mkdir(parents=True, exist_ok=False)
    return path


def main(args):
    from data.get_data import get_dataload
    from models.get_model import get_model, get_opt
    configs = load_training_configs(args)
    if stable_layout(args) and len(configs) != 1:
        raise ValueError('Stable layout requires a single --config_path YAML, not multiple configurations')
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.load_selected_fold_only = True
    trains, vals, tests, ch_num, _ = get_dataload(args, include_subject_metadata=True)
    key = f'fold_{args.fold}'
    run_dir = make_run_dir(args, configs[0][0].stem)
    split_root = Path(args.split_root) if args.split_root else ROOT / 'preprocessing'
    split_hashes = {part: hashlib.sha256((split_root / args.dataset / f'json_{args.fold}' / f'{part}.json').read_bytes()).hexdigest()
                    for part in ('train', 'val', 'test')}
    summary = []
    for index, (path, hp, metadata) in enumerate(configs, 1):
        seed_everything(args.seed)
        out = run_dir if stable_layout(args) else run_dir / f'{index:02d}_{path.stem}'
        if out != run_dir:
            out.mkdir()
        model = get_model(args.model_name, ch_num, hp).to(device)
        complexity = None
        if not args.skip_complexity:
            from utils import get_model_complexity_info
            sample = next(iter(vals[key]))[0]
            params_m, flops_g = get_model_complexity_info(model, (1, *sample.shape[1:]), device)
            complexity = dict(params_m=float(params_m), flops_g=float(flops_g))
            print(f'Complexity: {complexity}')
        optimizer = get_opt(model.parameters(), hp)
        model, validation, best_epoch, history = train_model(model, device, trains[key], vals[key],
            nn.CrossEntropyLoss(), optimizer, args.epochs, args.patience, args.lr_patience, args.lr_decay_factor)
        test, predictions = evaluate(model, device, tests[key])
        torch.save(dict(model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                        hp=hp, metadata=metadata, model_name=args.model_name, dataset=args.dataset,
                        fold=args.fold, seed=args.seed, best_epoch=best_epoch), out / 'best_model.pth')
        write_json(out / 'metrics.json', dict(source_config=str(path.resolve()), source_metadata=metadata,
            hp=hp, fold=args.fold, seed=args.seed, complexity=complexity, evaluation_level='subject_probability_mean',
            current_split_sha256=split_hashes, best_epoch=best_epoch, validation=validation, test=test,
            training_settings={name: getattr(args, name, default) for name, default in
                [('epochs', 100), ('patience', 20), ('lr_patience', 10), ('lr_decay_factor', .5),
                 ('batch_size', 64), ('num_workers', 4), ('sampling_rate', 200)]}))
        write_json(out / 'training_history.json', history)
        with (out / 'test_subject_predictions.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        print(f'\n{path.name}\nTEST subject-level metrics:\n{json.dumps(test, indent=2)}')
        row = dict(config=str(path), dataset=args.dataset, fold=args.fold, seed=args.seed,
                   best_epoch=best_epoch, validation_balanced_accuracy=validation['balanced_accuracy'],
                   **{f'test_{k}': v for k, v in test.items() if not isinstance(v, list)})
        tn, fp = test['confusion_matrix'][0]
        fn, tp = test['confusion_matrix'][1]
        row.update(tn=tn, fp=fp, fn=fn, tp=tp)
        summary.append(row)
        with (run_dir / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        if args.save_attributions:
            from utils import compute_fold_attributions
            data = compute_fold_attributions(model, device, tests[key], return_metadata=True,
                n_steps=getattr(args, 'ig_steps', 64), max_steps=getattr(args, 'ig_max_steps', 256),
                batch_size=getattr(args, 'ig_batch_size', 8),
                internal_batch_size=getattr(args, 'ig_internal_batch_size', 32),
                atol=getattr(args, 'ig_atol', .001), rtol=getattr(args, 'ig_rtol', .05))
            data.update(dataset=np.asarray(args.dataset), model_name=np.asarray(args.model_name),
                        configuration=np.asarray(path.stem), fold=np.asarray(args.fold), seed=np.asarray(args.seed),
                        source_config=np.asarray(str(path.resolve())),
                        source_config_sha256=np.asarray(hashlib.sha256(path.read_bytes()).hexdigest()),
                        split_sha256=np.asarray(json.dumps(split_hashes, sort_keys=True)),
                        protocol=np.asarray(json.dumps(dict(steps=getattr(args, 'ig_steps', 64),
                            max_steps=getattr(args, 'ig_max_steps', 256), atol=getattr(args, 'ig_atol', .001),
                            rtol=getattr(args, 'ig_rtol', .05), batch_size=getattr(args, 'ig_batch_size', 8),
                            internal_batch_size=getattr(args, 'ig_internal_batch_size', 32)), sort_keys=True)))
            dataset = tests[key].dataset
            if hasattr(dataset, 'mean_value'):
                data.update(normalization_mean=np.asarray(dataset.mean_value),
                            normalization_std=np.asarray(dataset.std_value),
                            original_sampling_rate=np.asarray(dataset.old_sr),
                            sampling_rate=np.asarray(dataset.new_sr))
            filename = f'{args.dataset}_{path.stem}_fold_{args.fold}_seed_{args.seed}.npz'
            np.savez_compressed(out / filename, **data)
            print(f'Attributions: {out / filename}')
    print(f'Results: {run_dir}. Do not select configurations using test scores.')


if __name__ == '__main__':
    main(get_args())
