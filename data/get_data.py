import utils
import torch
import os


def get_hpo_dataload(args):
    """Only load the selected fold's train/validation data; never cache test data."""
    root = getattr(args, 'split_root', None) or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'preprocessing')
    fold_dir = os.path.join(root, args.dataset, f'json_{args.fold_idx}')
    result = []
    cache_bytes = 0
    for part in ('train', 'val'):
        dataset = utils.CustomDataLoader(os.path.join(fold_dir, f'{part}.json'), args.sampling_rate, return_subject_id=part == 'val')
        if args.data_cache == 'ram':
            dataset.preload()
            cache_bytes += dataset.cache_bytes
        options = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=torch.device(args.device).type == 'cuda', shuffle=part == 'train', drop_last=part == 'train')
        if args.num_workers > 0:
            options.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
        result.append(torch.utils.data.DataLoader(dataset, **options))
    return result[0], result[1], len(dataset.get_ch_names()), cache_bytes


# ------------------------------------------Load the dataset-------------------------------------------------------
def get_datasets(args, data_name, include_subject_metadata=False):
    cur_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    split_root = getattr(args, "split_root", None)
    if split_root:
        target_path = os.path.join(split_root, data_name)
    else:
        target_path = os.path.join(cur_path, 'preprocessing', data_name)
    dataset_train_folds, dataset_val_folds, dataset_test_folds = {}, {}, {}

    folds = [args.fold] if getattr(args, 'load_selected_fold_only', False) else range(1, 6)
    for fold in folds:
        fold_dir = os.path.join(target_path, 'json_' + str(fold))
        dataset_train = utils.CustomDataLoader(os.path.join(fold_dir, 'train.json'), args.sampling_rate)
        dataset_val = utils.CustomDataLoader(
            os.path.join(fold_dir, 'val.json'), args.sampling_rate,
            return_subject_id=include_subject_metadata,
        )
        dataset_test = utils.CustomDataLoader(
            os.path.join(fold_dir, 'test.json'), args.sampling_rate,
            return_subject_id=include_subject_metadata,
        )
        dataset_train_folds['fold_' + str(fold)] = dataset_train
        dataset_val_folds['fold_' + str(fold)] = dataset_val
        dataset_test_folds['fold_' + str(fold)] = dataset_test
    ch_names = dataset_test.get_ch_names()
    return dataset_train_folds, dataset_val_folds, dataset_test_folds, ch_names


def get_dataload(args, include_subject_metadata=False):
    # get dataset
    dataset_train_folds, dataset_val_folds, dataset_test_folds, ch_names = get_datasets(
        args, args.dataset, include_subject_metadata=include_subject_metadata
    )
    dataloader_train_folds, dataloader_val_folds, dataloader_test_folds = {}, {}, {}
    for key in dataset_train_folds:
        dataset_train = dataset_train_folds[key]
        dataset_val = dataset_val_folds[key]
        dataset_test = dataset_test_folds[key]
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train, batch_size=int(args.batch_size), num_workers=args.num_workers, pin_memory=True, drop_last=True)
        data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val, batch_size=int(args.batch_size),num_workers=args.num_workers,pin_memory=True, drop_last=False)
        data_loader_test = torch.utils.data.DataLoader(dataset_test, sampler=sampler_test, batch_size=int(args.batch_size),num_workers=args.num_workers,pin_memory=True, drop_last=False)
        dataloader_train_folds[key] = data_loader_train
        dataloader_val_folds[key] = data_loader_val
        dataloader_test_folds[key] = data_loader_test

    return dataloader_train_folds, dataloader_val_folds, dataloader_test_folds, len(ch_names), ch_names
