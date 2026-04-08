import utils
import torch
import os


# ------------------------------------------Load the dataset-------------------------------------------------------
def get_datasets(args, data_name):
    cur_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_path = cur_path + '/preprocessing'+'/'+data_name+'/'
    dataset_train_folds, dataset_val_folds, dataset_test_folds = {}, {}, {}

    folds = 5
    for fold in range(1, folds + 1):
        dataset_train = utils.CustomDataLoader(target_path + '/json_'+str(fold)+'/train.json', args.sampling_rate)
        dataset_val = utils.CustomDataLoader(target_path + '/json_'+str(fold)+'/val.json', args.sampling_rate)
        dataset_test = utils.CustomDataLoader(target_path + '/json_'+str(fold)+'/test.json', args.sampling_rate)
        dataset_train_folds['fold_' + str(fold)] = dataset_train
        dataset_val_folds['fold_' + str(fold)] = dataset_val
        dataset_test_folds['fold_' + str(fold)] = dataset_test
    ch_names = dataset_test.get_ch_names()
    return dataset_train_folds, dataset_val_folds, dataset_test_folds, ch_names


def get_dataload(args):
    # get dataset
    dataset_train_folds, dataset_val_folds, dataset_test_folds, ch_names = get_datasets(args, args.dataset)
    dataloader_train_folds, dataloader_val_folds, dataloader_test_folds = {}, {}, {}
    for fold in range(1, len(dataset_train_folds)+1):
        dataset_train = dataset_train_folds['fold_'+str(fold)]
        dataset_val = dataset_val_folds['fold_'+str(fold)]
        dataset_test = dataset_test_folds['fold_'+str(fold)]
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train, batch_size=int(args.batch_size), num_workers=args.num_workers, pin_memory=True, drop_last=True)
        data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val, batch_size=int(args.batch_size),num_workers=args.num_workers,pin_memory=True, drop_last=False)
        data_loader_test = torch.utils.data.DataLoader(dataset_test, sampler=sampler_test, batch_size=int(args.batch_size),num_workers=args.num_workers,pin_memory=True, drop_last=False)
        dataloader_train_folds['fold_' + str(fold)] = data_loader_train
        dataloader_val_folds['fold_' + str(fold)] = data_loader_val
        dataloader_test_folds['fold_' + str(fold)] = data_loader_test

    return dataloader_train_folds, dataloader_val_folds, dataloader_test_folds, len(ch_names), ch_names
