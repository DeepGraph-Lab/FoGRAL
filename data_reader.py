# data_reader_preprocessed.py
import os
import torch

def data_reader(logger, config, dataset='Fdataset'):
    data_dir = f'./preprocessed_folds/{dataset}'
    n_splits = config.get('n_splits', 10)
    folds = []
    for fold_idx in range(n_splits):
        fold_path = os.path.join(data_dir, f'fold_{fold_idx}.pt')
        if not os.path.exists(fold_path):
            raise FileNotFoundError(f"Fold file not found: {fold_path}")
        fold = torch.load(fold_path)
        folds.append(fold)
        logger.info(f"[INFO] Loaded fold {fold_idx} from {fold_path}")
    return folds