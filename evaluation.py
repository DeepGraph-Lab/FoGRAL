import numpy as np
from sklearn import metrics

def evaluate(predict, label):

    try:
        import torch
        if isinstance(predict, torch.Tensor):
            predict = predict.detach().cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.detach().cpu().numpy()
    except ImportError:
        pass


    predict = np.asarray(predict).ravel()
    label = np.asarray(label).ravel()

    aupr = metrics.average_precision_score(label, predict)
    auroc = metrics.roc_auc_score(label, predict)
    return {'aupr': aupr, 'auroc': auroc}
