#以下是evaluation文件
import numpy as np
from sklearn import metrics

def evaluate(predict, label):
    """
    计算并返回预测结果的 AUPR 和 AUROC。

    参数：
    - predict: 可为 numpy 数组或 PyTorch 张量的预测分数
    - label:   可为 numpy 数组或 PyTorch 张量的真实标签

    返回：
    字典，包含 'aupr' 和 'auroc' 两个键。
    """
    # 支持 PyTorch Tensor 输入
    try:
        import torch
        if isinstance(predict, torch.Tensor):
            predict = predict.detach().cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.detach().cpu().numpy()
    except ImportError:
        pass

    # 转为一维 numpy 数组
    predict = np.asarray(predict).ravel()
    label = np.asarray(label).ravel()

    aupr = metrics.average_precision_score(label, predict)
    auroc = metrics.roc_auc_score(label, predict)
    return {'aupr': aupr, 'auroc': auroc}