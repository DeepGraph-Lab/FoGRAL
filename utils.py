############以下是utils文件
#以下是utils.py

import os
import logging
import torch


def get_logger(log_path: str) -> logging.Logger:
    """
    返回同时输出到文件和控制台的 logger。
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # 文件 Handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    # 控制台 Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 格式化
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 避免重复添加 handler
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def save_model(model: torch.nn.Module, path: str, logger: logging.Logger) -> None:
    """
    保存 PyTorch 模型的 state_dict。
    - model: 要保存的 nn.Module。
    - path: 保存文件路径（含文件名，如 './ckpt/model.pth'）。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")


def load_model(model: torch.nn.Module, path: str, device: torch.device, logger: logging.Logger) -> torch.nn.Module:
    """
    从指定路径加载 state_dict 到 model，并返回 model。
    - model: 已实例化但未加载参数的 nn.Module。
    - path: 保存的 state_dict 文件路径。
    - device: 加载到的设备（cpu 或 cuda）。
    """
    if os.path.isfile(path):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        logger.info(f"Loaded model parameters from {path}")
    else:
        logger.warning(f"No checkpoint found at {path}, using fresh parameters.")
    return model
def create_model(Model_class, config, logger):
    model = Model_class(config)
    logger.info("Created model with fresh parameters.")
    return model