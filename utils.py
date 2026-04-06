import os
import logging
import torch
from torch.optim.lr_scheduler import CyclicLR


def get_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def save_model(model: torch.nn.Module, path: str, logger: logging.Logger) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")


def load_model(model: torch.nn.Module, path: str, device: torch.device, logger: logging.Logger) -> torch.nn.Module:
   
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


def configure_optimizers(model, config):
    
    hetero_params = []
    other_decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if ('mask_dd' in name) or ('mask_pp' in name) or name.endswith('.bias') or ('temp' in name):
            no_decay_params.append(param)
        elif ('conv_dp' in name) or ('conv_pd' in name):
            hetero_params.append(param)
        else:
            other_decay_params.append(param)

    param_groups = []
    het_wd = 0.01
    other_wd = config.get('weight_decay', 0.0001)

    if len(hetero_params) > 0:
        param_groups.append({
            'params': hetero_params,
            'weight_decay': het_wd
        })

    if len(other_decay_params) > 0:
        param_groups.append({
            'params': other_decay_params,
            'weight_decay': other_wd
        })

    if len(no_decay_params) > 0:
        param_groups.append({
            'params': no_decay_params,
            'weight_decay': 0.0001
        })

    optim = torch.optim.Adam(param_groups, lr=config['lr'])

    sched = None
    if config.get('use_cyclic', False):
        sched = CyclicLR(
            optim,
            base_lr=config['lr'] * config.get('lr_scale'),
            max_lr=config['lr'],
            step_size_up=config['step_size_up'],
            mode='exp_range',
            gamma=0.999,
            cycle_momentum=False
        )

    return optim, sched
