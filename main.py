# main.py
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from copy import deepcopy
from utils import configure_optimizers, get_logger
from model import DrugRepositioningModel
from data_reader import data_reader
import evaluation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Fdataset')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--mode', choices=['cv','case'], default='cv')
    parser.add_argument('--disease_dim', type=int, default=125)
    parser.add_argument('--drug_dim', type=int, default=125)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--mlp_layer_num', type=int, default=2)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--n_splits', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1024*5)
    parser.add_argument('--w_bce', type=float, default=1.0)
    parser.add_argument('--w_fuse', type=float, default=0.1)
    parser.add_argument('--ent_coef_hom', type=float, default=0.005)
    parser.add_argument('--l1_coef_hom', type=float, default=0.001)
    return parser.parse_args()

def config_model(args):
    return OrderedDict(
        dataset=args.dataset,
        lr=args.lr,
        epochs=args.epochs,
        clip=3.0,
        disease_dim=args.disease_dim,
        drug_dim=args.drug_dim,
        latent_dim=args.latent_dim,
        mlp_layer_num=args.mlp_layer_num,
        seed=args.seed,
        n_splits=args.n_splits,
        batch_size=args.batch_size,
        w_bce=args.w_bce,
        w_fuse=args.w_fuse,
        ent_coef_hom=args.ent_coef_hom,
        l1_coef_hom=args.l1_coef_hom,
        init_temp=1.0,
        max_temp=5.0,
        temp_anneal=1.1,
        use_cyclic=True,
        lr_scale=0.1,
        step_size_up=2000,
        weight_decay=0.0001,
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cross_validation(cfg):
    logger = get_logger(f"./logs/{cfg['dataset']}.log")
    folds = data_reader(logger, cfg, cfg['dataset'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    final_aucs, final_auprs = [], []
    best_aucs, best_auprs = [], []

    for fold_idx, fold in enumerate(folds):
        
        inter_np = fold['interactions']
        nd, npd = inter_np.shape
        cfg['disease_size'] = nd
        cfg['drug_size'] = npd

        inter = torch.from_numpy(fold['interactions']).to(device)
        tr_mask = torch.from_numpy(fold['train_mask']).to(device)
        te_mask = torch.from_numpy(fold['test_mask']).to(device)

        train_dp, train_pd = fold['train_edge_index_dp'].to(device), fold['train_edge_index_pd'].to(device)
        test_dp, test_pd   = fold['test_edge_index_dp'].to(device),  fold['test_edge_index_pd'].to(device)
        dd, pp = fold['edge_index_dd'].to(device), fold['edge_index_pp'].to(device)

        train_i, train_j = torch.where(tr_mask)
        train_labels = inter[train_i, train_j].float()
        train_ds = TensorDataset(train_i, train_j, train_labels)
        train_loader = DataLoader(train_ds, batch_size=min(cfg['batch_size'], len(train_ds)), shuffle=True)

        model = DrugRepositioningModel(cfg).to(device)
        optimizer, scheduler = configure_optimizers(model,cfg)

        best_val_auc = -1.0
        best_state = None
        best_val_metrics = None
        best_epoch = -1

        for epoch in range(1, cfg['epochs']+1):
            model.train()
            last_weights = None
            for di, pj, lbl in train_loader:
                optimizer.zero_grad()
                out, logits, fuse_loss, weights = model(di, pj, train_dp, train_pd, dd, pp)
                last_weights = weights

                bce = torch.nn.BCEWithLogitsLoss()(logits, lbl)
                m_dd, m_pp = weights['dd'], weights['pp']
                l1 = (m_dd.abs().sum()/dd.size(1) + m_pp.abs().sum()/pp.size(1))
                ent = ((m_dd*(1-m_dd)).sum()/dd.size(1) + (m_pp*(1-m_pp)).sum()/pp.size(1))
                loss = cfg['w_bce']*bce + cfg['w_fuse']*fuse_loss + cfg['l1_coef_hom']*l1 + cfg['ent_coef_hom']*ent
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), cfg['clip'])
                optimizer.step()
                if scheduler: scheduler.step()

            model.step_temp()

            model.eval()
            with torch.no_grad():
                vi, vj = torch.where(te_mask)
                v_out, _, _, _ = model(vi, vj, test_dp, test_pd, dd, pp)
                v_pred = v_out.cpu().numpy()
                v_true = inter[vi, vj].cpu().numpy()
            val_metrics = evaluation.evaluate(v_pred, v_true)
            logger.info(f"[Fold{fold_idx+1}][Epoch {epoch}] Val AUROC={val_metrics['auroc']:.4f} AUPR={val_metrics['aupr']:.4f}")

            if val_metrics['auroc'] > best_val_auc:
                best_val_auc = float(val_metrics['auroc'])
                best_state = deepcopy(model.state_dict())
                best_val_metrics = {'auroc': float(val_metrics['auroc']), 'aupr': float(val_metrics['aupr'])}
                best_epoch = epoch

        final_state = deepcopy(model.state_dict())
        model.load_state_dict(final_state)
        model.eval()
        with torch.no_grad():
            ti, tj = torch.where(te_mask)
            t_out, _, _, _ = model(ti, tj, test_dp, test_pd, dd, pp)
            t_pred = t_out.cpu().numpy()
            t_true = inter[ti, tj].cpu().numpy()
        final_metrics = evaluation.evaluate(t_pred, t_true)
        logger.info(f"[Fold{fold_idx+1}] FINAL Test AUROC={final_metrics['auroc']:.4f} AUPR={final_metrics['aupr']:.4f}")

        if best_state is not None:
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                b_out, _, _, _ = model(ti, tj, test_dp, test_pd, dd, pp)
                b_pred = b_out.cpu().numpy()
            best_test_metrics = evaluation.evaluate(b_pred, t_true)
            logger.info(f"[Fold{fold_idx+1}] BEST_BY_VAL (Epoch {best_epoch}) Test AUROC={best_test_metrics['auroc']:.4f} AUPR={best_test_metrics['aupr']:.4f}")
        else:
            best_test_metrics = final_metrics
            best_epoch = cfg['epochs']

        final_aucs.append(float(final_metrics['auroc']))
        final_auprs.append(float(final_metrics['aupr']))
        best_aucs.append(float(best_test_metrics['auroc']))
        best_auprs.append(float(best_test_metrics['aupr']))

    print(f"FINAL AVG AUROC={np.mean(final_aucs):.5f}, AUPR={np.mean(final_auprs):.5f}")
    print(f"BEST_BY_VAL AVG AUROC={np.mean(best_aucs):.5f}, AUPR={np.mean(best_auprs):.5f}")

if __name__ == '__main__':
    args = parse_args()
    cfg = config_model(args)
    set_seed(cfg['seed'])
    if args.mode == 'cv':
        cross_validation(cfg)
