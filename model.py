import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch_geometric.nn import (
    GCNConv, GraphConv, TAGConv, GATConv,
    HypergraphConv, ClusterGCNConv, SimpleConv,
    SGConv, ChebConv, GatedGraphConv, ResGatedGraphConv,
    APPNP, ARMAConv, LEConv, WLConvContinuous, NNConv,
    TransformerConv, SSGConv, DNAConv, FAConv, LGConv,
    GCN2Conv
)

class ConvPool(nn.Module):
    """
    Pool of PyG convolutional layers. Supports homogeneous and heterogeneous dims.
    in_dim: int or (in_src, in_dst)
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.convs = nn.ModuleDict()  # use ModuleDict to register submodules

        # homogeneous convs (single in_dim)
        hom_convs = {
            'GCNConv': GCNConv,
            'TAGConv': TAGConv,
            'ARMAConv': ARMAConv,
            'SGConv': SGConv,
            'SSGConv': SSGConv,
            'HypergraphConv': HypergraphConv,
            'ClusterGCNConv': ClusterGCNConv,
            'SimpleConv': SimpleConv,
            'ChebConv': ChebConv,
            'GatedGraphConv': GatedGraphConv,
            'APPNP': APPNP,
            'DNAConv': DNAConv,
            'FAConv': FAConv,
            'LGConv': LGConv,
            'GCN2Conv': GCN2Conv,
        }
        # hetero convs (may expect tuple in/out sizes)
        hetero_convs = {
            'GraphConv': GraphConv,
            'LEConv': LEConv,
            'GATConv': GATConv,
            'WLConvContinuous': lambda in_c, out_c: WLConvContinuous(in_c, out_c, num_steps=5),
            'NNConv': NNConv,
            'ResGatedGraphConv': ResGatedGraphConv,
            'TransformerConv': TransformerConv,
        }

        all_convs = {**hom_convs, **hetero_convs}
        for name, ConvCls in all_convs.items():
            # Try a couple of constructor signatures to be robust
            try:
                # attempt common signature (in_dim, out_dim)
                self.convs[name] = ConvCls(in_dim, out_dim)
            except Exception:
                try:
                    # if in_dim is tuple (in_src, in_dst), try to expand
                    if isinstance(in_dim, (list, tuple)) and len(in_dim) == 2:
                        self.convs[name] = ConvCls(in_dim[0], in_dim[1], out_dim)
                    else:
                        # try passing as separate args if ConvCls expects two numbers
                        if isinstance(in_dim, (list, tuple)):
                            self.convs[name] = ConvCls(in_dim, out_dim)
                        else:
                            # final fallback: try no-arg construction (some convs have different APIs)
                            self.convs[name] = ConvCls(out_dim)
                except Exception:
                    # skip convs that are incompatible on this environment / version
                    continue

    def get_conv(self, name: str) -> nn.Module:
        if name not in self.convs:
            raise KeyError(f"Conv '{name}' not supported. Available: {list(self.convs.keys())}")
        return self.convs[name]

class DrugRepositioningModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        D, P, L = config['disease_dim'], config['drug_dim'], config['latent_dim']
        self.d_emb = nn.Embedding(config['disease_size'], D)
        self.p_emb = nn.Embedding(config['drug_size'], P)

        # config for conv types
        het_conv = config.get('het_conv', 'GraphConv')
        hom_conv = config.get('hom_conv', 'GCNConv')

        # conv pools
        pool_dp = ConvPool((D, P), L)
        pool_pd = ConvPool((P, D), L)
        pool_dd = ConvPool(D, L)
        pool_pp = ConvPool(P, L)

        # assign conv layers
        self.conv_dp = pool_dp.get_conv(het_conv)
        self.conv_pd = pool_pd.get_conv(het_conv)
        self.conv_dd = pool_dd.get_conv(hom_conv)
        self.conv_pp = pool_pp.get_conv(hom_conv)

        # mask networks for homogeneous edge weights
        self.mask_dd = nn.Sequential(
            nn.Linear(2 * D, L), nn.ReLU(), nn.Linear(L, 1)
        )
        self.mask_pp = nn.Sequential(
            nn.Linear(2 * P, L), nn.ReLU(), nn.Linear(L, 1)
        )

        # temperature
        self.temp = nn.Parameter(torch.tensor(config['init_temp']), requires_grad=False)
        self.max_temp = config['max_temp']
        self.temp_anneal = config['temp_anneal']

        # downstream MLP & predictor with SELU
        mlp_het, mlp_hom = [], []
        for _ in range(config.get('mlp_layer_num', 1)):
            mlp_het += [nn.Linear(L, L), nn.SELU()]
            mlp_hom += [nn.Linear(L, L), nn.SELU()]
        self.mlp_het = nn.Sequential(*mlp_het)
        self.mlp_hom = nn.Sequential(*mlp_hom)
        self.project = nn.Sequential(nn.Linear(2 * L, L), nn.SELU())
        self.pred = nn.Linear(L, 1)

    def forward(self, d_idx, p_idx, e_dp, e_pd, e_dd, e_pp):
        x_d, x_p = self.d_emb.weight, self.p_emb.weight
        weights = {}

        # Hetero DP
        out_dp = self.conv_dp((x_d, x_p), e_dp)
        h_dp = F.selu(out_dp)
        # Hetero PD
        out_pd = self.conv_pd((x_p, x_d), e_pd)
        h_pd = F.selu(out_pd)

        # Homo DD
        src, dst = e_dd.long()
        feats_dd = torch.cat([x_d[src], x_d[dst]], dim=1)
        w_dd = torch.sigmoid(self.mask_dd(feats_dd).squeeze() * self.temp)
        weights['dd'] = w_dd
        out_dd = self.conv_dd(x_d, e_dd, edge_weight=w_dd)
        h_dd = F.selu(out_dd)

        # Homo PP
        src, dst = e_pp.long()
        feats_pp = torch.cat([x_p[src], x_p[dst]], dim=1)
        w_pp = torch.sigmoid(self.mask_pp(feats_pp).squeeze() * self.temp)
        weights['pp'] = w_pp
        out_pp = self.conv_pp(x_p, e_pp, edge_weight=w_pp)
        h_pp = F.selu(out_pp)

        # Gather per-sample embeddings
        z_homo = h_dd[d_idx] * h_pp[p_idx]
        z_het = h_dp[p_idx] * h_pd[d_idx]
        # Apply MLPs
        z2_h = self.mlp_hom(z_homo)
        z2_t = self.mlp_het(z_het)

        # Concatenate and predict
        
        z_cat = torch.cat([z2_t, z2_h], dim=1)
        z_fuse = self.project(z_cat)
        logits = self.pred(z_fuse).squeeze(-1)
        out = torch.sigmoid(logits)

        fuse_loss = F.mse_loss(z2_t, z2_h)
        return out, logits, fuse_loss, weights

    
    
    def configure_optimizers(self, config):

        hetero_params = []
        other_decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if ('mask_dd' in name) or ('mask_pp' in name) or name.endswith('.bias') or ('temp' in name):
                no_decay_params.append(param)
            elif ('conv_dp' in name) or ('conv_pd' in name):
                hetero_params.append(param)
            else:
                other_decay_params.append(param)

        param_groups = []
        het_wd = config.get('weight_decay_het', 0.01)
        other_wd = config.get('weight_decay', 0.0001)

        if len(hetero_params) > 0:
            param_groups.append({'params': hetero_params, 'weight_decay': het_wd})
        if len(other_decay_params) > 0:
            param_groups.append({'params': other_decay_params, 'weight_decay': other_wd})
        if len(no_decay_params) > 0:
            param_groups.append({'params': no_decay_params, 'weight_decay': 0.0001})

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


    def step_temp(self):
        new_t = min(self.temp.item() * self.temp_anneal, self.max_temp)
        self.temp.data = torch.tensor(new_t, device=self.temp.device)