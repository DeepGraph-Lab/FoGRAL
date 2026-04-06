import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.convs = nn.ModuleDict()

        # homogeneous convs
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

        # hetero convs
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
            try:
                self.convs[name] = ConvCls(in_dim, out_dim)
            except Exception:
                try:
                    if isinstance(in_dim, (list, tuple)) and len(in_dim) == 2:
                        self.convs[name] = ConvCls(in_dim[0], in_dim[1], out_dim)
                    else:
                        if isinstance(in_dim, (list, tuple)):
                            self.convs[name] = ConvCls(in_dim, out_dim)
                        else:
                            self.convs[name] = ConvCls(out_dim)
                except Exception:
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

        het_conv = config.get('het_conv', 'GraphConv')
        hom_conv = config.get('hom_conv', 'GCNConv')

        pool_dp = ConvPool((D, P), L)
        pool_pd = ConvPool((P, D), L)
        pool_dd = ConvPool(D, L)
        pool_pp = ConvPool(P, L)

        self.conv_dp = pool_dp.get_conv(het_conv)
        self.conv_pd = pool_pd.get_conv(het_conv)
        self.conv_dd = pool_dd.get_conv(hom_conv)
        self.conv_pp = pool_pp.get_conv(hom_conv)

        self.mask_dd = nn.Sequential(
            nn.Linear(2 * D, L), nn.ReLU(), nn.Linear(L, 1)
        )
        self.mask_pp = nn.Sequential(
            nn.Linear(2 * P, L), nn.ReLU(), nn.Linear(L, 1)
        )

        self.temp = nn.Parameter(torch.tensor(config['init_temp']), requires_grad=False)
        self.max_temp = config['max_temp']
        self.temp_anneal = config['temp_anneal']

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

        z2_h = self.mlp_hom(z_homo)
        z2_t = self.mlp_het(z_het)

        z_cat = torch.cat([z2_t, z2_h], dim=1)
        z_fuse = self.project(z_cat)
        logits = self.pred(z_fuse).squeeze(-1)
        out = torch.sigmoid(logits)

        fuse_loss = F.mse_loss(z2_t, z2_h)
        return out, logits, fuse_loss, weights

    def step_temp(self):
        new_t = min(self.temp.item() * self.temp_anneal, self.max_temp)
        self.temp.data = torch.tensor(new_t, device=self.temp.device)
