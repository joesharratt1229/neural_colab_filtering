import torch
import torch.nn as nn

import model
import gmf 

class NeuMF(nn.Module):
    def __init__(self, config, mlp_mod, gmf_mod):
        super().__init__()
        self.user_embedding_ncf = nn.Embedding(config.num_users, config.latent_dim)
        self.item_embedding_ncf = nn.Embedding(config.num_items, config.latent_dim)
        self.user_embedding_mlp = nn.Embedding(config.num_users, config.hidden_dim)
        self.item_embedding_mlp = nn.Embedding(config.num_items, config.hidden_dim)
        