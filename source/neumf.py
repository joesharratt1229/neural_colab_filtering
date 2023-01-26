import torch
import torch.nn as nn

import model
import gmf 

class NeuMF(nn.Module):
    def __init__(self, mlp_mod, gmf_mod, output_range = (1,5)):
        super().__init__()
        self.mlp_mod = mlp_mod
        self.gmf_mod = gmf_mod
        self.embedding_user_mlp = mlp_mod.embedding_user
        self.embedding_item_mlp = mlp_mod.embedding_item
        self.embedding_user_mf = gmf_mod.embedding_user
        self.embedding_item_mf = gmf_mod.embedding_item

        self.mlp = mlp_mod.MLP[:9]
        self.affline_output = nn.Linear(self.mlp[6].out_features + 
                              gmf_mod.config['embed_dim'], 1)
        self.sigmoid = nn.Sigmoid()

        self.norm_min = min(output_range)
        self.norm_range = abs(output_range[1] - output_range[0]) +1
        self._load_pretrained()
        self.config = { "mlp_config": mlp_mod.config, 
                        "gmf_config": gmf_mod.config}

    def forward(self, user_indices, item_indices):
        user_embed_mlp = self.embedding_user_mlp(user_indices)
        item_embed_mlp = self.embedding_item_mlp(item_indices)
        user_embed_mf = self.embedding_user_mf(user_indices)
        item_embed_mf = self.embedding_item_mf(item_indices)

        x_mlp = torch.cat([user_embed_mlp, item_embed_mlp], dim = -1)
        x_mf = torch.mul(user_embed_mf, item_embed_mf)

        x_mlp = self.mlp(x_mlp)
        output = torch.cat([x_mlp, x_mf], dim = -1)
        output = self.affline_output(output)
        output = self.sigmoid(output)
        normalised_output = output * self.norm_range + self.norm_min
        return normalised_output

    def _load_pretrained(self):
        self.affline_output.weight.data = (0.5 * 
                        torch.cat([self.mlp_mod.MLP[9].weight.data,
                        self.gmf_mod.layers[0].weight.data], dim = -1))
        self.affline_output.bias.data = (0.5 * self.mlp_mod.MLP[9].bias.data + 
                                 self.gmf_mod.layers[0].bias.data)

    
