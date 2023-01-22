import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_films, embed_dim=32,
    hidden_layers = (64, 32, 16, 8), output_range = (1, 5),
    dropout_rate = None ):
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_films = num_films
        self.embedding_user = nn.Embedding(num_users, embed_dim)
        self.embedding_item = nn.Embedding(num_films, embed_dim)
        self.MLP = self._MLP(hidden_layers, embed_dim, dropout_rate)
        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)
        self._init_params()

        assert output_range and len(output_range) == 2
        self.norm_min = min(output_range)
        self.norm_range = abs(output_range[1] - output_range[0]) +1

    def _MLP(self, hidden_layer_unit, embed_unit, dropout):
        assert hidden_layer_unit[0] == 2 * embed_unit
        hidden_layers = []
        initial_input = hidden_layer_unit[0]
        for layer in hidden_layer_unit[1:]:
            hidden_layers.append(nn.Linear(initial_input, layer))
            hidden_layers.append(nn.ReLU())
            if dropout:
                hidden_layers.append(nn.Dropout(dropout))
            initial_input = layer
        
        hidden_layers.append(nn.Linear(hidden_layer_unit[-1], 1))
        hidden_layers.append(nn.Sigmoid())
        return nn.Sequential(*hidden_layers)

    def _init_params(self):
        def _weights_init(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)
            self.embedding_user.weight.uniform_(-.05, .05)
            self.embedding_item.weight.uniform_(-.05, .05)
            self.MLP.apply(_weights_init)
    
    
    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        x = torch.cat([user_embedding, item_embedding], dim=-1)

        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.MLP(x)
        normalised_output = x * self.norm_range + self.norm_min
        return normalised_output







