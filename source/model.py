import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_films, embed_dim=64,
    hidden_layers = (64, 32, 16, 8), output_range = (1, 5),
    dropout_rate = None ):
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_films = num_films
        self.embedding_user = nn.Embedding(num_users, embed_dim)
        self.embedding_item = nn.Embedding(num_films, embed_dim)
        self.MLP = self._MLP(hidden_layers, embed_dim, dropout_rate)

    def _MLP(self, hidden_layer_unit, embed_unit, dropout):
        assert hidden_layer_unit == 2 * embed_unit
        hidden_layers = []
        initial_input = hidden_layer_unit[0]
        for i in hidden_layer_unit[1:]:
            hidden_layers.append(nn.Linear(initial_input, hidden_layer_unit[i]))
            hidden_layers.append(nn.ReLU)
            if dropout:
                hidden_layers.append(nn.Dropout(dropout))
            initial_input = hidden_layer_unit[i]
        
        hidden_layers.append(nn.Linear(hidden_layer_unit[-1], 1))
        hidden_layers.append(nn.Sigmoid())
        
        return nn.Sequential(*hidden_layers)


    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        torch.cat([user_embedding, item_embedding], dim=-1)


