import torch
import torch.nn as nn

class GMF(nn.Module):
    def __init__(self, num_users, num_films, embed_dim=32):
        super().__init__()
        self.num_users = num_users
        self.num_films = num_films
        self.embedding_user = nn.Embedding(num_users, embed_dim)
        self.embedding_item = nn.Embedding(num_films, embed_dim)
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid())
        self.config = { 'num_users': num_users, 
        'num_films': num_films, 
        'embed_dim': embed_dim }

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        x = torch.mul(user_embedding, item_embedding)
        x = self.layers(x)
        return x