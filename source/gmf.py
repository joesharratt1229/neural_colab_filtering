import torch
import torch.nn as nn

class GMF(nn.Module):
    def __init__(self, num_users, num_films, embed_dim=32,
    output_range = (1, 5)):
        super().__init__()
        self.num_users = num_users
        self.num_films = num_films
        self.embedding_user = nn.Embedding(num_users, embed_dim)
        self.embedding_item = nn.Embedding(num_films, embed_dim)
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid())
        
        self.norm_min = min(output_range)
        self.norm_range = abs(output_range[1] - output_range[0]) +1
        
        self.config = { 'num_users': num_users, 
        'num_films': num_films, 
        'embed_dim': embed_dim }

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        x = torch.mul(user_embedding, item_embedding)
        x = self.layers(x)
        normalised_output = x * self.norm_range + self.norm_min
        return normalised_output