import torch


class filmDataset(torch.utils.data.Dataset):
    def __init__(self, user_list, film_list, rating_list):
        super(filmDataset, self).__init__()
        self.user_list = user_list
        self.film_list = film_list
        self.rating_list = rating_list

    def __len__(self):
        return len(self.user_list)
    
    def __getitem__(self, index):
        user = self.user_list[index]
        film = self.film_list[index]
        rating = self.rating_list[index]
        return (torch.tensor(user), 
                torch.tensor(film), 
                torch.tensor(rating))