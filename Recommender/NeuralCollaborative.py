# script
from matplotlib.pyplot import title
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

np.random.seed(123)


class MovieTrainDataset(Dataset):
    def __init__(self, ratings, all_movieIds):
        self.users, self.items, self.labels = self.get_dataset(ratings, all_movieIds)

    def __len__(self):
        return len(self.users)
  
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]
    
    def get_dataset(self, ratings, all_movieIds):
        users, items, labels = [], [], []
        user_item_set = set(zip(ratings['userId'], ratings['movieId']))

        num_negatives = 4
        for u, i in user_item_set:
            users.append(u)
            items.append(i)
            labels.append(1)
            for _ in range(num_negatives):
                negative_item = np.random.choice(all_movieIds)
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_movieIds)
                users.append(u)
                items.append(negative_item)
                labels.append(0)
                
        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)
    
    
class NCF(pl.LightningModule):

    def __init__(self, num_users, num_items, ratings, all_movieIds):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=8)
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)
        self.ratings = ratings
        self.all_movieIds = all_movieIds
        
    def forward(self, user_input, item_input):
        
        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        # Concat the two embedding layers
        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        # Pass through dense layer
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))

        # Output layer
        pred = nn.Sigmoid()(self.output(vector))

        return pred

    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(MovieTrainDataset(self.ratings, self.all_movieIds),
                            batch_size=512)
        
ratings = pd.read_csv('Dataset/small_ratings.csv')    
train_ratings = pd.read_csv('Dataset/trained_ratings.csv')

all_movieIds = pd.read_csv('Dataset/neuralAllMovie.csv')
all_movieIds = all_movieIds['0'].to_list()

model = NCF(9181, 175476, train_ratings, all_movieIds)
model.load_state_dict(torch.load('Model/neural_collaborative_filtering.th'))


movies = pd.read_csv('Dataset/movies.csv')

def recommend(user_id, topN):
    q_reacted = ratings['userId'] == user_id
    reacted = set(ratings[q_reacted]['movieId'].to_list())
    uninteracted_item = set(all_movieIds) - set(reacted)

    predictions = \
        np.squeeze(
            model\
                (
                    torch.tensor([1] *len(uninteracted_item)), 
                    torch.tensor(list(uninteracted_item))
                )\
                .detach().numpy())
    top_result = list(np.argsort(predictions)[::-1][0:topN])
        
    recommendation = []
    for index in top_result:
        actual_id = all_movieIds[index]
        q_title = movies['movie_id'] == actual_id
        title = movies[q_title]['movie_title'].iloc[0]
        recommendation.append(title)
    
    viewed = []
    for index in reacted:
        q_title = movies['movie_id'] == index
        title = movies[q_title]['movie_title'].iloc[0]
        viewed.append(title)
        
    return viewed, recommendation

