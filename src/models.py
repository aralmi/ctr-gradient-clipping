import torch
from torch import nn


class CTRMLP(nn.Module):
    def __init__(self, cardinalities, numeric_features, embedding_dim, dropout):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(size, embedding_dim) for size in cardinalities
        )
        width = len(cardinalities) * embedding_dim + numeric_features
        self.layers = nn.Sequential(
            nn.Linear(width, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, categories, numbers):
        embeddings = [table(categories[:, i]) for i, table in enumerate(self.embeddings)]
        return self.layers(torch.cat([*embeddings, numbers], dim=1)).squeeze(1)


class CTRLogistic(nn.Module):
    def __init__(self, cardinalities, numeric_features):
        super().__init__()
        self.categories = nn.ModuleList(nn.Embedding(size, 1) for size in cardinalities)
        self.numbers = nn.Linear(numeric_features, 1)
        for table in self.categories:
            nn.init.zeros_(table.weight)
        nn.init.zeros_(self.numbers.weight)
        nn.init.zeros_(self.numbers.bias)

    def forward(self, categories, numbers):
        terms = [table(categories[:, i]) for i, table in enumerate(self.categories)]
        return (self.numbers(numbers) + torch.stack(terms).sum(dim=0)).squeeze(1)
