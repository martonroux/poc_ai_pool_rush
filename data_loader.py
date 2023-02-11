from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_data(path: str, batch_size: int):
    train_set = SpamEmailDataset(path, 0.8, 0)
    test_set = SpamEmailDataset(path, 1, 0.8)

    train_load = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_load = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_load, test_load


class SpamEmailDataset(Dataset):

    def __init__(self, file_name, ratio: float, start_ratio: float):
        assert 1 >= ratio > start_ratio, 'Wrong ratio for train / test set'

        file_out = pd.read_csv(file_name)

        start = round(len(file_out) * start_ratio)
        end = round(len(file_out) * ratio)

        file_out = file_out.drop('Email No.', axis=1)

        file_out = file_out.fillna(0)

        x = file_out.loc[start:end, file_out.columns != 'Prediction']
        y = file_out.loc[start:end, 'Prediction']

        # Feature Scaling
        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = np.array(y)

        self.x_train = torch.tensor(x_train, device=DEVICE)
        self.y_train = torch.tensor(y_train, device=DEVICE)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
