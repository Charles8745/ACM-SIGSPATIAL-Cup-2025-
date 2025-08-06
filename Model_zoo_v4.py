import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time 
import geobleu
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class TrajectoryDataset(Dataset):
    def __init__(self, df, uid_list, max_len=400):
        self.uid_list = uid_list
        self.max_len = max_len
        self.data = []
        self.lengths = []
        self.masks = []
        for uid in uid_list:
            user_df = df[df['uid'] == uid]
            user_df = user_df[user_df['d'] <= 60]
            traj = user_df[['t', 'x', 'y', 'working_day', 'delta_t']].values
            length = len(traj)
            self.lengths.append(length)
            # padding
            if length < max_len:
                pad = np.zeros((max_len - length, traj.shape[1]))
                traj = np.vstack([traj, pad])
                mask = np.concatenate([np.ones(length), np.zeros(max_len - length)])
            else:
                traj = traj[:max_len]
                mask = np.ones(max_len)
            self.data.append(torch.tensor(traj, dtype=torch.float32))
            self.masks.append(torch.tensor(mask, dtype=torch.float32))
        self.data = torch.stack(self.data)
        self.masks = torch.stack(self.masks)

    def __len__(self):
        return len(self.uid_list)

    def __getitem__(self, idx):
        return self.data[idx], self.masks[idx], self.lengths[idx], self.uid_list[idx]


if __name__ == "__main__":
    raw_train_df = pd.read_csv(f'./Training_Testing_Data/A_x_train.csv')
    raw_feature_df = pd.read_csv(f'./Stability/A_features.csv')
    raw_cluster_df = pd.read_csv(f'./Stability/A_activity_space.csv')

    valid_uid_list = raw_cluster_df['uid'].unique().tolist()
    print(valid_uid_list[:10])