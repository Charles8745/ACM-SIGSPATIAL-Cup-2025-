import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class data_preprocessing:
    """
    對單一一個 uid 的訓練集和測試集進行特徵合併和處理
    """
    @staticmethod
    def load_merge_feature(train_df_path, test_df_path,feature_df_path, uid):
        train_df = pd.read_csv(train_df_path)
        test_df = pd.read_csv(test_df_path)
        train_df = train_df[train_df['uid']== uid]
        test_df = test_df[test_df['uid']== uid]
        feature_df = pd.read_csv(feature_df_path)
        # 合併訓練集和測試集，並貼上特徵    
        total_df = pd.concat([train_df, test_df], ignore_index=True)
        merged_df = total_df.merge(feature_df, on="uid", how="left")
        merged_df = merged_df.sort_values(['uid', 'd', 't'])

        # 計算delta t
        merged_df['delta_t'] = merged_df.groupby('uid')['t'].diff()
        # 處理跨天：明天第一個的 t + 48 - 今天最後一個的 t
        cross_day = merged_df['d'] != merged_df.groupby('uid')['d'].shift()
        # 找出跨天的 index
        cross_day_idx = merged_df.index[cross_day]
        for idx in cross_day_idx:
            if idx == 0:
                merged_df.at[idx, 'delta_t'] = 1
            else:
                prev_idx = idx - 1
                merged_df.at[idx, 'delta_t'] = merged_df.at[idx, 't'] + 48 - merged_df.at[prev_idx, 't']
        merged_df['delta_t'] = merged_df['delta_t'].fillna(1)

        # 選擇要得feature列
        feature_cols = ['t', 'x', 'y', 'day_of_week', 'working_day', 'delta_t']
        result_df = merged_df[feature_cols].astype(int)
        sequences = [row for row in result_df.values]

        return result_df, sequences

class TrajDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.float32)  # shape: (seq_len, feature_dim)
    
class TLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input gate
        self.Wi = nn.Linear(input_dim, hidden_dim)
        self.Ui = nn.Linear(hidden_dim, hidden_dim)
        # Forget gate
        self.Wf = nn.Linear(input_dim, hidden_dim)
        self.Uf = nn.Linear(hidden_dim, hidden_dim)
        # Output gate
        self.Wog = nn.Linear(input_dim, hidden_dim)
        self.Uog = nn.Linear(hidden_dim, hidden_dim)
        # Cell candidate
        self.Wc = nn.Linear(input_dim, hidden_dim)
        self.Uc = nn.Linear(hidden_dim, hidden_dim)
        # Decomposition
        self.W_decomp = nn.Linear(hidden_dim, hidden_dim)
        # Time mapping
        self.c2 = 2.7183

    def map_elapse_time(self, t):
        # t: [batch, 1]
        T = 1.0 / t
        T = T.repeat(1, self.hidden_dim)  # [batch, hidden_dim]
        return T

    def forward(self, x, t, prev_hidden, prev_cell):
        # x: [batch, input_dim], t: [batch, 1]
        # prev_hidden, prev_cell: [batch, hidden_dim]
        T = self.map_elapse_time(t)
        C_ST = torch.tanh(self.W_decomp(prev_cell))
        C_ST_dis = T * C_ST
        prev_cell = prev_cell - C_ST + C_ST_dis

        i = torch.sigmoid(self.Wi(x) + self.Ui(prev_hidden))
        f = torch.sigmoid(self.Wf(x) + self.Uf(prev_hidden))
        o = torch.sigmoid(self.Wog(x) + self.Uog(prev_hidden))
        C = torch.tanh(self.Wc(x) + self.Uc(prev_hidden))
        Ct = f * prev_cell + i * C
        current_hidden = o * torch.tanh(Ct)
        return current_hidden, Ct


if __name__ == "__main__":
    train_path = './Training_Testing_Data/A_x_train.csv'
    test_path = './Training_Testing_Data/A_x_test.csv'
    feature_path = './Stability/A_features.csv'
    target_uid = 1
    result_df, sequences  = data_preprocessing.load_merge_feature(train_path, 
                                                                  test_path, 
                                                                  feature_path, 
                                                                  target_uid)
    print(f"Number of sequences: {len(sequences)}")
    print(f"Shape of first sequence: {sequences[0].shape}")
