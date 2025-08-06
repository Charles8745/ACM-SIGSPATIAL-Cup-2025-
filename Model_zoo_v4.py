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
            traj = user_df[['x', 'y', 't', 'working_day', 'delta_t']].values
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

class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, uid_dim, uid_embed_dim, hidden_dim, max_len, num_layers=1):
        super().__init__()
        self.uid_embedding = nn.Embedding(uid_dim, uid_embed_dim)
        self.encoder_rnn = nn.LSTM(input_dim + uid_embed_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim * 2)
        self.decoder_rnn = nn.LSTM(latent_dim + uid_embed_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.decoder_fc = nn.Linear(hidden_dim, input_dim)
        self.max_len = max_len

    def encode(self, x, uid, mask):
        # x: (batch, max_len, input_dim)
        uid_embed = self.uid_embedding(uid)  # (batch, uid_embed_dim)
        uid_embed_expand = uid_embed.unsqueeze(1).expand(-1, x.size(1), -1)
        x_cat = torch.cat([x, uid_embed_expand], dim=-1)
        lengths = mask.sum(dim=1).long().cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(x_cat, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.encoder_rnn(packed)
        h = h_n[-1]  # (batch, hidden_dim)
        h = self.encoder_fc(h)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, uid):
        # z: (batch, latent_dim)
        uid_embed = self.uid_embedding(uid)  # (batch, uid_embed_dim)
        z_cat = torch.cat([z, uid_embed], dim=-1)  # (batch, latent_dim + uid_embed_dim)
        z_cat_seq = z_cat.unsqueeze(1).expand(-1, self.max_len, -1)  # (batch, max_len, latent_dim + uid_embed_dim)
        out, _ = self.decoder_rnn(z_cat_seq)
        out = self.decoder_fc(out)  # (batch, max_len, input_dim)
        return out

    def forward(self, x, uid, mask):
        mu, logvar = self.encode(x, uid, mask)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, uid)
        return x_hat, mu, logvar

def cvae_loss(x_hat, x, mu, logvar, mask):
    # x_hat, x: (batch, max_len, input_dim)
    # mask: (batch, max_len)
    # 重建 loss，只計算有效點
    recon_loss = ((x_hat - x) ** 2).sum(dim=-1)  # (batch, max_len)
    recon_loss = (recon_loss * mask).sum() / mask.sum()

    # KL loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)  # (batch,)
    kl_loss = kl_loss.mean()

    return recon_loss + kl_loss, recon_loss, kl_loss

if __name__ == "__main__":
    # 資料準備
    raw_x_train_df = pd.read_csv(f'./Training_Testing_Data/A_x_train.csv')
    raw_y_train_df = pd.read_csv(f'./Training_Testing_Data/A_y_train.csv')
    raw_train_df = pd.concat([raw_x_train_df, raw_y_train_df], ignore_index=True)
    raw_feature_df = pd.read_csv(f'./Stability/A_features.csv')
    raw_cluster_df = pd.read_csv(f'./Stability/A_activity_space.csv')

    valid_uid_list = raw_cluster_df[raw_cluster_df['cluster'] == 1]['uid'].unique().tolist()
    valid_uid_list = valid_uid_list[100:]  # 可調整
    dataset = TrajectoryDataset(raw_train_df, valid_uid_list, max_len=50)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 模型初始化
    input_dim = 5
    latent_dim = 32
    uid_dim = max(valid_uid_list) + 1
    uid_embed_dim = 32
    hidden_dim = 128
    max_len = 50
    num_layers = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(input_dim, latent_dim, uid_dim, uid_embed_dim, hidden_dim, max_len, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 訓練迴圈
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, mask, lengths, uid in dataloader:
            x = x.to(device)
            mask = mask.to(device)
            uid = torch.tensor(uid, dtype=torch.long).to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(x, uid, mask)
            loss, recon_loss, kl_loss = cvae_loss(x_hat, x, mu, logvar, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")