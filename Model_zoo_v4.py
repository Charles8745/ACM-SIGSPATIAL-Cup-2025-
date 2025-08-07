import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time 
import geobleu
import validator_InModify as validator
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
            traj = user_df[['x', 'y', 't']].values
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
        traj = self.data[idx]           # (max_len, input_dim)
        mask = self.masks[idx]          # (max_len,)
        length = self.lengths[idx]
        uid = self.uid_list[idx]
        t_seq = traj[:, 2].long()       # 假設 t 在第3欄
        return traj, mask, length, uid, t_seq

class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, uid_dim, uid_embed_dim, hidden_dim, max_len, num_layers=1):
        super().__init__()
        # conditional embedding
        self.uid_embedding = nn.Embedding(uid_dim, uid_embed_dim)
        self.t_embedding = nn.Embedding(49, 24) # 假設 t 的範圍是 0~48 壓到24維

        self.encoder_rnn = nn.LSTM(input_dim + uid_embed_dim +24, hidden_dim, batch_first=True, num_layers=num_layers)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim * 2)
        self.decoder_rnn = nn.LSTM(latent_dim + uid_embed_dim + 24, hidden_dim, batch_first=True, num_layers=num_layers)
        self.decoder_fc = nn.Linear(hidden_dim, input_dim)
        self.max_len = max_len

    def encode(self, x, uid, t, mask):
        # x: (batch, max_len, input_dim)  # 包含 t
        # t: (batch, max_len)
        uid_embed = self.uid_embedding(uid)  # (batch, uid_embed_dim)
        t_embed = self.t_embedding(t)        # (batch, max_len, t_embed_dim)
        uid_embed_expand = uid_embed.unsqueeze(1).expand(-1, x.size(1), -1)
        x_wo_t = x[..., :2]  # 只取 x, y
        x_cat = torch.cat([x_wo_t, uid_embed_expand, t_embed], dim=-1)
        lengths = mask.sum(dim=1).long().cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(x_cat, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.encoder_rnn(packed)
        h = h_n[-1]
        h = self.encoder_fc(h)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, uid, t):
        uid_embed = self.uid_embedding(uid)
        t_embed = self.t_embedding(t)
        uid_embed_expand = uid_embed.unsqueeze(1).expand(-1, t.size(1), -1)
        z_expand = z.unsqueeze(1).expand(-1, t.size(1), -1)
        z_cat_seq = torch.cat([z_expand, uid_embed_expand, t_embed], dim=-1)
        out, _ = self.decoder_rnn(z_cat_seq)
        out = self.decoder_fc(out)  # (batch, max_len, 2)
        return out

    def forward(self, x, uid, t, mask):
        mu, logvar = self.encode(x, uid, t, mask)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, uid, t)
        return x_hat, mu, logvar

def cvae_loss(x_hat, x, mu, logvar, mask, beta=0.05):
    # x_hat, x: (batch, max_len, 2)
    # mask: (batch, max_len)
    # step 1: 只考慮 x, y
    xy = x[..., :2]  # (batch, max_len, 2)
    # step 2: 計算每個 batch, seq 位置的 (x, y) 在該序列中出現幾次
    # 這裡以 batch 維度分開計算
    weights = []
    for seq in xy.cpu().numpy():
        # seq: (max_len, 2)
        # 轉成 tuple 方便比對
        seq_tuples = [tuple(p) for p in seq]
        counts = np.array([seq_tuples.count(p) for p in seq_tuples])
        weights.append(counts)
    weights = torch.tensor(weights, dtype=torch.float32, device=x.device)  # (batch, max_len)
    # step 3: 計算加權重構 loss
    recon_loss = ((x_hat - xy) ** 2).sum(dim=-1)  # (batch, max_len)
    recon_loss = (recon_loss * weights * mask).sum() / (weights * mask).sum()
    # step 4: KL loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    kl_loss = kl_loss.mean()
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def generate_future_trajectory(model, user_train_df, user_test_df, uid, device):
    # user_train_df: 該使用者的 1~60 天資料
    # user_test_df:  該使用者的 61~75 天資料
    # uid: 該使用者的 uid
    # device: cuda/cpu
    model.eval()
    with torch.no_grad():
        # 準備訓練資料
        traj = user_train_df[['x', 'y', 't']].values
        length = len(traj)
        max_len = model.max_len
        if length < max_len:
            pad = np.zeros((max_len - length, traj.shape[1]))
            traj = np.vstack([traj, pad])
            mask = np.concatenate([np.ones(length), np.zeros(max_len - length)])
        else:
            traj = traj[:max_len]
            mask = np.ones(max_len)
        x = torch.tensor(traj, dtype=torch.float32).unsqueeze(0).to(device)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
        uid_tensor = torch.tensor([uid], dtype=torch.long).to(device)
        t_seq_train = torch.tensor(traj[:, 2], dtype=torch.long).unsqueeze(0).to(device)  # (1, max_len)

        # 編碼得到 z
        mu, logvar = model.encode(x, uid_tensor, t_seq_train, mask)
        z = model.reparameterize(mu, logvar)

        # 生成未來軌跡，這裡用 test 的 t 作為條件
        gen_len = user_test_df.shape[0]
        t_seq_gen = torch.tensor(user_test_df['t'].values, dtype=torch.long).unsqueeze(0).to(device)  # (1, gen_len)
        future_traj = model.decode(z, uid_tensor, t_seq_gen)  # (1, gen_len, input_dim)
        return future_traj.squeeze(0).cpu().numpy()

if __name__ == "__main__":
    # 資料準備
    raw_x_train_df = pd.read_csv(f'./Training_Testing_Data/A_x_train.csv')
    raw_y_train_df = pd.read_csv(f'./Training_Testing_Data/A_y_train.csv')
    raw_train_df = pd.concat([raw_x_train_df, raw_y_train_df], ignore_index=True)
    raw_feature_df = pd.read_csv(f'./Stability/A_features.csv')
    raw_cluster_df = pd.read_csv(f'./Stability/A_activity_space.csv')
    valid_uid_list = raw_cluster_df[raw_cluster_df['cluster'] == 1]['uid'].unique().tolist()
    valid_uid_list = valid_uid_list
    print(f"有效的使用者數量: {len(valid_uid_list)}")


    # 模型初始化
    input_dim = 2 # 目前僅考慮 x, y
    latent_dim = 32
    uid_dim = max(valid_uid_list) + 1
    uid_embed_dim = 128
    hidden_dim = 128
    batch_size = 512
    max_len = 500
    num_layers = 1
    dataset = TrajectoryDataset(raw_train_df, valid_uid_list, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(input_dim, latent_dim, uid_dim, uid_embed_dim, hidden_dim, max_len, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 訓練迴圈 + EarlyStopping
    epochs = 50000
    patience = 250  # 多少 epoch 沒改善就停止
    best_loss = float('inf')
    wait = 0
    loss_list = []
    recon_list = []
    kl_list = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        for x, mask, lengths, uid, t in dataloader:
            x = x.to(device)
            mask = mask.to(device)
            t = t.to(device)
            uid = torch.tensor(uid, dtype=torch.long).to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(x, uid, t, mask)
            loss, recon_loss, kl_loss = cvae_loss(x_hat, x, mu, logvar, mask, beta=1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        loss_list.append(avg_loss)
        recon_list.append(avg_recon)
        kl_list.append(avg_kl)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")

        # EarlyStopping 機制
        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            torch.save(model.state_dict(), "./ckpt/CVAE/cvae_model_best.pth")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best loss: {best_loss:.4f}")
                break


    # 儲存模型
    os.makedirs('./ckpt/CVAE', exist_ok=True)
    torch.save(model.state_dict(), "./ckpt/CVAE/cvae_model.pth")
    print("模型已儲存至 ./ckpt/CVAE/cvae_model.pth")

    # 顯示 loss 趨勢圖
    plt.figure(figsize=(8, 6))
    plt.plot(loss_list, label='Total Loss')
    plt.plot(recon_list, label='Reconstruction Loss')
    plt.plot(kl_list, label='KL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CVAE Loss Trend')
    plt.legend()
    plt.grid()
    plt.show()

    # 載入最佳模型權重
    model.load_state_dict(torch.load("./ckpt/CVAE/cvae_model_best.pth", map_location=device))
    model.eval()

    # 測試生成其中一個使用者的未來軌跡
    target_uid = valid_uid_list[0]
    user_train_df = raw_train_df[raw_train_df['uid'] == target_uid]
    test_df = pd.read_csv(f'./Training_Testing_Data/A_x_test.csv')
    user_test_df = test_df[test_df['uid'] == target_uid]
    future_traj = generate_future_trajectory(model, 
                                             user_train_df, 
                                             user_test_df,
                                             target_uid, 
                                             device)
    print("生成第61~75天軌跡（x, y）：")
    for i, row in enumerate(future_traj):
        t = int(user_test_df.iloc[i]['t'])
        d = int(user_test_df.iloc[i]['d'])
        # 若有 working_day、delta_t 欄位也可一併取出
        print(f"Day{d}, t={t}, x={int(row[0])}, y={int(row[1])}, uid:{target_uid}")


    # 可視化比較
    true_traj = user_test_df[['x', 'y']].values  # 真實第61~75天
    gen_traj = future_traj[:, :2]                # 生成第61~75天

    plt.figure(figsize=(8, 6))
    plt.scatter(true_traj[:, 0], true_traj[:, 1], label='True', color='blue', marker='o', alpha=0.5, s=3)
    plt.scatter(gen_traj[:, 0], gen_traj[:, 1], label='Generated', color='red', marker='x', alpha=0.5, s=3)
    plt.title(f'UID {target_uid} 第61~75天軌跡比較')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()

    # 預測此cluster所有 <147000 的 uid
    results = []
    test_df = pd.read_csv(f'./Training_Testing_Data/A_x_test.csv')
    for idx, uid in enumerate(valid_uid_list):
        if uid > 147000:
            break
        user_train_df = raw_train_df[raw_train_df['uid'] == uid]
        user_test_df = test_df[test_df['uid'] == uid]
        if len(user_test_df) == 0:
            continue
        future_traj = generate_future_trajectory(model, user_train_df, user_test_df, uid, device)
        # future_traj shape: (gen_len, 2)
        for i, row in enumerate(future_traj):
            d = int(user_test_df.iloc[i]['d'])
            t = int(user_test_df.iloc[i]['t'])
            x = int(row[0])
            y = int(row[1])
            results.append([uid, d, t, x, y])
        print(f'預測進度: {idx+1}/{len(valid_uid_list)}', end='\r')
    
    # 轉成 DataFrame 並輸出
    pred_df = pd.DataFrame(results, columns=['uid', 'd', 't', 'x', 'y'])
    os.makedirs('./Predictions/CVAE', exist_ok=True)
    pred_df.to_csv('./Predictions/CVAE/A_x_cvae_pred_cluster1.csv', index=False)
    print("已輸出預測結果至 ./Predictions/CVAE/A_x_cvae_pred_cluster1.csv")


    # 計算 geobleu 分數
    def Evaluation(generated_data_input, reference_data_input, valid=False, city_name=None, raw_data_path=None):
        # 檢查生成的資料是否符合規範
        if valid:
            validator.main(city_name, raw_data_path, generated_data_input)

        # 讀取生成與參考資料
        if isinstance(generated_data_input, pd.DataFrame):
            generated_df = generated_data_input

        elif isinstance(generated_data_input, str):
            generated_df = pd.read_csv(generated_data_input, header=0, dtype=int)

        else:
            raise ValueError("只能接受DataFrame或資料路徑字串（csv檔）。") 
        
        if isinstance(reference_data_input, pd.DataFrame):
            reference_df = reference_data_input
 
        elif isinstance(reference_data_input, str):
            reference_df = pd.read_csv(reference_data_input, header=0, dtype=int)

        else:
            raise ValueError("只能接受DataFrame或資料路徑字串（csv檔）。") 
        
        # 檢查有哪些uid要check
        valid_uid_list = generated_df['uid'].unique()
        print(f'要檢查的UID數量: {len(valid_uid_list)}')

        # 計算每個 uid GEO-BLEU 和 dtw分數
        GEOBLEU_scores = []
        DTW_scores = []
        for idx, uid in enumerate(valid_uid_list):
            gen_user = generated_df[generated_df['uid'] == uid]
            ref_user = reference_df[reference_df['uid'] == uid]

            gen_traj = gen_user[['d', 't', 'x', 'y']].to_records(index=False)
            ref_traj = ref_user[['d', 't', 'x', 'y']].to_records(index=False)
            gen_traj = [tuple(row) for row in gen_traj]
            ref_traj = [tuple(row) for row in ref_traj]

            # GEOBLEU_score
            GEOBLEU_score = geobleu.calc_geobleu_single(gen_traj, ref_traj)
            GEOBLEU_scores.append(GEOBLEU_score)

            # dtw
            DTW_score = geobleu.calc_dtw_single(gen_traj, ref_traj)
            DTW_scores.append(DTW_score)

            print(f"{idx}/{len(valid_uid_list)}人--uid={uid}", end='\r')

        final_GEOBLEU_score = sum(GEOBLEU_scores) / len(GEOBLEU_scores) if GEOBLEU_scores else 0.0
        final_DTW_score = sum(DTW_scores) / len(DTW_scores) if DTW_scores else 0.0

        return final_GEOBLEU_score, final_DTW_score

    final_GEOBLEU_score, final_DTW_score = Evaluation(
    generated_data_input = f'./Predictions/CVAE/A_x_cvae_pred_cluster1.csv',
    reference_data_input = test_df,
    )
    print(f"最終GEO-BLEU分數: {final_GEOBLEU_score:.4f}, 最終DTW分數: {final_DTW_score:.4f}\n\n")