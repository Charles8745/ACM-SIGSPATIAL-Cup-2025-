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
import matplotlib
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 或 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正確顯示負號
"""
針對個人去訓練的 CVAE 模型
Ablation Study 1: Linear Weight (權重改成線性權重，即此點位出現幾次權重就是多少)
"""
class TrajectoryDataset(Dataset):
    def __init__(self, df, uid_list, xy2idx, max_len=400):
        self.uid_list = uid_list
        self.max_len = max_len
        self.xy2idx = xy2idx
        self.data = []
        self.lengths = []
        self.masks = []
        self.weights = []
        self.xy_idx_list = []
        for uid in uid_list:
            user_df = df[df['uid'] == uid]
            traj = user_df[['x', 'y', 't', 'working_day']].values
            length = len(traj)
            self.lengths.append(length)
            if length < max_len:
                pad = np.zeros((max_len - length, traj.shape[1]))
                traj = np.vstack([traj, pad])
                mask = np.concatenate([np.ones(length), np.zeros(max_len - length)])
            else:
                traj = traj[:max_len]
                mask = np.ones(max_len)
            self.data.append(torch.tensor(traj, dtype=torch.float32))
            self.masks.append(torch.tensor(mask, dtype=torch.float32))
            xy = traj[:, :2]
            # ===== MODIFIED: 改成線性權重（計數） =====
            xy_valid = traj[:length, :2]
            pts = [f"{p[0]}_{p[1]}" for p in xy_valid]
            unique, counts = np.unique(pts, return_counts=True)
            count_dict = dict(zip(unique, counts))
            # 線性權重：直接使用計數作為權重
            linear_weights_valid = np.array([count_dict[p] for p in pts], dtype=np.float32)
            weights_full = np.zeros(max_len, dtype=np.float32)
            weights_full[:length] = linear_weights_valid
            # ===== END MODIFIED =====
            self.weights.append(torch.tensor(weights_full, dtype=torch.float32))
            # xy_idx
            xy_idx = [self.xy2idx.get((int(p[0]), int(p[1])), 0) for p in xy]
            self.xy_idx_list.append(torch.tensor(xy_idx, dtype=torch.long))
        self.data = torch.stack(self.data)
        self.masks = torch.stack(self.masks)
        self.weights = torch.stack(self.weights)
        self.xy_idx_list = torch.stack(self.xy_idx_list)

    def __len__(self):
        return len(self.uid_list)

    def __getitem__(self, idx):
        traj = self.data[idx]
        mask = self.masks[idx]
        length = self.lengths[idx]
        uid = self.uid_list[idx]
        t_seq = traj[:, 2].long()
        working_day_seq = traj[:, 3].long()
        weights = self.weights[idx]
        xy_idx = self.xy_idx_list[idx]
        return traj, mask, length, uid, t_seq, working_day_seq, weights, xy_idx

class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, uid_dim, uid_embed_dim, hidden_dim, max_len, N_valid, num_layers=1, dropout=0.3):
        super().__init__()
        self.uid_embedding = nn.Embedding(uid_dim, uid_embed_dim)
        self.t_embedding = nn.Embedding(49, 24)
        self.working_day_embedding = nn.Embedding(2, 2)
        self.encoder_rnn = nn.LSTM(input_dim + uid_embed_dim +24+2, hidden_dim, batch_first=True, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim * 2)
        self.decoder_rnn = nn.LSTM(latent_dim + uid_embed_dim +24+2, hidden_dim, batch_first=True, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.decoder_fc_xy = nn.Linear(hidden_dim, N_valid)  # 只預測有效點位
        self.max_len = max_len

    def encode(self, x, uid, t, working_day, mask):
        uid_embed = self.uid_embedding(uid)
        t_embed = self.t_embedding(t)
        wd_embed = self.working_day_embedding(working_day)
        uid_embed_expand = uid_embed.unsqueeze(1).expand(-1, x.size(1), -1)
        x_wo_t = x[..., :2]
        x_cat = torch.cat([x_wo_t, uid_embed_expand, t_embed, wd_embed], dim=-1)
        lengths = mask.sum(dim=1).long().cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(x_cat, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.encoder_rnn(packed)
        h = h_n[-1]
        h = self.dropout(h)
        h = self.encoder_fc(h)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, uid, t, working_day):
        uid_embed = self.uid_embedding(uid)
        t_embed = self.t_embedding(t)
        wd_embed = self.working_day_embedding(working_day)
        uid_embed_expand = uid_embed.unsqueeze(1).expand(-1, t.size(1), -1)
        z_expand = z.unsqueeze(1).expand(-1, t.size(1), -1)
        z_cat_seq = torch.cat([z_expand, uid_embed_expand, t_embed, wd_embed], dim=-1)
        out, _ = self.decoder_rnn(z_cat_seq)
        out = self.dropout(out)
        xy_logits = self.decoder_fc_xy(out)  # (batch, seq_len, N_valid)
        return xy_logits

    def forward(self, x, uid, t, working_day, mask):
        mu, logvar = self.encode(x, uid, t, working_day, mask)
        z = self.reparameterize(mu, logvar)
        xy_logits = self.decode(z, uid, t, working_day)
        return xy_logits, mu, logvar

def cvae_loss(xy_logits, xy_idx, mu, logvar, mask, weights, beta=0.05, lambda_entropy=0.01):
    # 從 logits 自行取得 N_valid，避免使用未定義的全域變數
    N_valid = xy_logits.size(-1)
    xy_logits = xy_logits.reshape(-1, N_valid)
    xy_idx = xy_idx.view(-1)
    mask = mask.view(-1)
    weights = weights.view(-1)

    valid = mask > 0
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    ce_loss = loss_fn(xy_logits[valid], xy_idx[valid])
    ce_loss = ce_loss * weights[valid]
    weighted_loss = ce_loss.sum() / (weights[valid].sum() + 1e-8)

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

    probs = F.softmax(xy_logits[valid], dim=-1)
    entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1).mean()
    loss = weighted_loss + beta * kl_loss - lambda_entropy * entropy
    return loss, weighted_loss, kl_loss

def sample_from_logits(logits, temperature=0.7, top_k=None, top_p=None):
    # 決定性取樣：直接取 argmax（不使用隨機）
    if temperature is not None and temperature > 0:
        logits = logits / temperature
    return torch.argmax(logits, dim=-1)

def generate_future_trajectory(model, user_train_df, user_test_df, uid_idx, device, valid_xy_list, temperature=1.0, top_k=None, top_p=None):
    model.eval()
    with torch.no_grad():
        traj = user_train_df[['x', 'y', 't', 'working_day']].values
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
        uid_tensor = torch.tensor([uid_idx], dtype=torch.long, device=device)
        t_seq_train = torch.tensor(traj[:, 2], dtype=torch.long).unsqueeze(0).to(device)
        wd_seq_train = torch.tensor(traj[:, 3], dtype=torch.long).unsqueeze(0).to(device)

        mu, logvar = model.encode(x, uid_tensor, t_seq_train, wd_seq_train, mask)
        z = model.reparameterize(mu, logvar)

        gen_len = user_test_df.shape[0]
        t_seq_gen = torch.tensor(user_test_df['t'].values, dtype=torch.long).unsqueeze(0).to(device)
        wd_seq_gen = torch.tensor(user_test_df['working_day'].values, dtype=torch.long).unsqueeze(0).to(device)
        xy_logits = model.decode(z, uid_tensor, t_seq_gen, wd_seq_gen)
        samples = sample_from_logits(xy_logits[:, :gen_len, :], temperature, top_k, top_p).squeeze(0).cpu().numpy()
        future_traj = np.array([valid_xy_list[i] for i in samples])
        return future_traj


if __name__ == "__main__": 
    # 加速選項
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')  # 允許 TF32 / 加速 matmul
    except Exception:
        pass

    # 資料準備 完整3000人1-60天訓練
    city = 'B'
    raw_x_train_df = pd.read_csv(f'./Training_Testing_Data/{city}_x_train.csv')
    raw_train_df = raw_x_train_df.copy()

    valid_uid_list = raw_train_df["uid"].unique()
    print(f'有效的使用者ID數量: {len(valid_uid_list)}')

    # for uid_idx, valid_uid in enumerate(valid_uid_list): # 逐 uid 訓練
    #     # 統計有效點位
    #     xy_set = set()
    #     user_df = raw_train_df[raw_train_df['uid'] == valid_uid]
    #     xy_arr = user_df[['x', 'y']].values
    #     for x, y in xy_arr:
    #         xy_set.add((int(x), int(y)))
    #     valid_xy_list = sorted(list(xy_set))
    #     xy2idx = {xy: i for i, xy in enumerate(valid_xy_list)}
    #     N_valid = len(valid_xy_list)
    #     print(f"uid:{valid_uid} 有效點位數量: {N_valid}")

    #     # 模型初始化
    #     input_dim = 2 # 目前僅考慮 x, y
    #     latent_dim = N_valid # 潛在空間維度
    #     uid_dim = 1
    #     uid_embed_dim = 1
    #     hidden_dim = N_valid
    #     batch_size = 1
    #     max_len = 550
    #     num_layers = 1

    #     dataset = TrajectoryDataset(raw_train_df, [valid_uid], xy2idx, max_len=max_len)
    #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     model = CVAE(input_dim, latent_dim, uid_dim, uid_embed_dim, hidden_dim, max_len, N_valid, num_layers, dropout=0.3).to(device)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #     # 訓練迴圈 + EarlyStopping（每 uid 分開）
    #     epochs = 20000
    #     patience = 100
    #     best_loss = float('inf')
    #     wait = 0
    #     loss_list = []
    #     recon_list = []
    #     kl_list = []
    #     for epoch in range(epochs):
    #         beta = min(1.0, epoch / 10000)
    #         model.train()
    #         total_loss = 0
    #         total_recon = 0
    #         total_kl = 0
    #         for x, mask, lengths, uid, t, working_day, weights, xy_idx in dataloader:
    #             x = x.to(device)
    #             mask = mask.to(device)
    #             t = t.to(device)
    #             working_day = working_day.long().to(device)
    #             # 單 uid 設為索引 0，避免嵌入越界
    #             uid = torch.zeros(x.size(0), dtype=torch.long, device=device)
    #             weights = weights.to(device)
    #             xy_idx = xy_idx.to(device)

    #             optimizer.zero_grad()
    #             xy_logits, mu, logvar = model(x, uid, t, working_day, mask)
    #             loss, recon_loss, kl_loss = cvae_loss(xy_logits, xy_idx, mu, logvar, mask, weights, beta=beta)
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #             optimizer.step()

    #             total_loss += loss.item()
    #             total_recon += recon_loss.item()
    #             total_kl += kl_loss.item()

    #         avg_loss = total_loss / len(dataloader)
    #         avg_recon = total_recon / len(dataloader)
    #         avg_kl = total_kl / len(dataloader)
    #         loss_list.append(avg_loss)
    #         recon_list.append(avg_recon)
    #         kl_list.append(avg_kl)

    #         if (epoch + 1) % 1000 == 0:
    #             print(f"{uid_idx+1}/{len(valid_uid_list)} uid:{valid_uid} Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Recon: {avg_recon:.6f}, KL: {avg_kl:.6f}")

    #         if avg_loss < best_loss:
    #             best_loss = avg_loss
    #             wait = 0
    #             os.makedirs(f"./ckpt/CVAE/uid_level_class/", exist_ok=True)
    #             torch.save(model.state_dict(), f"./ckpt/CVAE/uid_level_class/cvae_model_uid{valid_uid}_l{latent_dim}h{hidden_dim}_city{city}.pth")
    #         else:
    #             wait += 1
    #             if wait >= patience:
    #                 print(f"uid:{valid_uid} Early stopping at epoch {epoch+1}. Best loss: {best_loss:.4f}")
    #                 break


    # ===== 針對每個 uid 載入各自模型做推論 =====
    results = []
    test_df = pd.read_csv(f'./Training_Testing_Data/{city}_x_test.csv')
    valid_uid_list = test_df["uid"].unique()
    print(f'有效的測試使用者ID數量: {len(valid_uid_list)}')

    for idx, uid in enumerate(valid_uid_list):
        user_train_df = raw_train_df[raw_train_df['uid'] == uid]
        user_test_df = test_df[test_df['uid'] == uid]

        # 重建該 uid 的有效點位與模型尺寸
        xy_set = set((int(x), int(y)) for x, y in user_train_df[['x', 'y']].values)
        valid_xy_list = sorted(list(xy_set))
        N_valid = len(valid_xy_list)
        latent_dim = N_valid
        hidden_dim = N_valid 

        ckpt_path = f"./ckpt/CVAE/uid_level_class/cvae_model_uid{uid}_l{latent_dim}h{hidden_dim}_city{city}.pth"
        if not os.path.exists(ckpt_path):
            print(f"[warn] 找不到 uid {uid} 的權重，略過")
            continue

        # 建立同尺寸模型並載入該 uid 權重
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CVAE(input_dim=2, latent_dim=latent_dim, uid_dim=1, uid_embed_dim=1,
                     hidden_dim=hidden_dim, max_len=550, N_valid=N_valid, num_layers=1, dropout=0.3).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

        future_traj = generate_future_trajectory(
            model=model,
            user_train_df=user_train_df,
            user_test_df=user_test_df,
            uid_idx=0,
            device=device,
            valid_xy_list=valid_xy_list,
        )
        for i, row in enumerate(future_traj):
            d = int(user_test_df.iloc[i]['d'])
            t = int(user_test_df.iloc[i]['t'])
            x = int(row[0])
            y = int(row[1])
            results.append([uid, d, t, x, y])

        print(f'預測進度: {idx+1}/{len(valid_uid_list)}', end='\r')

    pred_df = pd.DataFrame(results, columns=['uid', 'd', 't', 'x', 'y'])
    os.makedirs('./Predictions/CVAE/', exist_ok=True)
    pred_df.to_csv(f'./Predictions/CVAE/{city}_x_cvae_pred.csv', index=False)
    print(f"已輸出預測結果至 ./Predictions/CVAE/{city}_x_cvae_pred.csv")

    # ===== scatter cvae 61-75天 vs. gt 61-75 =====
    mode_pred_df = pd.read_csv(f'./Predictions/{city}_x_Per_User_Per_t_Mode_working_day.csv')
    cvae_pred_df = pd.read_csv(f'./Predictions/CVAE/{city}_x_cvae_pred.csv')
    gt_df = pd.read_csv(f'./Training_Testing_Data/{city}_x_test.csv')
    valid_uid_list = cvae_pred_df['uid'].unique().tolist()
    np.random.shuffle(valid_uid_list)

    fig, axes = plt.subplots(2, len(valid_uid_list[:5]), figsize=(20,12))
    for i, uid in enumerate(valid_uid_list[:5]):
        axes[0, i].scatter(cvae_pred_df[cvae_pred_df['uid'] == uid]['x'],
                        cvae_pred_df[cvae_pred_df['uid'] == uid]['y'],
                        label='CVAE 61-75', alpha=0.8, s=10, color='red', marker='x')
        axes[0, i].scatter(gt_df[gt_df['uid'] == uid]['x'],
                gt_df[gt_df['uid'] == uid]['y'],
                label='GT 1-60', alpha=0.1, s=3, color='green')
        axes[0, i].set_title(f'UID {uid} Mode')
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('y')
        axes[0, i].set_aspect('equal')
        axes[0, i].set_xlim(1, 200)
        axes[0, i].set_ylim(1, 200)
        axes[0, i].grid(True)
        axes[0, i].invert_yaxis()
        axes[0, i].legend()

        axes[1, i].scatter(gt_df[gt_df['uid'] == uid]['x'],
                gt_df[gt_df['uid'] == uid]['y'],
                label='gt', alpha=0.5, s=10, color='green')
        axes[1, i].set_title(f'UID {uid} GT')
        axes[1, i].set_xlabel('x')
        axes[1, i].set_ylabel('y')
        axes[1, i].set_aspect('equal')
        axes[1, i].set_xlim(1, 200)
        axes[1, i].set_ylim(1, 200)
        axes[1, i].grid(True)
        axes[1, i].invert_yaxis()
        axes[1, i].legend()

    plt.tight_layout()
    plt.show()

    # ===== 把mode, cvae的x,y拉出來看未來61-75天時間線段上的重合性 =====
    def plot_x_y_sequence_compare(uid, mode_df, cvae_df):
        # 依照時間排序
        mode_user = mode_df[mode_df['uid'] == uid].sort_values(['d', 't'])
        cvae_user = cvae_df[cvae_df['uid'] == uid].sort_values(['d', 't'])

        fig, axes = plt.subplots(2, figsize=(18, 12), sharex=True)

        # 左：x的mode和cvae
        axes[0].plot(mode_user['x'].values, '-o', label='Mode', color='red', alpha=0.7)
        axes[0].plot(cvae_user['x'].values, '-o', label='CVAE', color='green', alpha=0.7)
        axes[0].set_title(f'UID {uid} x 時序 (Mode vs CVAE)')
        axes[0].set_ylabel('x')
        axes[0].legend()
        axes[0].grid(True)

        # 右：y的mode和cvae
        axes[1].plot(mode_user['y'].values, '-o', label='Mode', color='red', alpha=0.7)
        axes[1].plot(cvae_user['y'].values, '-o', label='CVAE', color='green', alpha=0.7)
        axes[1].set_title(f'UID {uid} y 時序 (Mode vs CVAE)')
        axes[1].set_ylabel('y')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    for i in range(10):
        plot_x_y_sequence_compare(uid=valid_uid_list[i],
                                    mode_df=mode_pred_df,
                                    cvae_df=cvae_pred_df,
                                    )
        
