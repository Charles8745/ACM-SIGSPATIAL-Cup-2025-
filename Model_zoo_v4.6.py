# ...existing code...
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
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 或 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正確顯示負號
"""
採用 Diffusion (DDPM) 取代 CVAE。
條件：uid, t(時間槽), working_day, day_of_week
"""

# -------------------------
# Diffusion 模型與工具
# -------------------------
class GaussianDiffusion:
    def __init__(self, timesteps=1000, device='cpu', beta_schedule='linear'):
        self.timesteps = timesteps
        self.device = torch.device(device)
        if beta_schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float32)
        else:
            betas = torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register(betas=betas, alphas=alphas, alphas_cumprod=alphas_cumprod)
        self.register(
            sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
            sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod),
            sqrt_recip_alphas=torch.sqrt(1.0 / alphas),
            posterior_variance=betas
        )

    def register(self, **tensors):
        for k, v in tensors.items():
            setattr(self, k, v.to(self.device))

    def q_sample(self, x0, t, noise):
        # t: (B,) long -> broadcast per position
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_ac * x0 + sqrt_om * noise

    def p_mean_variance(self, x, t, eps_pred):
        # DDPM posterior mean
        beta_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_ac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_recip_alpha = self.sqrt_recip_alphas[t].view(-1, 1, 1)
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)

        model_mean = sqrt_recip_alpha * (x - beta_t * eps_pred / (sqrt_one_minus_ac + 1e-8))
        model_var = beta_t
        x0_pred = (x - sqrt_one_minus_ac * eps_pred) / (torch.sqrt(alphas_cumprod_t) + 1e-8)
        return model_mean, model_var, x0_pred

class CondDDPM(nn.Module):
    def __init__(self, uid_dim, uid_embed_dim, hidden_dim, timesteps, num_layers=1):
        super().__init__()
        # 條件嵌入
        self.uid_embedding = nn.Embedding(uid_dim, uid_embed_dim)
        self.t_embedding = nn.Embedding(49, 24)
        self.working_day_embedding = nn.Embedding(2, 2)
        self.day_of_week_embedding = nn.Embedding(7, 7)
        self.diff_step_embedding = nn.Embedding(timesteps, 64)

        in_dim = 2 + uid_embed_dim + 24 + 2 + 7 + 64
        self.rnn = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x_noisy, uid, t_slot, working_day, day_of_week, diff_step):
        # x_noisy: (B, L, 2)
        B, L, _ = x_noisy.shape
        uid_embed = self.uid_embedding(uid).unsqueeze(1).expand(-1, L, -1)
        t_embed = self.t_embedding(t_slot)
        wd_embed = self.working_day_embedding(working_day)
        dow_embed = self.day_of_week_embedding(day_of_week)
        ds_embed = self.diff_step_embedding(diff_step).unsqueeze(1).expand(-1, L, -1)

        x_cat = torch.cat([x_noisy, uid_embed, t_embed, wd_embed, dow_embed, ds_embed], dim=-1)
        out, _ = self.rnn(x_cat)  # (B, L, 2*hidden)
        eps_hat = self.head(out)  # (B, L, 2)
        return eps_hat

def ddpm_loss(model, diffusion, x0, uid, t_slot, working_day, day_of_week, mask, weights):
    # 取每個 batch 一個 diffusion step
    B = x0.size(0)
    diff_t = torch.randint(0, diffusion.timesteps, (B,), device=x0.device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_noisy = diffusion.q_sample(x0, diff_t, noise)
    eps_hat = model(x_noisy, uid, t_slot, working_day, day_of_week, diff_t)

    mse = ((eps_hat - noise) ** 2).sum(dim=-1)  # (B, L)
    loss = (mse * weights * mask).sum() / (weights * mask).sum().clamp_min(1.0)
    return loss

@torch.no_grad()
def generate_future_trajectory_ddpm(model, diffusion, user_test_df, uid, device):
    model.eval()
    gen_len = user_test_df.shape[0]
    uid_tensor = torch.tensor([uid], dtype=torch.long, device=device)
    t_seq = torch.tensor(user_test_df['t'].values, dtype=torch.long, device=device).unsqueeze(0)
    wd_seq = torch.tensor(user_test_df['working_day'].values, dtype=torch.long, device=device).unsqueeze(0)
    dow_seq = torch.tensor(user_test_df['day_of_week'].values, dtype=torch.long, device=device).unsqueeze(0)

    x = torch.randn((1, gen_len, 2), device=device)
    for step in reversed(range(diffusion.timesteps)):
        t_step = torch.full((1,), step, dtype=torch.long, device=device)
        eps_hat = model(x, uid_tensor, t_seq, wd_seq, dow_seq, t_step)
        mean, var, _ = diffusion.p_mean_variance(x, t_step, eps_hat)
        if step > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(var) * noise
        else:
            x = mean
    return x.squeeze(0).clamp(min=1.0, max=100.0).cpu().numpy()
# ...existing code...

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

    # 超參數
    uid_dim = max(valid_uid_list) + 1
    uid_embed_dim = 256
    hidden_dim = 256
    batch_size = 512
    max_len = 550
    num_layers = 1
    num_diff_steps = 200  # 可調整到 1000 但訓練/取樣較慢
    lr = 1e-3

    dataset = TrajectoryDataset(raw_train_df, valid_uid_list, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CondDDPM(uid_dim=uid_dim, uid_embed_dim=uid_embed_dim, hidden_dim=hidden_dim,
                     timesteps=num_diff_steps, num_layers=num_layers).to(device)
    diffusion = GaussianDiffusion(timesteps=num_diff_steps, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 訓練 + EarlyStopping（以訓練 loss）
    epochs = 2000
    patience = 200
    best_loss = float('inf')
    wait = 0
    loss_list = []

    os.makedirs('./ckpt/DDPM', exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, mask, lengths, uid, t_slot, working_day, day_of_week, weights in dataloader:
            x = x[..., :2].to(device)                # 只取 (x,y)
            mask = mask.to(device)
            t_slot = t_slot.to(device)
            working_day = working_day.long().to(device)
            day_of_week = day_of_week.to(device)
            uid = torch.tensor(uid, dtype=torch.long, device=device)
            weights = weights.to(device)

            optimizer.zero_grad()
            loss = ddpm_loss(model, diffusion, x, uid, t_slot, working_day, day_of_week, mask, weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_list.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            torch.save(model.state_dict(), "./ckpt/DDPM/ddpm_model_best.pth")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best loss: {best_loss:.6f}")
                break

    # 儲存最後模型
    torch.save(model.state_dict(), "./ckpt/DDPM/ddpm_model_last.pth")
    print("模型已儲存至 ./ckpt/DDPM/")

    # 顯示 loss 趨勢圖
    plt.figure(figsize=(8, 6))
    plt.plot(loss_list, label='DDPM Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DDPM Loss Trend')
    plt.legend()
    plt.grid()
    plt.show()

    # 載入最佳模型權重
    model.load_state_dict(torch.load("./ckpt/DDPM/ddpm_model_best.pth", map_location=device))
    model.eval()

    # 產生預測 (對 cluster==1 且 uid<147000)
    results = []
    test_df = pd.read_csv(f'./Training_Testing_Data/A_x_test.csv')
    os.makedirs('./Predictions/DDPM', exist_ok=True)

    for idx, uid in enumerate(valid_uid_list):
        if uid > 147000:
            break
        user_test_df = test_df[test_df['uid'] == uid]
        if len(user_test_df) == 0:
            continue
        future_traj = generate_future_trajectory_ddpm(model, diffusion, user_test_df, uid, device)
        for i, row in enumerate(future_traj):
            d = int(user_test_df.iloc[i]['d'])
            t_val = int(user_test_df.iloc[i]['t'])
            x_pred = int(np.clip(row[0], 1, 100))
            y_pred = int(np.clip(row[1], 1, 100))
            results.append([uid, d, t_val, x_pred, y_pred])
        print(f'預測進度: {idx+1}/{len(valid_uid_list)}', end='\r')

    pred_df = pd.DataFrame(results, columns=['uid', 'd', 't', 'x', 'y'])
    pred_path = './Predictions/DDPM/A_x_ddpm_pred_cluster1.csv'
    pred_df.to_csv(pred_path, index=False)
    print(f"\n已輸出預測結果至 {pred_path}")

    # 評估 GEO-BLEU / DTW
    def Evaluation(generated_data_input, reference_data_input, valid=False, city_name=None, raw_data_path=None):
        if valid:
            validator.main(city_name, raw_data_path, generated_data_input)
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
        
        valid_uid_list_eval = generated_df['uid'].unique()
        print(f'要檢查的UID數量: {len(valid_uid_list_eval)}')

        GEOBLEU_scores, DTW_scores = [], []
        for idx, uid_val in enumerate(valid_uid_list_eval):
            gen_user = generated_df[generated_df['uid'] == uid_val]
            ref_user = reference_df[reference_df['uid'] == uid_val]

            gen_traj = [tuple(row) for row in gen_user[['d', 't', 'x', 'y']].to_records(index=False)]
            ref_traj = [tuple(row) for row in ref_user[['d', 't', 'x', 'y']].to_records(index=False)]

            GEOBLEU_scores.append(geobleu.calc_geobleu_single(gen_traj, ref_traj))
            DTW_scores.append(geobleu.calc_dtw_single(gen_traj, ref_traj))
            print(f"{idx}/{len(valid_uid_list_eval)}人--uid={uid_val}", end='\r')

        final_GEOBLEU_score = sum(GEOBLEU_scores) / len(GEOBLEU_scores) if GEOBLEU_scores else 0.0
        final_DTW_score = sum(DTW_scores) / len(DTW_scores) if DTW_scores else 0.0
        return final_GEOBLEU_score, final_DTW_score

    final_GEOBLEU_score, final_DTW_score = Evaluation(
        generated_data_input = pred_path,
        reference_data_input = test_df,
    )
    print(f"最終GEO-BLEU分數: {final_GEOBLEU_score:.4f}, 最終DTW分數: {final_DTW_score:.4f}\n\n")

    # 以下視覺化與比較維持，但改讀取 DDPM 結果
    mode_pred_df = pd.read_csv('./Predictions/A_x_cluster1_modify_Per_User_Per_t_Mode_working_day_modify.csv')
    ddpm_pred_df = pd.read_csv(pred_path)
    gt_df = pd.read_csv('./Training_Testing_Data/A_x_test.csv')
    valid_uid_list = mode_pred_df['uid'].unique().tolist()
    valid_uid_list = valid_uid_list[:5]

    fig, axes = plt.subplots(3, len(valid_uid_list), figsize=(20,12))
    for i, uid in enumerate(valid_uid_list):
        axes[0, i].scatter(mode_pred_df[mode_pred_df['uid'] == uid]['x'],
                           mode_pred_df[mode_pred_df['uid'] == uid]['y'],
                           label='Mode', alpha=0.8, s=10, color='red', marker='x')
        axes[0, i].scatter(gt_df[gt_df['uid'] == uid]['x'],
                           gt_df[gt_df['uid'] == uid]['y'],
                           label='gt', alpha=0.1, s=3, color='green')
        axes[0, i].set_title(f'UID {uid} Mode')
        axes[0, i].set_xlabel('x'); axes[0, i].set_ylabel('y')
        axes[0, i].set_aspect('equal'); axes[0, i].set_xlim(1, 100); axes[0, i].set_ylim(1, 100)
        axes[0, i].grid(True); axes[0, i].invert_yaxis(); axes[0, i].legend()

        axes[1, i].scatter(ddpm_pred_df[ddpm_pred_df['uid'] == uid]['x'],
                           ddpm_pred_df[ddpm_pred_df['uid'] == uid]['y'],
                           label='DDPM', alpha=0.8, s=10, color='blue', marker='x')
        axes[1, i].scatter(gt_df[gt_df['uid'] == uid]['x'],
                           gt_df[gt_df['uid'] == uid]['y'],
                           label='gt', alpha=0.1, s=3, color='green')
        axes[1, i].set_title(f'UID {uid} DDPM')
        axes[1, i].set_xlabel('x'); axes[1, i].set_ylabel('y')
        axes[1, i].set_aspect('equal'); axes[1, i].set_xlim(1, 100); axes[1, i].set_ylim(1, 100)
        axes[1, i].grid(True); axes[1, i].invert_yaxis(); axes[1, i].legend()

        axes[2, i].scatter(gt_df[gt_df['uid'] == uid]['x'],
                           gt_df[gt_df['uid'] == uid]['y'],
                           label='gt', alpha=0.5, s=10, color='green')
        axes[2, i].set_title(f'UID {uid} GT')
        axes[2, i].set_xlabel('x'); axes[2, i].set_ylabel('y')
        axes[2, i].set_aspect('equal'); axes[2, i].set_xlim(1, 100); axes[2, i].set_ylim(1, 100)
        axes[2, i].grid(True); axes[2, i].invert_yaxis(); axes[2, i].legend()

    plt.tight_layout()
    plt.show()

    def plot_top10_ratio_compare_single_uid_by_gt(mode_df, gt_df, ddpm_df, uid):
        def get_xy_counts(df):
            df_uid = df[df['uid'] == uid]
            return df_uid.groupby(['x', 'y']).size()

        gt_counts = get_xy_counts(gt_df)
        mode_counts = get_xy_counts(mode_df)
        ddpm_counts = get_xy_counts(ddpm_df)

        gt_top15 = gt_counts.nlargest(15)
        gt_total = gt_counts.sum()
        mode_total = mode_counts.sum()
        ddpm_total = ddpm_counts.sum()

        labels = [f'{idx[0]},{idx[1]}' for idx in gt_top15.index]
        gt_ratios = (gt_top15 / gt_total).values
        mode_ratios = [(mode_counts.get(idx, 0) / mode_total) if mode_total > 0 else 0 for idx in gt_top15.index]
        ddpm_ratios = [(ddpm_counts.get(idx, 0) / ddpm_total) if ddpm_total > 0 else 0 for idx in gt_top15.index]

        x = np.arange(len(labels))
        width = 0.25

        plt.figure(figsize=(24, 12))
        plt.bar(x - width, gt_ratios, width, label='GT')
        plt.bar(x, mode_ratios, width, label='Mode')
        plt.bar(x + width, ddpm_ratios, width, label='DDPM')
        plt.ylabel('出現佔比'); plt.xlabel('(x, y)')
        plt.title(f'UID {uid} (依據GT前15大) (x,y) 出現佔比比較')
        plt.xticks(x, labels, rotation=45)
        plt.legend(); plt.tight_layout(); plt.show()

    plot_top10_ratio_compare_single_uid_by_gt(mode_pred_df, gt_df, ddpm_pred_df, uid=valid_uid_list[0])

    def plot_x_y_sequence_compare(uid, mode_df, ddpm_df, gt_df):
        mode_user = mode_df[mode_df['uid'] == uid].sort_values(['d', 't'])
        ddpm_user = ddpm_df[ddpm_df['uid'] == uid].sort_values(['d', 't'])
        gt_user = gt_df[gt_df['uid'] == uid].sort_values(['d', 't'])

        fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)

        axes[0, 0].plot(mode_user['x'].values, '-o', label='Mode', color='red', alpha=0.7)
        axes[0, 0].plot(gt_user['x'].values, '-o', label='GT', color='green', alpha=0.7)
        axes[0, 0].set_title(f'UID {uid} x 時序 (Mode vs GT)'); axes[0, 0].set_ylabel('x'); axes[0, 0].legend(); axes[0, 0].grid(True)

        axes[1, 0].plot(ddpm_user['x'].values, '-o', label='DDPM', color='blue', alpha=0.7)
        axes[1, 0].plot(gt_user['x'].values, '-o', label='GT', color='green', alpha=0.7)
        axes[1, 0].set_title(f'UID {uid} x 時序 (DDPM vs GT)'); axes[1, 0].set_xlabel('時間點'); axes[1, 0].set_ylabel('x'); axes[1, 0].legend(); axes[1, 0].grid(True)

        axes[0, 1].plot(mode_user['y'].values, '-o', label='Mode', color='red', alpha=0.7)
        axes[0, 1].plot(gt_user['y'].values, '-o', label='GT', color='green', alpha=0.7)
        axes[0, 1].set_title(f'UID {uid} y 時序 (Mode vs GT)'); axes[0, 1].set_ylabel('y'); axes[0, 1].legend(); axes[0, 1].grid(True)

        axes[1, 1].plot(ddpm_user['y'].values, '-o', label='DDPM', color='blue', alpha=0.7)
        axes[1, 1].plot(gt_user['y'].values, '-o', label='GT', color='green', alpha=0.7)
        axes[1, 1].set_title(f'UID {uid} y 時序 (DDPM vs GT)'); axes[1, 1].set_xlabel('時間點'); axes[1, 1].set_ylabel('y'); axes[1, 1].legend(); axes[1, 1].grid(True)

        plt.tight_layout(); plt.show()

    plot_x_y_sequence_compare(uid=valid_uid_list[1],
                              mode_df=mode_pred_df,
                              ddpm_df=ddpm_pred_df,
                              gt_df=gt_df)
# ...existing code...