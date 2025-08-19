# ...existing code...
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time 
import geobleu
import copy
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
class TrajectoryDataset(Dataset):
    def __init__(self, df, uid_list, max_len=400):
        self.uid_list = uid_list
        self.max_len = max_len
        self.data = []
        self.lengths = []
        self.masks = []
        self.weights = []
        self.day_of_week_seqs = []  # 改名
        for uid in uid_list:
            user_df = df[df['uid'] == uid]
            traj = user_df[['x', 'y', 't', 'working_day', 'day_of_week']].values  # 換成 day_of_week
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
            pts = [f"{p[0]}_{p[1]}" for p in xy]
            unique, counts = np.unique(pts, return_counts=True)
            count_dict = dict(zip(unique, counts))
            log_base = 10
            log_weights = (np.log([count_dict[p] + 1 for p in pts]) / np.log(log_base))
            self.weights.append(torch.tensor(log_weights, dtype=torch.float32))
            self.day_of_week_seqs.append(torch.tensor(traj[:, 4], dtype=torch.long))  # day_of_week
        self.data = torch.stack(self.data)
        self.masks = torch.stack(self.masks)
        self.weights = torch.stack(self.weights)
        self.day_of_week_seqs = torch.stack(self.day_of_week_seqs)

    def __len__(self):
        return len(self.uid_list)

    def __getitem__(self, idx):
        traj = self.data[idx]
        mask = self.masks[idx]
        length = self.lengths[idx]
        uid = self.uid_list[idx]
        t_seq = traj[:, 2].long()
        working_day_seq = traj[:, 3].long()
        day_of_week_seq = self.day_of_week_seqs[idx]  # 改名
        weights = self.weights[idx]
        return traj, mask, length, uid, t_seq, working_day_seq, day_of_week_seq, weights
# -------------------------
# Diffusion 模型與工具
# -------------------------
class GaussianDiffusion:
    def __init__(self, timesteps=1000, device='cpu', beta_schedule='linear'):
        self.timesteps = timesteps
        self.device = torch.device(device)

        if beta_schedule == 'cosine':
            # Nichol & Dhariwal cosine schedule
            s = 0.008
            steps = timesteps
            t = torch.linspace(0, steps, steps + 1, dtype=torch.float32) / steps
            alphas_bar = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            betas = betas.clamp(1e-6, 0.999)
        else:
            betas = torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float32), alphas_cumprod[:-1]], dim=0)

        self.register(betas=betas, alphas=alphas, alphas_cumprod=alphas_cumprod, alphas_cumprod_prev=alphas_cumprod_prev)
        self.register(
            sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
            sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod),
            sqrt_recip_alphas=torch.sqrt(1.0 / alphas),
        )

        # 正確的 posterior 係數
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

        self.register(
            posterior_variance=posterior_variance,
            posterior_log_variance_clipped=posterior_log_variance_clipped,
            posterior_mean_coef1=posterior_mean_coef1,
            posterior_mean_coef2=posterior_mean_coef2
        )

    def register(self, **tensors):
        for k, v in tensors.items():
            setattr(self, k, v.to(self.device))

    def q_sample(self, x0, t, noise):
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_ac * x0 + sqrt_om * noise

    def predict_x0(self, x_t, t, eps_pred):
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return (x_t - sqrt_om * eps_pred) / (sqrt_ac + 1e-8)

    def p_mean_variance(self, x_t, t, eps_pred):
        x0_pred = self.predict_x0(x_t, t, eps_pred)
        coef1 = self.posterior_mean_coef1[t].view(-1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].view(-1, 1, 1)
        model_mean = coef1 * x0_pred + coef2 * x_t
        model_var = self.posterior_variance[t].view(-1, 1, 1)
        return model_mean, model_var, x0_pred

GRID_MIN = 1.0
GRID_MAX = 200.0
NORM_CENTER = (GRID_MIN + GRID_MAX) / 2.0
NORM_SCALE = (GRID_MAX - GRID_MIN) / 2.0

def set_grid_bounds(grid_max: float):
    global GRID_MAX, NORM_CENTER, NORM_SCALE
    GRID_MAX = float(grid_max)
    NORM_CENTER = (GRID_MIN + GRID_MAX) / 2.0
    NORM_SCALE = (GRID_MAX - GRID_MIN) / 2.0

def normalize_xy(xy):
    return (xy - NORM_CENTER) / NORM_SCALE

def denormalize_xy(xy_norm):
    return xy_norm * NORM_SCALE + NORM_CENTER

class CondDDPM(nn.Module):
    def __init__(self, uid_dim, uid_embed_dim, hidden_dim, timesteps, num_layers=1):
        super().__init__()
        self.uid_embedding = nn.Embedding(uid_dim, uid_embed_dim)
        self.t_embedding = nn.Embedding(49, 24)
        self.working_day_embedding = nn.Embedding(2, 2)
        self.day_of_week_embedding = nn.Embedding(7, 7)
        # 移除 time_of_week_embedding 以提速
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

    def forward(self, x_noisy, uid, t_slot, working_day, day_of_week, diff_step, cond_mask=None):
        B, L, _ = x_noisy.shape
        if cond_mask is None:
            cond_mask = torch.ones(B, 1, device=x_noisy.device)
        uid_embed = self.uid_embedding(uid).unsqueeze(1).expand(-1, L, -1) * cond_mask.unsqueeze(-1)
        t_embed = self.t_embedding(t_slot) * cond_mask.unsqueeze(-1)
        wd_embed = self.working_day_embedding(working_day) * cond_mask.unsqueeze(-1)
        dow_embed = self.day_of_week_embedding(day_of_week) * cond_mask.unsqueeze(-1)
        ds_embed = self.diff_step_embedding(diff_step).unsqueeze(1).expand(-1, L, -1)

        x_cat = torch.cat([x_noisy, uid_embed, t_embed, wd_embed, dow_embed, ds_embed], dim=-1)
        out, _ = self.rnn(x_cat)
        eps_hat = self.head(out)
        return eps_hat

def ddpm_loss(model, diffusion, x0, uid, t_slot, working_day, day_of_week, mask, weights, cond_drop_prob=0.1):
    B = x0.size(0)
    diff_t = torch.randint(0, diffusion.timesteps, (B,), device=x0.device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_noisy = diffusion.q_sample(x0, diff_t, noise)

    # 隨機條件丟棄 (CFG 訓練)
    keep = (torch.rand(B, 1, device=x0.device) > cond_drop_prob).float()
    eps_hat = model(x_noisy, uid, t_slot, working_day, day_of_week, diff_t, cond_mask=keep)

    mse = ((eps_hat - noise) ** 2).sum(dim=-1)  # (B, L)
    loss = (mse * weights * mask).sum() / (weights * mask).sum().clamp_min(1.0)
    return loss

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        esd = self.ema_model.state_dict()
        for k in esd.keys():
            esd[k].mul_(self.decay).add_(msd[k], alpha=1.0 - self.decay)

@torch.no_grad()
def generate_future_trajectory_ddpm(model, diffusion, user_test_df, uid, device, guidance_scale=1.5, sampler='ddim', steps=50, eta=0.0):
    model.eval()
    gen_len = user_test_df.shape[0]
    uid_tensor = torch.tensor([uid], dtype=torch.long, device=device)
    t_seq = torch.tensor(user_test_df['t'].values, dtype=torch.long, device=device).unsqueeze(0)
    wd_seq = torch.tensor(user_test_df['working_day'].values, dtype=torch.long, device=device).unsqueeze(0)
    dow_seq = torch.tensor(user_test_df['day_of_week'].values, dtype=torch.long, device=device).unsqueeze(0)

    # 從標準高斯開始於 [-1,1] 空間
    x = torch.randn((1, gen_len, 2), device=device)

    # 建立 timestep 子序列（DDIM）
    if sampler == 'ddim':
        ts = torch.linspace(diffusion.timesteps - 1, 0, steps, dtype=torch.long, device=device)
        ts = ts.tolist()
    else:
        ts = list(reversed(range(diffusion.timesteps)))

    for i, step in enumerate(ts):
        t_step = torch.full((1,), int(step), dtype=torch.long, device=device)

        # 條件與無條件兩次前向，做 CFG
        eps_cond = model(x, uid_tensor, t_seq, wd_seq, dow_seq, t_step, cond_mask=torch.ones(1,1,device=device))
        eps_uncond = model(x, uid_tensor, t_seq, wd_seq, dow_seq, t_step, cond_mask=torch.zeros(1,1,device=device))
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        if sampler == 'ddim' and i < len(ts) - 1:
            t_prev = torch.full((1,), int(ts[i+1]), dtype=torch.long, device=device)
            alpha_t = diffusion.alphas_cumprod[t_step].view(-1,1,1)
            alpha_prev = diffusion.alphas_cumprod[t_prev].view(-1,1,1)
            x0 = diffusion.predict_x0(x, t_step, eps)
            sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev + 1e-8))
            dir_xt = torch.sqrt(torch.clamp(1 - alpha_prev - sigma_t**2, min=0.0)) * eps
            noise = torch.randn_like(x) if eta > 0 else 0.0
            x = torch.sqrt(alpha_prev) * x0 + dir_xt + sigma_t * noise
        else:
            mean, var, _ = diffusion.p_mean_variance(x, t_step, eps)
            if int(step) > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(var) * noise
            else:
                x = mean

    xy = denormalize_xy(x.squeeze(0)).clamp(min=GRID_MIN, max=GRID_MAX).cpu().numpy()
    return xy


if __name__ == "__main__":
    # 資料準備
    raw_x_train_df = pd.read_csv(f'./Training_Testing_Data/A_x_train.csv')
    raw_y_train_df = pd.read_csv(f'./Training_Testing_Data/A_y_train.csv')
    raw_train_df = pd.concat([raw_x_train_df, raw_y_train_df], ignore_index=True)
    set_grid_bounds(max(raw_train_df['x'].max(), raw_train_df['y'].max()))
    raw_feature_df = pd.read_csv(f'./Stability/A_features.csv')
    raw_cluster_df = pd.read_csv(f'./Stability/A_activity_space.csv')
    valid_uid_list = raw_cluster_df[raw_cluster_df['cluster'] == 1]['uid'].unique().tolist()
    valid_uid_list = valid_uid_list
    print(f"有效的使用者數量: {len(valid_uid_list)}")

    # 超參數
    uid_dim = max(valid_uid_list) + 1
    uid_embed_dim = 256
    hidden_dim = 512
    batch_size = 512
    max_len = 500
    num_layers = 1
    num_diff_steps = 1000          # 增加 T，配合 DDIM 可快速取樣
    lr = 3e-4                      # 降低學習率，讓訓練更穩
    cond_drop_prob = 0.1
    ema_decay = 0.999

    dataset = TrajectoryDataset(raw_train_df, valid_uid_list, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CondDDPM(uid_dim=uid_dim, uid_embed_dim=uid_embed_dim, hidden_dim=hidden_dim,
                     timesteps=num_diff_steps, num_layers=num_layers).to(device)
    diffusion = GaussianDiffusion(timesteps=num_diff_steps, device=device, beta_schedule='cosine')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ema = EMA(model, decay=ema_decay)

    # 訓練 + EarlyStopping（以訓練 loss）
    epochs = 5000
    patience = 200
    best_loss = float('inf')
    wait = 0
    loss_list = []

    os.makedirs('./ckpt/DDPM', exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, mask, lengths, uid, t_slot, working_day, day_of_week, weights in dataloader:
            x = normalize_xy(x[..., :2].to(device))   # 正規化到 [-1,1]
            mask = mask.to(device)
            t_slot = t_slot.to(device)
            working_day = working_day.long().to(device)
            day_of_week = day_of_week.to(device)
            uid = torch.tensor(uid, dtype=torch.long, device=device)
            weights = weights.to(device)

            optimizer.zero_grad()
            loss = ddpm_loss(model, diffusion, x, uid, t_slot, working_day, day_of_week, mask, weights, cond_drop_prob=cond_drop_prob)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_list.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            torch.save(model.state_dict(), "./ckpt/DDPM/ddpm_model_best.pth")
            torch.save(ema.ema_model.state_dict(), "./ckpt/DDPM/ddpm_model_ema_best.pth")
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
    # 載入最佳 EMA 權重推論
    ema_path = "./ckpt/DDPM/ddpm_model_ema_best.pth"
    if os.path.exists(ema_path):
        model.load_state_dict(torch.load(ema_path, map_location=device))
    else:
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
        future_traj = generate_future_trajectory_ddpm(
            model, diffusion, user_test_df, uid, device,
            guidance_scale=2.0, sampler='ddim', steps=100, eta=0.0
        )
        for i, row in enumerate(future_traj):
            d = int(user_test_df.iloc[i]['d'])
            t_val = int(user_test_df.iloc[i]['t'])
            x_pred = int(np.clip(row[0], 1, 200))
            y_pred = int(np.clip(row[1], 1, 200))
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
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('y')
        axes[0, i].set_aspect('equal')
        axes[0, i].set_xlim(1, 100)
        axes[0, i].set_ylim(1, 100)
        axes[0, i].grid(True)
        axes[0, i].invert_yaxis()
        axes[0, i].legend()

        axes[1, i].scatter(ddpm_pred_df[ddpm_pred_df['uid'] == uid]['x'],
                           ddpm_pred_df[ddpm_pred_df['uid'] == uid]['y'],
                           label='DDPM', alpha=0.8, s=10, color='blue', marker='x')
        axes[1, i].scatter(gt_df[gt_df['uid'] == uid]['x'],
                           gt_df[gt_df['uid'] == uid]['y'],
                           label='gt', alpha=0.1, s=3, color='green')
        axes[1, i].set_title(f'UID {uid} DDPM')
        axes[1, i].set_xlabel('x')  
        axes[1, i].set_ylabel('y')
        axes[1, i].set_aspect('equal')
        axes[1, i].set_xlim(1, 100)
        axes[1, i].set_ylim(1, 100)
        axes[1, i].grid(True)
        axes[1, i].invert_yaxis()
        axes[1, i].legend()

        axes[2, i].scatter(gt_df[gt_df['uid'] == uid]['x'],
                           gt_df[gt_df['uid'] == uid]['y'],
                           label='gt', alpha=0.5, s=10, color='green')
        axes[2, i].set_title(f'UID {uid} GT')
        axes[2, i].set_xlabel('x')  
        axes[2, i].set_ylabel('y')
        axes[2, i].set_aspect('equal')
        axes[2, i].set_xlim(1, 100)
        axes[2, i].set_ylim(1, 100)
        axes[2, i].grid(True)
        axes[2, i].invert_yaxis()
        axes[2, i].legend()
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