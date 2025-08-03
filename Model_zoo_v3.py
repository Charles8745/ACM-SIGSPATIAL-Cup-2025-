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
        merged_df['delta_t'] = merged_df['delta_t'].fillna(1)
        merged_df['delta_t'] = merged_df['delta_t'].replace(0, 1e-6)  # 避免為0
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
        # 正規化
        merged_df['x'] = (merged_df['x'] - 100.5) / 99.5
        merged_df['y'] = (merged_df['y'] - 100.5) / 99.5
        
        # 分出訓練用和測試用的資料
        train_data = merged_df[merged_df['d'] <= 60]
        test_data = merged_df[merged_df['d'] > 60]

        # 選擇要得feature列
        feature_cols = ['x', 'y', 't', 'day_of_week', 'working_day', 'delta_t']
        assert not merged_df[feature_cols].isnull().any().any(), "資料有 nan"
        train_data = train_data[feature_cols].astype(float)
        test_data = test_data[feature_cols].astype(float)
        train_seq = [row for row in train_data.values]
        test_seq = [row for row in test_data.values]

        return train_data, test_data, train_seq, test_seq

class SlidingSeqDataset(Dataset):
    def __init__(self, train_seq, batch_size):
        self.data_x = np.array(train_seq[:-1]) # 所有行，除了最後一行
        self.data_y = np.array(train_seq[1:]) # 所有行，除了第一行
        self.batch_size = batch_size
        # 預先計算所有可用的起始點
        seq_size = len(self.data_x)
        self.start_points = sorted([int(x) for x in np.random.randint(0, seq_size, batch_size)])

    def __len__(self):
        """
        作用：回傳這個資料集（Dataset）有多少個「樣本」
        """
        # 每個起始點都能產生一個序列
        return len(self.start_points)
    
    def __getitem__(self, idx):
        """
        作用：根據給定的索引 idx，回傳一個樣本（或一組資料）。
        讓你可以用 dataset[idx] 取得第 idx 個資料。
        DataLoader 會自動呼叫這個方法來取出 batch。
        """
        j = self.start_points[idx]
        x_seq = self.data_x[j:]
        y_seq = self.data_y[j:]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)

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
        assert not torch.isnan(t).any(), "delta_t 有 nan"
        assert not (t == 0).any(), "delta_t 有 0"
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

class TLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TLSTMModel, self).__init__()
        self.cell = TLSTMCell(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        # x: [batch, seq_len, feature_dim]
        batch_size, seq_len, feature_dim = x.size()
        h = torch.zeros(batch_size, self.cell.hidden_dim, device=x.device)
        c = torch.zeros(batch_size, self.cell.hidden_dim, device=x.device)
        # 每個 batch 一個 list
        batch_hiddens = [[] for _ in range(batch_size)]

        # 計算每個 batch 最後一個 x, y 不為 0 的 index
        xy = x[:, :, 0:2]
        mask = (xy.abs().sum(dim=2) != 0)  # [batch, seq_len]
        valid_lens = mask.sum(dim=1)  # [batch]，每個 batch 有效長度
        max_valid_len = valid_lens.max().item()

        for t in range(max_valid_len):
            # 只對有效序列做運算，超過有效長度的 batch 不更新 h, c
            active_mask = (valid_lens > t)
            if not active_mask.any():
                break
            xt = x[active_mask, t, :-1]
            delta_t = x[active_mask, t, -1].unsqueeze(1)
            h_active = h[active_mask]
            c_active = c[active_mask]
            h_new, c_new = self.cell(xt, delta_t, h_active, c_active)
            h[active_mask] = h_new
            c[active_mask] = c_new
            # 只把 active 的 h_new append 到對應 batch
            idxs = active_mask.nonzero(as_tuple=True)[0]
            for i, idx in enumerate(idxs):
                batch_hiddens[idx].append(h_new[i].unsqueeze(0))  # [1, hidden_dim]
                
        # 對每個 batch 用 pad_sequence 補齊
        padded_hiddens = pad_sequence(
            [torch.cat(hs, dim=0) if len(hs) > 0 else torch.zeros(1, self.cell.hidden_dim, device=x.device) for hs in batch_hiddens],
            batch_first=True
        )  # [batch, max_valid_len, hidden_dim]

        out = self.fc1(padded_hiddens)
        out = self.fc2(F.relu(out))
        return out, valid_lens

def padding_fn(batch):
    x_seqs, y_seqs = zip(*batch)
    x_padded = pad_sequence(x_seqs, batch_first=True, padding_value=0)  # [batch, max_seq_len, feature_dim]
    y_padded = pad_sequence(y_seqs, batch_first=True, padding_value=0)  # [batch, max_seq_len, 3])
    return x_padded, y_padded

if __name__ == "__main__":
    start_time = time.time()
    train_path = './Training_Testing_Data/A_x_train.csv'
    test_path = './Training_Testing_Data/A_x_test.csv'
    feature_path = './Stability/A_features.csv'
    target_uid = 91
    batch_size = 256
    num_epochs = 128
    input_dim = 5  # x, y, 't', 'day_of_week', 'working_day', (delta_t單獨處理)
    hidden_dim = 256
    output_dim = 2  # 只預測 x, y
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")  # 強制使用 CPU
    best_loss = float('inf')
    patience = 10
    counter = 0
    os.makedirs('./ckpt/TLSTM', exist_ok=True)
    train_data, test_data, train_seq, test_seq  = data_preprocessing.load_merge_feature(
        train_path, test_path, feature_path, target_uid)
    batch_size = batch_size if batch_size < len(train_data)-1 else len(train_data)-1  # 確保不超過資料長度
    dataset = SlidingSeqDataset(train_seq, batch_size=batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=padding_fn)

    model = TLSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    loss_history = []

    # Train
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred_xy, valid_lens = model(batch_x)  # pred_xy: [batch, max_valid_len, 2]
            xy = batch_y[:, :, 0:2]
            mask = (xy.abs().sum(dim=2) != 0)  # [batch, seq_len]
            
            # 只計算有效長度內的 loss
            losses = []
            for i in range(batch_x.size(0)):
                valid_len = valid_lens[i]
                if valid_len == 0:
                    continue
                pred = pred_xy[i, :valid_len, :]  # [valid_len, 2]
                true = batch_y[i, :valid_len, 0:2]  # [valid_len, 2]   

                # 計算 weighted loss，分開對 x, y
                pred_x = pred[:, 0]
                pred_y = pred[:, 1]
                true_x = true[:, 0]
                true_y = true[:, 1]

                loss_x = ((pred_x - true_x) ** 2).mean()
                loss_y = ((pred_y - true_y) ** 2).mean()
                loss = 0.6 * loss_x + 0.4 * loss_y  # 可調整權重
                loss = loss.mean()
                losses.append(loss)
                    
            if len(losses) > 0:
                loss = torch.stack(losses).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)  # 加入 loss 紀錄
        if (epoch+1) % 10 == 0 or epoch == num_epochs - 1:
            print(f"Pred vs True (last 5):\n{np.concatenate([pred.detach().cpu().numpy()[-5:] * 99.5 + 100.5, true.detach().cpu().numpy()[-5:] * 99.5 + 100.5], axis=1).astype(int)}")
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Early stopping 檢查
        if avg_loss < best_loss - 1e-5:  # 1e-4 可調整，避免浮點誤差
            best_loss = avg_loss
            counter = 0
            # 你也可以在這裡順便存下最佳模型
            torch.save(model.state_dict(), f'./ckpt/TLSTM/{target_uid}_tlstm_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time//60:.2f} minutes.")

    # 訓練結束後畫 loss 曲線
    plt.figure()
    plt.plot(loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid()
    plt.show()


    # 測試模型
    model.load_state_dict(torch.load(f'./ckpt/TLSTM/{target_uid}_tlstm_model.pth', weights_only=True))
    # 取得 1~60 天資料
    history_df = train_data.astype(float).values
    # 取得 61~75 天的 t, day_of_week, working_day, delta_t
    future_t = test_data['t'].astype(float).values
    future_day_of_week = test_data['day_of_week'].astype(float).values
    future_working_day = test_data['working_day'].astype(float).values
    future_delta_t = test_data['delta_t'].astype(float).values
    future_true_xy = test_data[['x', 'y']].values * 99.5 + 100.5  # 還原
    # 預測 61~75 天
    model.eval()
    with torch.no_grad():
        # 先用 1~60天的資料初始化
        history = torch.tensor(history_df, dtype=torch.float32).unsqueeze(0).to(device)  # [1, seq_len, 3]
        pred_xy, valid_lens = model(history)
        # 取最後一個有效 hidden state 的 x, y 作為預測起點
        last_xy = history[0, -1, :2].cpu().numpy()  # [x, y]
        # 預測未來 15 天
        preds = []
        for i in range(len(future_delta_t)):
            input_step = np.concatenate([
                last_xy, 
                [future_t[i], future_day_of_week[i], future_working_day[i], future_delta_t[i]]
            ])  # [x, y, t, day_of_week, working_day, delta_t]
            input_tensor = torch.tensor(input_step, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            pred_xy, _ = model(input_tensor)
            last_xy = pred_xy[0, -1, :].cpu().numpy()
            preds.append(last_xy)
        
        # 輸出預測與真實
        preds = np.array(preds) * 99.5 + 100.5
        print("61~75天預測 vs 真實 (x, y):")
        out = np.concatenate([preds.astype(int), future_true_xy.astype(int)], axis=1)
        print("pred_x, pred_y, true_x, true_y")
        print(out[-20:])  # 只顯示最後20點的預測結果

    # 包裝成輸出
    raw_test_df = pd.read_csv(test_path)
    uid_test_df = raw_test_df[raw_test_df['uid'] == target_uid]
    uid_test_df['x'] = preds[:, 0]
    uid_test_df['y'] = preds[:, 1]
    output_df = uid_test_df[['uid', 'd', 't', 'x', 'y']]
    output_df = pd.DataFrame(output_df, columns=['uid', 'd', 't', 'x', 'y']).astype(int)

    # 輸出成 csv
    os.makedirs('./Predictions/TLSTM', exist_ok=True)
    output_df.to_csv(f'./Predictions/TLSTM/{target_uid}.csv', index=False)
    print(f"已輸出預測結果到 ./Predictions/TLSTM/{target_uid}.csv")

    # LSTM成績計算
    GEOBLEU_scores = []
    DTW_scores = []
    generated_df = pd.read_csv(f'./Predictions/TLSTM/{target_uid}.csv')
    reference_df = pd.read_csv('./Training_Testing_Data/A_x_test.csv')

    gen_user = generated_df[generated_df['uid'] == target_uid]
    ref_user = reference_df[reference_df['uid'] == target_uid]

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

    final_GEOBLEU_score = sum(GEOBLEU_scores) / len(GEOBLEU_scores) if GEOBLEU_scores else 0.0
    final_DTW_score = sum(DTW_scores) / len(DTW_scores) if DTW_scores else 0.0
    print(f'\nLSTM---GEOBLEU平均分數: {final_GEOBLEU_score:.4f}, DTW平均分數: {final_DTW_score:.4f}')

