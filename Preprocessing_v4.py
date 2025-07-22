import os
import seaborn as sns
import numpy as np
import pandas as pd
import hdbscan
import time
import fastdtw
import matplotlib.pyplot as plt
import matplotlib.animation as anime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from matplotlib.ticker import MultipleLocator, AutoLocator
from DataVisualize_v2 import DataVisualizer as dv
from collections import Counter
from scipy.stats import entropy
from fastdtw import fastdtw
from scipy import stats
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class DataPreprocessor:
    def __init__(self, city_name, data_input):
        self.city_name = city_name
        if isinstance(data_input, pd.DataFrame):
            self.raw_data_df = data_input
            print(f"直接使用DataFrame資料，共有{self.raw_data_df.shape[0]}筆資料\n",
                  f"資料範圍: uid={self.raw_data_df['uid'].min()}~{self.raw_data_df['uid'].max()}\n",
                  f"時間範圍: days={self.raw_data_df['d'].min()}~{self.raw_data_df['d'].max()}\n")
        elif isinstance(data_input, str):
            self.data_path = data_input
            self.raw_data_df = pd.read_csv(self.data_path, header=0, dtype=int)
            print(f"讀取資料成功，共有{self.raw_data_df.shape[0]}筆資料\n",
                f"資料範圍: uid={self.raw_data_df['uid'].min()}~{self.raw_data_df['uid'].max()}\n",
                f"時間範圍: days={self.raw_data_df['d'].min()}~{self.raw_data_df['d'].max()}\n")
        else:
            raise ValueError("只能接受DataFrame或資料路徑字串（csv檔）。")

    def get_raw_data(self):
        return self.raw_data_df

    def label_working_day(self, df):
        """
        將工作日標記為1，非工作日標記為0。
        1: Monday, 2: Tuesday, 3: Wednesday, 4: Thursday, 5: Friday, 6: Saturday, 0: Sunday
        額外規則：第31天和第35天也是假日
        """
        # 城市A和B是1~75連續，已知第七天是禮拜五，則可推出每週的對應關係。
        if self.city_name == 'A' or self.city_name == 'B': 
            df['day_of_week'] = ((df['d'] - 7) % 7 + 5) % 7
            df['working_day'] = 1
            df.loc[df['day_of_week'].isin([6, 0]), 'working_day'] = 0

        # 城市C和D是1~60是一個區間(一樣第七天是禮拜五)，而61~75是另一個區間(第65和第72是禮拜五)
        elif self.city_name == 'C' or self.city_name == 'D':
            df['day_of_week'] = 0  # 先預設
            # 1~60天：第7天是禮拜五
            mask1 = df['d'] <= 60
            df.loc[mask1, 'day_of_week'] = ((df.loc[mask1, 'd'] - 7) % 7 + 5) % 7
            # 61~75天：第65天是禮拜五
            mask2 = df['d'] > 60
            df.loc[mask2, 'day_of_week'] = ((df.loc[mask2, 'd'] - 65) % 7 + 5) % 7
            df['working_day'] = 1
            df.loc[df['day_of_week'].isin([6, 0]), 'working_day'] = 0

        # 額外規則：第31天和第35天也是假日
        df.loc[df['d'].isin([31, 35]), 'working_day'] = 0

        return df

    def get_training_testing_data(self):
        """
        依據訓練及測試分類，並儲存成新的csv檔案。
        """
        os.makedirs('./Training_Testing_Data', exist_ok=True)

        # label working day
        self.raw_data_df = self.label_working_day(self.raw_data_df)

        # 找到x=999的最小uid
        split_point = self.raw_data_df[self.raw_data_df['x'] == 999]['uid'].unique().min()
        print(f"資料分割點: uid={split_point}\n")

        # x_train;x_test (x,y中無999的uid，完整1-75天資料)
        x_train_df = self.raw_data_df[(self.raw_data_df['d'] <= 60) & (self.raw_data_df['uid'] < split_point)]
        x_test_df = self.raw_data_df[(self.raw_data_df['d'] > 60) & (self.raw_data_df['uid'] < split_point)]
        x_train_df.to_csv(f'./Training_Testing_Data/{self.city_name}_x_train.csv', index=False)
        x_test_df.to_csv(f'./Training_Testing_Data/{self.city_name}_x_test.csv', index=False)
        print(f"x_train有{x_train_df.shape[0]}筆資料, 有{x_train_df['uid'].nunique()}個uid, 從{x_train_df['uid'].min()}到{x_train_df['uid'].max()}")
        print(f"x_test有{x_test_df.shape[0]}筆資料, 有{x_test_df['uid'].nunique()}個uid, 從{x_test_df['uid'].min()}到{x_test_df['uid'].max()}\n")

        # y_train;y_test (y中有999的uid)
        y_train_df = self.raw_data_df[(self.raw_data_df['d'] <= 60) & (self.raw_data_df['uid'] >= split_point)]
        y_test_df = self.raw_data_df[(self.raw_data_df['d'] > 60) & (self.raw_data_df['uid'] >= split_point)]
        y_train_df.to_csv(f'./Training_Testing_Data/{self.city_name}_y_train.csv', index=False)
        y_test_df.to_csv(f'./Training_Testing_Data/{self.city_name}_y_test.csv', index=False)
        print(f"y_train有{y_train_df.shape[0]}筆資料, 有{y_train_df['uid'].nunique()}個uid, 從{y_train_df['uid'].min()}到{y_train_df['uid'].max()}")
        print(f"y_test有{y_test_df.shape[0]}筆資料, 有{y_test_df['uid'].nunique()}個uid, 從{y_test_df['uid'].min()}到{y_test_df['uid'].max()}\n")


        print("資料處理完成，已儲存在Training_Testing_Data資料夾中\n")
        return x_train_df, x_test_df, y_train_df, y_test_df

    def stability_analysis_std(self, df, output_name):
        """
        歐式距離std計算，並儲存工作日和非工作日的標準差資料。
        新增x_std_mean, y_std_mean欄位（每個uid的x_std/y_std平均值），計算時忽略值為0或count<2的情況。
        """
        os.makedirs('./Stability', exist_ok=True)

        # 工作日的標準差
        working_day_df = df[df['working_day'] == 1]
        working_day_std_df = working_day_df.groupby(['uid', 't']).agg(
            x_mean=('x', 'mean'),
            x_std=('x', 'std'),
            x_count=('x', 'count'),
            y_mean=('y', 'mean'),
            y_std=('y', 'std'),
            y_count=('y', 'count')
        ).reset_index()
        # 建立所有 uid-t 組合
        all_uids = working_day_std_df['uid'].unique()
        all_t = np.sort(working_day_std_df['t'].unique())  
        full_index = pd.MultiIndex.from_product([all_uids, all_t], names=['uid', 't'])
        working_day_std_df = working_day_std_df.set_index(['uid', 't']).reindex(full_index).fillna(0).reset_index()

        for col in ['x_mean', 'x_std', 'y_mean', 'y_std']:
            if col in working_day_std_df.columns:
                working_day_std_df[col] = working_day_std_df[col].round(1)

        # 新增x_std_mean, y_std_mean欄位（忽略0或count<=2）
        def mean_ignore_zero_and_count(series, count_series):
            mask = (series != 0) & (count_series > 2)
            filtered = series[mask]
            return filtered.mean() if not filtered.empty else 0

        x_std_mean_map = (
            working_day_std_df.groupby('uid')
            .apply(lambda g: mean_ignore_zero_and_count(g['x_std'], g['x_count']))
            .round(2)
        )
        y_std_mean_map = (
            working_day_std_df.groupby('uid')
            .apply(lambda g: mean_ignore_zero_and_count(g['y_std'], g['y_count']))
            .round(2)
        )
        working_day_std_df['x_std_mean'] = working_day_std_df['uid'].map(x_std_mean_map)
        working_day_std_df['y_std_mean'] = working_day_std_df['uid'].map(y_std_mean_map)

        working_day_std_df.to_csv(f'./Stability/{output_name}_working_day_stability.csv', index=False)
        print(f"工作日標準差資料已儲存至 ./Stability/{output_name}_working_day_stability.csv")

        # 非工作日的標準差
        non_working_day_df = df[df['working_day'] == 0]
        non_working_day_std_df = non_working_day_df.groupby(['uid', 't']).agg(
            x_mean=('x', 'mean'),
            x_std=('x', 'std'),
            x_count=('x', 'count'),
            y_mean=('y', 'mean'),
            y_std=('y', 'std'),
            y_count=('y', 'count')
        ).reset_index()
        # 建立所有 uid-t 組合
        all_uids = non_working_day_std_df['uid'].unique()
        all_t = np.sort(non_working_day_std_df['t'].unique())
        full_index = pd.MultiIndex.from_product([all_uids, all_t], names=['uid', 't'])
        non_working_day_std_df = non_working_day_std_df.set_index(['uid', 't']).reindex(full_index).fillna(0).reset_index()

        for col in ['x_mean', 'x_std', 'y_mean', 'y_std']:
            if col in non_working_day_std_df.columns:
                non_working_day_std_df[col] = non_working_day_std_df[col].round(1)

        # 新增x_std_mean, y_std_mean欄位（忽略0或count<2）
        x_std_mean_map = (
            non_working_day_std_df.groupby('uid')
            .apply(lambda g: mean_ignore_zero_and_count(g['x_std'], g['x_count']))
            .round(2)
        )
        y_std_mean_map = (
            non_working_day_std_df.groupby('uid')
            .apply(lambda g: mean_ignore_zero_and_count(g['y_std'], g['y_count']))
            .round(2)
        )
        non_working_day_std_df['x_std_mean'] = non_working_day_std_df['uid'].map(x_std_mean_map)
        non_working_day_std_df['y_std_mean'] = non_working_day_std_df['uid'].map(y_std_mean_map)

        non_working_day_std_df.to_csv(f'./Stability/{output_name}_non_working_day_stability.csv', index=False)
        print(f"非工作日標準差資料已儲存至 ./Stability/{output_name}_non_working_day_stability.csv")

        return working_day_std_df, non_working_day_std_df

    def stability_analysis_trajectories(self, df_path, std_df_path, output_name, IsWorkingDay=True):
        """
        對每個人的 x, y 軌跡做 DTW 分析（使用 fastdtw）。
        只抓 8 點到 18 點的資料，先計算代表性軌跡（最小總DTW距離的那一天），再對每一天的軌跡做 DTW。
        結果儲存每個人每天的 DTW 距離，並加上該 uid 的 DTW 平均值。
        """
        os.makedirs('./Stability', exist_ok=True)
        df = pd.read_csv(df_path)
        std_df = pd.read_csv(std_df_path)

        # 確保working_day為int型態
        if df['working_day'].dtype != int:
            df['working_day'] = df['working_day'].astype(int)

        # 選取x_std_mean和y_std_mean為>5 <20的資料(5以下穩定20以上混亂就不計算)
        valid_uids = std_df[
            std_df['x_std_mean'].between(5, 20) & std_df['y_std_mean'].between(5, 20)
        ]['uid'].unique()

        # 判斷是否為工作日且uid在valid_uids的資料
        if IsWorkingDay:
            working_day_df = df[(df['working_day'] == 1) & (df['uid'].isin(valid_uids))]
        else:
            working_day_df = df[(df['working_day'] == 0) & (df['uid'].isin(valid_uids))]
        uid_count = working_day_df['uid'].nunique()
        count = 0
        results = []

        for uid, group in working_day_df.groupby('uid'):
            iter_start = time.time()
            days = np.sort(group['d'].unique())
            t_range = np.arange(16, 37)  # 16~36對應8點到18點

            # 先用mean的方式計算初始代表軌跡
            mean_traj = []
            for t in t_range:
                sub = group[group['t'] == t]
                if not sub.empty:
                    mean_traj.append([sub['x'].mean(), sub['y'].mean()])
                else:
                    mean_traj.append([0, 0])  # 如果沒有資料，則填0

            # 準備所有天的軌跡
            day_trajs = []
            for day in days:
                sub_day = group[group['d'] == day]
                traj = []
                for idx, t in enumerate(t_range):
                    sub = sub_day[sub_day['t'] == t]
                    if not sub.empty:
                        xy = [sub['x'].values[0], sub['y'].values[0]]
                        traj.append(xy)
                    else: # 若有nan則用mean_traj補
                        xy = [mean_traj[idx][0], mean_traj[idx][1]]
                        traj.append(xy)
                day_trajs.append(np.array(traj))

            # 計算DTW距離矩陣
            n_days = len(day_trajs)
            dtw_matrix = np.zeros((n_days, n_days))
            for i in range(n_days):
                for j in range(i+1, n_days):
                    dist, _ = fastdtw(day_trajs[i], day_trajs[j])
                    dtw_matrix[i, j] = dist
                    dtw_matrix[j, i] = dist

            # 找出代表軌跡（總距離最小的那一天）
            total_dists = dtw_matrix.sum(axis=1)
            rep_idx = np.argmin(total_dists)
            rep_traj = day_trajs[rep_idx]

            # 計算每一天與代表軌跡的DTW距離
            uid_distances = []
            for idx, day in enumerate(days):
                distance, _ = fastdtw(day_trajs[idx], rep_traj)
                results.append({'uid': uid, 'd': day, 'dtw_distance': round(distance, 2)})
                uid_distances.append(distance)

            # 加上該 uid 的 DTW 平均值
            dtw_mean = np.mean(uid_distances) if uid_distances else 0
            for i in range(len(days)):
                results[-len(days)+i]['dtw_mean'] = round(dtw_mean, 2)

            count += 1
            elapsed = time.time() - iter_start
            remaining = uid_count - count
            est_sec = elapsed * remaining
            est_min = int(est_sec // 60)
            est_sec = int(est_sec % 60)
            print(f"處理進度: {count}/{uid_count} (uid={uid})，預估剩餘時間: {est_min}分{est_sec}秒", end='\r')
            # 提早結束測試
            # if count == 100: break

        dtw_df = pd.DataFrame(results)
        dtw_df.to_csv(f'./Stability/{output_name}', index=False)
        print(f"\nDTW分析結果已儲存至 ./Stability/{output_name}")
        return dtw_df

    def stability_analysis_GDiagnostic(self, df_path, std_df_path, alpha, output_path, IsWorkingDay=True):
        os.makedirs('./Stability', exist_ok=True)
        df = pd.read_csv(df_path)
        std_df = pd.read_csv(std_df_path)

        # 計算t_threshold
        t_threshold = stats.norm.ppf(1 - alpha/2)

        # 確保working_day為int型態
        if df['working_day'].dtype != int:
            df['working_day'] = df['working_day'].astype(int)

        # 選取x_std_mean和y_std_mean為>5 (5以下穩定)
        valid_uids = std_df[(std_df['x_std_mean'] >= 5) | (std_df['y_std_mean'] >= 5)]['uid'].unique()

        # 只要8點到18點的資料
        t_range = np.arange(16, 37)

        # 判斷是否為工作日且uid在valid_uids的資料
        if IsWorkingDay:
            df = df[(df['working_day'] == 1) & (df['uid'].isin(valid_uids))]
        else:
            df = df[(df['working_day'] == 0) & (df['uid'].isin(valid_uids))]

        # 計算每個時間點的Geweke diagnostic 
        results = []
        for idx, uid in enumerate(valid_uids):
            start_time = time.time()
            frac_results = {}
            user_df = df[df['uid'] == uid]
            for t in t_range:
                sub = user_df[user_df['t'] == t]
                last_50 = sub[sub['d'] > 30]
                last_50_x_mean = last_50['x'].mean()
                last_50_x_var = last_50['x'].var()
                last_50_y_mean = last_50['y'].mean()
                last_50_y_var = last_50['y'].var()
                for n1_frac in [0.2, 0.3, 0.4, 0.5]:
                    n1 = 60 * n1_frac
                    first_n = sub[sub['d'] <= n1]
                    if len(first_n) < 1 or len(last_50) < 1:
                        frac_results[f"{int(n1_frac*100)}%"] = np.nan  # 無法計算則設為NaN
                        continue
                    first_n_x_mean = first_n['x'].mean()
                    first_n_y_mean = first_n['y'].mean()
                    first_n_x_var = first_n['x'].var()
                    first_n_y_var = first_n['y'].var()
                    z_x = (first_n_x_mean - last_50_x_mean) / np.sqrt(first_n_x_var/len(first_n) + last_50_x_var/len(last_50))
                    z_y = (first_n_y_mean - last_50_y_mean) / np.sqrt(first_n_y_var/len(first_n) + last_50_y_var/len(last_50))
                    # 只要有一個超過閾值就算fail
                    if abs(z_x) > t_threshold or abs(z_y) > t_threshold:
                        frac_results[f"{int(n1_frac*100)}%"] = 0  # fail
                    else:
                        frac_results[f"{int(n1_frac*100)}%"] = 1  # pass

                row = {'uid': uid, 't': t}
                row.update(frac_results)
                results.append(row)

            elapsed = time.time() - start_time
            remaining_time = elapsed*(len(valid_uids) - (idx+1))
            print(f"處理進度: {idx+1}/{len(valid_uids)} (uid={uid})，預估剩餘時間: {remaining_time//60}分, {output_path}", end='\r')

            # if idx == 1000:  # 測試用，處理前1000個uid
            #     break

        geweke_df = pd.DataFrame(results, columns=['uid', 't', '20%', '30%', '40%', '50%'])
        geweke_df.to_csv(f'./Stability/{output_path}', index=False, na_rep='null')
        print(f"Geweke diagnostic 結果已儲存至 ./Stability/{output_path}")
        return geweke_df

    def extract_features(self, df, output_name, valid_uid_list=None):
        if valid_uid_list is not None:
            df = df[df['uid'].isin(valid_uid_list)]
        total_count = df['uid'].nunique()


        # 特徵工程
        features = []
        for uid, group in df.groupby('uid'):
            # 住家點: 取工作日早上0點到6點及晚上8點到12點
            home = group[(group['working_day']==1) & ((group['t']<=12) | (group['t']>=40))]
            home_loc = home.groupby(['x','y']).size().idxmax() if not home.empty else (np.nan, np.nan)
            # 工作點: 工作日/白天最常出現的(x,y)早上8點到下午5點
            work = group[(group['working_day']==1) & (group['t']>=16) & (group['t']<=34)]
            work_loc = work.groupby(['x','y']).size().idxmax() if not work.empty else (np.nan, np.nan)
            # 通勤距離
            commute_dist = round(np.linalg.norm(np.array(home_loc)-np.array(work_loc)), 2) if not np.isnan(home_loc[0]) and not np.isnan(work_loc[0]) else np.nan
            # 熱點: HDBSCAN找活動熱點
            coords = group[['x','y']].values
            clusterer = hdbscan.HDBSCAN(min_cluster_size=15)
            clusterer.fit(coords)
            n_hotspots = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
            # 熱點中心點
            if n_hotspots > 0:
                centers = []
                for l in set(clusterer.labels_):
                    if l == -1:
                        continue
                    center = coords[clusterer.labels_==l].mean(axis=0)
                    centers.append((round(center[0],0), round(center[1],0)))
                hotspot_centers = ';'.join([f'({x},{y})' for x,y in centers])
                bbox = [coords[clusterer.labels_!=-1][:,0].min(), coords[clusterer.labels_!=-1][:,1].min(),
                        coords[clusterer.labels_!=-1][:,0].max(), coords[clusterer.labels_!=-1][:,1].max()]
                radius = np.mean([np.linalg.norm(coords[i]-coords[clusterer.labels_==l].mean(axis=0)) 
                                for l in set(clusterer.labels_) if l!=-1 for i in np.where(clusterer.labels_==l)[0]])
                radius = round(radius, 4) if not np.isnan(radius) else np.nan
            else:
                bbox = [np.nan]*4
                radius = np.nan
                hotspot_centers = ''
            # 活動entropy
            loc_counts = Counter(zip(group['x'], group['y']))
            act_entropy = round(entropy(list(loc_counts.values())), 4)

            features.append({
                'uid': uid,
                'home_x': home_loc[0], 'home_y': home_loc[1],
                'work_x': work_loc[0], 'work_y': work_loc[1],
                'commute_dist': commute_dist,
                'n_hotspots': n_hotspots,
                'bbox_xmin': bbox[0], 'bbox_ymin': bbox[1], 'bbox_xmax': bbox[2], 'bbox_ymax': bbox[3],
                'hotspot_radius': radius,
                'act_entropy': act_entropy,
                'hotspot_centers': hotspot_centers
            })
            print(f"處理進度: {len(features)}/{total_count} (uid={uid})", end='\r')

        output_df = pd.DataFrame(features)
        output_df.to_csv(f'./Stability/{output_name}_features.csv', index=False, na_rep='null')
        print(f"features結果已儲存至 ./Stability/{output_name}_features.csv")
        return output_df
 
        
"""
測試程式碼
"""
if __name__ == "__main__":

    ## 資料特徵工程
    output_name = 'D'
    dp = DataPreprocessor(f'{output_name}', f'./Data/city_{output_name}_challengedata.csv')
    std_df = pd.read_csv('./Stability/A_xtrain_working_day_stability.csv')
    input_df_1 = pd.read_csv(f'./Training_Testing_Data/{output_name}_x_train.csv')
    input_df_2 = pd.read_csv(f'./Training_Testing_Data/{output_name}_y_train.csv')
    input_df = pd.concat([input_df_1, input_df_2], ignore_index=True)
    # valid_uid_list = np.sort(np.random.choice(input_df['uid'].unique(), size=10000, replace=False))
    dp.extract_features(input_df, output_name= output_name)

    ## 分群: 生活圈重疊
    activity_space_df = pd.read_csv(f'./Stability/{output_name}_features.csv')
    print(f"包含{activity_space_df['uid'].nunique()}個uid")
    features = ['bbox_xmin','bbox_ymin','bbox_xmax','bbox_ymax']
    clusterer = hdbscan.HDBSCAN(min_cluster_size=300, min_samples=350)
    cluster_data = activity_space_df[features].fillna(0)
    cluster_labels = clusterer.fit_predict(cluster_data)
    activity_space_df['cluster'] = cluster_labels
    activity_space_df.to_csv(f'./Stability/{output_name}_activity_space.csv', index=False)
    print('分群結果已儲存:', f'./Stability/{output_name}_activity_space.csv')

    # 統一顏色映射: 只對有效cluster分配顏色
    cluster_ids = sorted([cid for cid in activity_space_df['cluster'].dropna().unique() if cid != -1])
    color_map = plt.get_cmap('tab10')
    cluster_color_dict = {cid: color_map(i % 10) for i, cid in enumerate(cluster_ids)}

    # 繪製分群結果 (2維度)
    X = activity_space_df[features].fillna(0).values
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(10, 8))
    # 依據cluster id分配顏色
    colors = activity_space_df['cluster'].apply(lambda x: cluster_color_dict.get(x, '#cccccc'))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=20, alpha=0.5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.title('Cluster Visualization (PCA 2D)')

    # 計算雜訊數量
    n_noise = (activity_space_df['cluster'] == -1).sum()
    print(f"雜訊數量 (cluster=-1): {n_noise}")

    # 自訂圖例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cluster_color_dict[cid], edgecolor='k', label=f'Cluster {cid}') for cid in cluster_ids]
    ax.legend(handles=legend_elements, title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # 畫出多個人的生活範圍(bbox)與home、work點（每群依群內人數動態SAMPLE，最多300人，同群同色）
    fig2, ax2 = plt.subplots(figsize=(10,10))
    for cid in cluster_ids:
        group = activity_space_df[activity_space_df['cluster'] == cid]
        n = len(group)
        # 動態決定sample數量: min(300, max(10, int(n * 0.1)))
        sample_n = min(300, max(10, int(n * 0.1))) if n > 10 else n
        if n > sample_n:
            group = group.sample(sample_n, random_state=42)
            print(f"Cluster {cid} sample size: {sample_n} (original size: {n})")
        color = cluster_color_dict[cid]
        for idx, row in group.iterrows():
            xmin, ymin, xmax, ymax = row['bbox_xmin'], row['bbox_ymin'], row['bbox_xmax'], row['bbox_ymax']
            if not np.isnan(xmin) and not np.isnan(ymin) and not np.isnan(xmax) and not np.isnan(ymax):
                rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor=color, alpha=0.3, linewidth=1.5)
                ax2.add_patch(rect)
            # home點
            if not np.isnan(row['home_x']) and not np.isnan(row['home_y']):
                ax2.scatter(row['home_x'], row['home_y'], c=[color], s=30, marker='o', alpha=0.5, edgecolors='k', linewidths=0.2)
            # work點
            if not np.isnan(row['work_x']) and not np.isnan(row['work_y']):
                ax2.scatter(row['work_x'], row['work_y'], c=[color], s=30, marker='^', alpha=0.5, edgecolors='k', linewidths=0.2)
        # 只加一次圖例
        ax2.plot([], [], color=color, lw=4, label=f'Cluster {cid}--{n} users')
    ax2.set_xlim(1, 200)
    ax2.set_ylim(1, 200)  # y軸反轉
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(MultipleLocator(10))
    ax2.yaxis.set_major_locator(MultipleLocator(10))
    ax2.set_title('User Living Ranges by Cluster (dynamic sample per cluster, max 300 users per cluster)')
    ax2.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    ## 地圖上的POI
    # 讀取 features 檔案
    # output_name = 'B'
    features_df = pd.read_csv(f'./Stability/{output_name}_features.csv')

    # 1. 收集所有 home、work、hotspot_centers
    points = []
    for _, row in features_df.iterrows():
        if not np.isnan(row['home_x']) and not np.isnan(row['home_y']):
            points.append((row['home_x'], row['home_y']))
        if not np.isnan(row['work_x']) and not np.isnan(row['work_y']):
            points.append((row['work_x'], row['work_y']))
        if isinstance(row['hotspot_centers'], str) and row['hotspot_centers']:
            for center in row['hotspot_centers'].split(';'):
                x, y = eval(center)
                points.append((x, y))

    # 2. 統計出現次數
    counter = Counter(points)
    sorted_points = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    # 3. 最小距離過濾
    top_poi = 50
    min_dist = 2  # 你可以調整這個距離
    selected_poi = []
    for (x, y), count in sorted_points:
        if all(np.linalg.norm(np.array([x, y]) - np.array([px, py])) >= min_dist for px, py, _ in selected_poi):
            selected_poi.append((x, y, count))
        if len(selected_poi) >= top_poi:  # 你可以調整最多POI數量
            break

    # 4. 儲存POI到CSV
    poi_df = pd.DataFrame(selected_poi, columns=['poi_x', 'poi_y', 'count'])
    poi_df.to_csv(f'./Stability/{output_name}_poi.csv', index=False)
    print(f"POI已儲存至 ./Stability/{output_name}_poi.csv")

    # 4. 可視化POI
    plt.figure(figsize=(8, 8))
    # 使用colormap分配顏色
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(len(poi_df))]
    plt.scatter(poi_df['poi_x'], poi_df['poi_y'], s=20, c=colors, alpha=1, edgecolors='k', linewidths=0.8)
    plt.xlim(1, 200)
    plt.ylim(1, 200)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().invert_yaxis()  # y軸反轉
    plt.title('Top POI Distribution')
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MultipleLocator(10))
    plt.gca().yaxis.set_major_locator(MultipleLocator(10))
    plt.tight_layout()
    plt.show()

    ## POI自動對齊

   

    


  