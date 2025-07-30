import os
import osmnx as ox
import seaborn as sns
import numpy as np
import pandas as pd
import hdbscan
import time
import fastdtw
import matplotlib.pyplot as plt
import matplotlib.animation as anime
import requests
import geopandas as gpd
import contextily as ctx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from matplotlib.ticker import MultipleLocator, AutoLocator
from DataVisualize_v2 import DataVisualizer as dv
from collections import Counter
from scipy.stats import entropy
from fastdtw import fastdtw
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.extra.rate_limiter import RateLimiter
from shapely.geometry import Point, LineString
import folium
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import ast


class DataPreprocessor:
    def __init__(self):
        pass

    def label_working_day(self, df, city_name):
        """
        將工作日標記為1，非工作日標記為0。
        1: Monday, 2: Tuesday, 3: Wednesday, 4: Thursday, 5: Friday, 6: Saturday, 0: Sunday
        額外規則：第31天和第35天也是假日
        """
        # 城市A和B是1~75連續，已知第七天是禮拜五，則可推出每週的對應關係。
        if city_name == 'A' or city_name == 'B':
            df['day_of_week'] = ((df['d'] - 7) % 7 + 5) % 7
            df['working_day'] = 1
            df.loc[df['day_of_week'].isin([6, 0]), 'working_day'] = 0

        # 城市C和D是1~60是一個區間(一樣第七天是禮拜五)，而61~75是另一個區間(第65和第72是禮拜五)
        elif city_name == 'C' or city_name == 'D':
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

    def get_training_testing_data(self, city_name, data_input_path):
        """
        依據訓練及測試分類，並儲存成新的csv檔案。
        """
        os.makedirs('./Training_Testing_Data', exist_ok=True)
        # 讀取資料
        raw_data_df = pd.read_csv(data_input_path, header=0, dtype=int) 

        # label working day
        raw_data_df = self.label_working_day(df=raw_data_df, city_name=city_name)

        # 找到x=999的最小uid
        split_point = raw_data_df[raw_data_df['x'] == 999]['uid'].unique().min()
        print(f"資料分割點: uid={split_point}\n")

        # x_train;x_test (x,y中無999的uid，完整1-75天資料)
        x_train_df = raw_data_df[(raw_data_df['d'] <= 60) & (raw_data_df['uid'] < split_point)]
        x_test_df = raw_data_df[(raw_data_df['d'] > 60) & (raw_data_df['uid'] < split_point)]
        x_train_df.to_csv(f'./Training_Testing_Data/{city_name}_x_train.csv', index=False)
        x_test_df.to_csv(f'./Training_Testing_Data/{city_name}_x_test.csv', index=False)
        print(f"x_train有{x_train_df.shape[0]}筆資料, 有{x_train_df['uid'].nunique()}個uid, 從{x_train_df['uid'].min()}到{x_train_df['uid'].max()}")
        print(f"x_test有{x_test_df.shape[0]}筆資料, 有{x_test_df['uid'].nunique()}個uid, 從{x_test_df['uid'].min()}到{x_test_df['uid'].max()}\n")

        # y_train;y_test (y中有999的uid)
        y_train_df = raw_data_df[(raw_data_df['d'] <= 60) & (raw_data_df['uid'] >= split_point)]
        y_test_df = raw_data_df[(raw_data_df['d'] > 60) & (raw_data_df['uid'] >= split_point)]
        y_train_df.to_csv(f'./Training_Testing_Data/{city_name}_y_train.csv', index=False)
        y_test_df.to_csv(f'./Training_Testing_Data/{city_name}_y_test.csv', index=False)
        print(f"y_train有{y_train_df.shape[0]}筆資料, 有{y_train_df['uid'].nunique()}個uid, 從{y_train_df['uid'].min()}到{y_train_df['uid'].max()}")
        print(f"y_test有{y_test_df.shape[0]}筆資料, 有{y_test_df['uid'].nunique()}個uid, 從{y_test_df['uid'].min()}到{y_test_df['uid'].max()}\n")


        print("資料處理完成，已儲存在Training_Testing_Data資料夾中\n")
        return x_train_df, x_test_df, y_train_df, y_test_df

    def stability_analysis_std(self, input_df, output_name='XX_xORytrain'):
        """
        歐式距離std計算，並儲存工作日和非工作日的標準差資料。
        新增x_std_mean, y_std_mean欄位（每個uid的x_std/y_std平均值），計算時忽略值為0或count<2的情況。
        """
        os.makedirs('./Stability', exist_ok=True)

        # 讀取資料
        if isinstance(input_df, str):
            df = pd.read_csv(input_df, header=0, dtype=int)
        elif not isinstance(input_df, pd.DataFrame):
            raise ValueError("只能接受DataFrame或資料路徑字串（csv檔）。")

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

    def stability_analysis_trajectories(self, df_path, std_df_path, output_name='XX_xORytrain', IsWorkingDay=True):
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
        dtw_df.to_csv(f'./Stability/{output_name}_working_day_dtw.csv', index=False)
        print(f"\nDTW分析結果已儲存至 ./Stability/{output_name}_working_day_dtw.csv")
        return dtw_df

    def stability_analysis_GDiagnostic(self, df_path, std_df_path, alpha, output_name='XX_xORytrain', IsWorkingDay=True):
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
        geweke_df.to_csv(f'./Stability/{output_name}_working_day_geweke.csv', index=False, na_rep='null')
        print(f"Geweke diagnostic 結果已儲存至 ./Stability/{output_name}_working_day_geweke.csv")
        return geweke_df

    def extract_features(self, input_x_train, input_y_train, output_name='Cityname', valid_uid_list=None):  
        # 讀取資料
        if isinstance(input_x_train, str):
            input_x_train = pd.read_csv(input_x_train, header=0, dtype=int)
        else:
            raise ValueError("只能接受DataFrame或資料路徑字串（csv檔）。")
        if isinstance(input_y_train, str):
            input_y_train = pd.read_csv(input_y_train, header=0, dtype=int)
        else:
            raise ValueError("只能接受DataFrame或資料路徑字串（csv檔）。")


        # 合併訓練資料
        df = pd.concat([input_x_train, input_y_train], ignore_index=True)
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

        # 儲存特徵結果
        os.makedirs('./Stability', exist_ok=True)
        output_df = pd.DataFrame(features)
        output_df.to_csv(f'./Stability/{output_name}_features.csv', index=False, na_rep='null')
        print(f"features結果已儲存至 ./Stability/{output_name}_features.csv")
        return output_df
 
    def activity_space_cluster(self, features_df, output_name='Cityname', output_img=False):
        """
        若cluster標籤為-1，則為雜訊點
        """
        # 讀取資料
        if isinstance(features_df, str):
            features_df = pd.read_csv(features_df, header=0, dtype=int)
        elif not isinstance(features_df, pd.DataFrame):
            raise ValueError("只能接受DataFrame或資料路徑字串（csv檔）。")
        
        # 分群: 生活圈重疊
        activity_space_df = features_df
        print(f"包含{activity_space_df['uid'].nunique()}個uid")
        features = ['bbox_xmin','bbox_ymin','bbox_xmax','bbox_ymax']
        clusterer = hdbscan.HDBSCAN(min_cluster_size=300, min_samples=350)
        cluster_data = activity_space_df[features].fillna(0)
        cluster_labels = clusterer.fit_predict(cluster_data)
        activity_space_df['cluster'] = cluster_labels
        n_noise = (activity_space_df['cluster'] == -1).sum()
        print(f"雜訊數量 (cluster=-1): {n_noise}")
        activity_space_df.to_csv(f'./Stability/{output_name}_activity_space.csv', index=False)
        print('分群結果已儲存:', f'./Stability/{output_name}_activity_space.csv')

        if output_img:
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

            # 自訂圖例
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=cluster_color_dict[cid], edgecolor='k', label=f'Cluster {cid}') for cid in cluster_ids]
            ax.legend(handles=legend_elements, title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            # 畫出多個人的生活範圍(bbox)與home、work點（每群依群內人數動態SAMPLE，最多300人，同群同色）
            fig2, ax2 = plt.subplots(figsize=(20,20))
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
            os.makedirs('./Animations', exist_ok=True)
            plt.savefig(f'./Animations/{output_name}_activity_space_clusters.png')
            plt.tight_layout()
            plt.show()

    def extract_POI(self, features_df, output_name='Cityname', top_poi=50, min_dist=2, output_img=False):
        """
        top_poi: 你可以調整輸出前多少POI數量
        min_dist: 你可以調整最小距離過濾，避免POI過於接近
        """
        # 讀取資料
        if isinstance(features_df, str):
            features_df = pd.read_csv(features_df, header=0, dtype=int)
        elif not isinstance(features_df, pd.DataFrame):
            raise ValueError("只能接受DataFrame或資料路徑字串（csv檔）。")
        
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
        selected_poi = []
        for (x, y), count in sorted_points:
            if all(np.linalg.norm(np.array([x, y]) - np.array([px, py])) >= min_dist for px, py, _ in selected_poi):
                selected_poi.append((x, y, count))
            if len(selected_poi) >= top_poi:  # 你可以調整最多POI數量
                break

        # 4. 儲存POI到CSV
        poi_df = pd.DataFrame(selected_poi, columns=['poi_x', 'poi_y', 'count'])
        poi_df.to_csv(f'./Stability/{output_name}_top_poi.csv', index=False)
        print(f"POI已儲存至 ./Stability/{output_name}_top_poi.csv")

        if output_img:
            # 5. 可視化POI
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
            os.makedirs('./Animations', exist_ok=True)
            plt.savefig(f'./Animations/{output_name}_top{top_poi}_poi_distribution.png')
            print(f"Top_POI分布圖已儲存至 ./Animations/{output_name}_top{top_poi}_poi_distribution.png")
            plt.show()

    def map_alignment(self, poi_xy_arr, poi_latlon_arr, output_name='Cityname'):
        """
        將POI的xy座標與經緯度對齊，並可視化在地圖上。
        poi_xy_arr: np.array([[135,77], [129,81], ...])
        poi_latlon_arr: np.array([[35.171686742452046, 136.8819338645865],[35.136656631816585, 136.9096687302056],...])
        """
        xy = poi_xy_arr
        latlon = poi_latlon_arr
        if xy.shape[0] != latlon.shape[0]:
            raise ValueError("xy和latlon的行數必須相同。")
        
        # x 代表緯度, y 代表經度，分開建模
        self.poly = PolynomialFeatures(degree=2)
        x_arr = xy[:,0].reshape(-1,1)
        y_arr = xy[:,1].reshape(-1,1)
        lat_arr = latlon[:,0]
        lon_arr = latlon[:,1]
        x_poly = self.poly.fit_transform(x_arr)
        y_poly = self.poly.fit_transform(y_arr)
        self.reg_lat = LinearRegression().fit(x_poly, lat_arr)
        self.reg_lon = LinearRegression().fit(y_poly, lon_arr)

        def xy_to_latlon(x, y):
            lat = self.reg_lat.predict(self.poly.transform(np.array([[x]])))[0]
            lon = self.reg_lon.predict(self.poly.transform(np.array([[y]])))[0]
            return lat, lon

        # 以第一個點為中心
        m = folium.Map(location=[latlon[0,0], latlon[0,1]], zoom_start=13)

        # 在地圖上標示POI
        poi_df = pd.read_csv(f'./Stability/{output_name}_top_poi.csv')
        for idx, row in poi_df.iterrows():
            lat, lon = xy_to_latlon(row['poi_x'], row['poi_y'])
            folium.CircleMarker(location=[lat, lon], radius=5, color='blue').add_to(m)

        m.save(f'./Stability/{output_name}_top_poi_map.html')

        # 輸出200*200所有x,y對應的經緯度
        grid = []
        for x in range(1, 201):
            for y in range(1, 201):
                lat = self.reg_lat.predict(self.poly.transform(np.array([[x]])))[0]
                lon = self.reg_lon.predict(self.poly.transform(np.array([[y]])))[0]
                grid.append({'x': x, 'y': y, 'lat': lat, 'lon': lon})
        grid_df = pd.DataFrame(grid)
        grid_df.to_csv(f'./Stability/{output_name}_xy_grid_latlon.csv', index=False)
        print(f"已輸出 ./Stability/{output_name}_xy_grid_latlon.csv (200*200全座標)")   
        return grid_df

    def mark_POI(self, grid_df, output_name='Cityname', output_img=False):
        """
        各poi對應的x,y會放在OSM中
        """
        # 1. 讀取x,y對應經緯度資料
        if isinstance(grid_df, str):
            grid_df = pd.read_csv(grid_df)
        elif not isinstance(grid_df, pd.DataFrame):
            raise ValueError("只能接受DataFrame或資料路徑字串（csv檔）")
            
        # 2. 用bounding box定義地圖範圍
        os.makedirs('./OSM', exist_ok=True)
        min_lat = grid_df['lat'].min()
        max_lat = grid_df['lat'].max()
        min_lon = grid_df['lon'].min()
        max_lon = grid_df['lon'].max()
        bbox = [min_lon, min_lat, max_lon, max_lat] # bounding box: [min_lon, min_lat, max_lon, max_lat]
        overpass_url = "http://overpass-api.de/api/interpreter"

        # 3. 經緯度轉 x, y函式
        def latlon_to_xy(lat, lon):
            # 反向用已擬合回歸模型
            # 由於是單變數回歸，這裡用最近格點法
            # 直接用 x = argmin |lat_pred(x) - lat|, y = argmin |lon_pred(y) - lon|
            x_grid = np.arange(1, 201)
            y_grid = np.arange(1, 201)
            lat_pred = self.reg_lat.predict(self.poly.transform(x_grid.reshape(-1,1)))
            lon_pred = self.reg_lon.predict(self.poly.transform(y_grid.reshape(-1,1)))
            x = x_grid[np.argmin(np.abs(lat_pred - lat))]
            y = y_grid[np.argmin(np.abs(lon_pred - lon))]
            return x, y
        
        # 4. 查詢 OSM 資料，並轉換儲存
        # 地鐵站
        query_station = f"""
        [out:json][timeout:60];
        node[railway=station][station=subway]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        out body;
        """
        response = requests.post(overpass_url, data={'data': query_station})
        data = response.json()
        stations = []
        for el in data['elements']:
            name = el['tags'].get('name','')
            lat = el['lat']
            lon = el['lon']
            x, y = latlon_to_xy(lat, lon)
            stations.append({'name': name, 'x': x, 'y': y})
        stations_df = pd.DataFrame(stations)
        stations_df.to_csv(f'./OSM/{output_name}_subway_stations.csv', index=False)
        print(f'已儲存地鐵站至 ./OSM/{output_name}_subway_stations.csv')

        # 地鐵線
        query_subway_lines = f"""
        [out:json][timeout:60];
        relation["route"="subway"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        out body;
        >;
        out skel qt;
        """
        response = requests.post(overpass_url, data={'data': query_subway_lines})
        data = response.json()
        nodes = {el['id']: (el['lat'], el['lon']) for el in data['elements'] if el['type'] == 'node'}
        subway_lines = []
        for el in data['elements']:
            if el['type'] == 'relation' and el['tags'].get('route') == 'subway':
                line_name = el['tags'].get('name', '')
                # 取得所有 member node id
                node_ids = [m['ref'] for m in el['members'] if m['type'] == 'node']
                # 轉成經緯度
                coords = [nodes[nid] for nid in node_ids if nid in nodes]
                # 轉成 x, y
                xy_coords = [latlon_to_xy(lat, lon) for lat, lon in coords]
                for idx, (x, y) in enumerate(xy_coords):
                    subway_lines.append({'line_name': line_name, 'order': idx, 'x': x, 'y': y})
        subway_lines_df = pd.DataFrame(subway_lines)
        subway_lines_df.to_csv(f'./OSM/{output_name}_subway_lines.csv', index=False)
        print(f'已儲存地鐵線至 ./OSM/{output_name}_subway_lines.csv')

        # 火車站
        query_train_station = f"""
        [out:json][timeout:60];
        node["railway"="station"]["station"!="subway"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        out body;
        """
        response = requests.post(overpass_url, data={'data': query_train_station})
        data = response.json()
        train_stations = []
        for el in data['elements']:
            if el['type'] == 'node' and el['tags'].get('railway') == 'station' and el['tags'].get('station') != 'subway':
                name = el['tags'].get('name','')
                lat = el['lat']
                lon = el['lon']
                x, y = latlon_to_xy(lat, lon)
                train_stations.append({'name': name, 'x': x, 'y': y})
        train_stations_df = pd.DataFrame(train_stations)
        train_stations_df.to_csv(f'./OSM/{output_name}_train_stations.csv', index=False)
        print(f'已儲存火車站至 ./OSM/{output_name}_train_stations.csv')

        # 大學
        query_universities = f"""
        [out:json][timeout:60];
        (
        node["amenity"="university"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        );
        out body;
        >;
        out skel qt;
        """
        response = requests.post(overpass_url, data={'data': query_universities})
        data = response.json()
        nodes = {el['id']: (el['lat'], el['lon']) for el in data['elements'] if el['type'] == 'node'}
        universities = []
        for el in data['elements']:
            if el['type'] == 'node' and el['tags'].get('amenity') == 'university':
                name = el['tags'].get('name','')
                lat = el['lat']
                lon = el['lon']
                x, y = latlon_to_xy(lat, lon)
                universities.append({'name': name, 'x': x, 'y': y})
        universities_df = pd.DataFrame(universities)
        universities_df.to_csv(f'./OSM/{output_name}_universities.csv', index=False)
        print(f'已儲存大學至 ./OSM/{output_name}_universities.csv')

        # 國高中
        query_high_schools = f"""
        [out:json][timeout:60];
        (
        node["amenity"="school"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        );
        out body;
        >;
        out skel qt;
        """
        response = requests.post(overpass_url, data={'data': query_high_schools})
        data = response.json()
        nodes = {el['id']: (el['lat'], el['lon']) for el in data['elements'] if el['type'] == 'node'}
        high_schools = []
        for el in data['elements']:
            if el['type'] == 'node' and el['tags'].get('amenity') == 'school':
                name = el['tags'].get('name','')
                if ('高等学校' in name) or ('中学校' in name):
                    lat = el['lat']
                    lon = el['lon']
                    x, y = latlon_to_xy(lat, lon)
                    high_schools.append({'name': name, 'x': x, 'y': y})
        high_schools_df = pd.DataFrame(high_schools)
        high_schools_df.to_csv(f'./OSM/{output_name}_high_schools.csv', index=False)
        print(f'已儲存國高中至 ./OSM/{output_name}_high_schools.csv')

        # 大型購物中心或百貨公司
        query_shopping_centers = f"""
        [out:json][timeout:60];
        (
        node["shop"="mall"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        node["shop"="department_store"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        );
        out body;
        >;
        out skel qt;
        """
        response = requests.post(overpass_url, data={'data': query_shopping_centers})
        data = response.json()
        nodes = {el['id']: (el['lat'], el['lon']) for el in data['elements'] if el['type'] == 'node'}
        shopping_centers = []
        for el in data['elements']:
            if el['type'] == 'node' and (el['tags'].get('shop') == 'mall' or el['tags'].get('shop') == 'department_store'):
                name = el['tags'].get('name','')
                lat = el['lat']
                lon = el['lon']
                x, y = latlon_to_xy(lat, lon)
                shopping_centers.append({'name': name, 'x': x, 'y': y})
        shopping_centers_df = pd.DataFrame(shopping_centers)
        shopping_centers_df.to_csv(f'./OSM/{output_name}_shopping_centers.csv', index=False)
        print(f'已儲存購物中心至 ./OSM/{output_name}_shopping_centers.csv')

        # 辦公區
        query_office_areas = f"""
        [out:json][timeout:60];
        (
        node["office"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        );
        out body;
        >;
        out skel qt;
        """
        response = requests.post(overpass_url, data={'data': query_office_areas})
        data = response.json()
        nodes = {el['id']: (el['lat'], el['lon']) for el in data['elements'] if el['type'] == 'node'}
        office_areas = []
        for el in data['elements']:
            if el['type'] == 'node' and el['tags'].get('office'):
                name = el['tags'].get('name','')
                lat = el['lat']
                lon = el['lon']
                x, y = latlon_to_xy(lat, lon)
                office_areas.append({'name': name, 'x': x, 'y': y})
        office_areas_df = pd.DataFrame(office_areas)
        office_areas_df.to_csv(f'./OSM/{output_name}_office_areas.csv', index=False)

        # 公園
        query_parks = f"""
        [out:json][timeout:60];
        node["leisure"="park"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        out body;
        """
        response = requests.post(overpass_url, data={'data': query_parks})
        data = response.json()
        nodes = {el['id']: (el['lat'], el['lon']) for el in data['elements'] if el['type'] == 'node'}
        parks = []
        for el in data['elements']:
            if el['type'] == 'node' and el['tags'].get('leisure') == 'park':
                name = el['tags'].get('name','')
                lat = el['lat']
                lon = el['lon']
                x, y = latlon_to_xy(lat, lon)
                parks.append({'name': name, 'x': x, 'y': y})
        parks_df = pd.DataFrame(parks)
        parks_df.to_csv(f'./OSM/{output_name}_parks.csv', index=False)
        print(f'已儲存公園至 ./OSM/{output_name}_parks.csv')

        # 酒吧或居酒屋
        query_bars = f"""
        [out:json][timeout:60];
        (
        node["amenity"="pub"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        node["amenity"="bar"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        node["leisure"="bar"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        );
        out body;
        """
        response = requests.post(overpass_url, data={'data': query_bars})
        data = response.json()
        nodes = {el['id']: (el['lat'], el['lon']) for el in data['elements'] if el['type'] == 'node'}
        bars = []
        for el in data['elements']:
            if el['type'] == 'node' and el['tags'].get('amenity') in ['pub', 'bar']:
                name = el['tags'].get('name','')
                lat = el['lat']
                lon = el['lon']
                x, y = latlon_to_xy(lat, lon)
                bars.append({'name': name, 'x': x, 'y': y})
        bars_df = pd.DataFrame(bars)
        bars_df.to_csv(f'./OSM/{output_name}_bars.csv', index=False)
        print(f'已儲存酒吧至 ./OSM/{output_name}_bars.csv')

        # 高速公路
        query_freeways = f"""
        [out:json][timeout:60];
        way["highway"~"motorway"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        (._;>;);
        out body;
        """
        response = requests.post(overpass_url, data={'data': query_freeways})
        data = response.json()
        nodes = {el['id']: (el['lat'], el['lon']) for el in data['elements'] if el['type'] == 'node'}
        freeways_roads = []
        for el in data['elements']:
            if el['type'] == 'way' and 'highway' in el['tags']:
                road_name = el['tags'].get('name', '')
                node_ids = el['nodes']
                coords = [nodes[nid] for nid in node_ids if nid in nodes]
                xy_coords = [latlon_to_xy(lat, lon) for lat, lon in coords]
                # Convert np.int64 to int for all coordinates
                xy_coords_clean = [(int(x), int(y)) for x, y in xy_coords]
                freeways_roads.append({'road_name': road_name, 'xy_coords': xy_coords_clean})
        freeways_roads = pd.DataFrame(freeways_roads)
        freeways_roads.to_csv(f'./OSM/{output_name}_motorways.csv', index=False)
        print(f'已儲存高速公路至 ./OSM/{output_name}_motorways.csv')

        # 主幹道
        query_main_roads = f"""
        [out:json][timeout:60];
        way["highway"~"trunk|primary"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        (._;>;);
        out body;
        """
        response = requests.post(overpass_url, data={'data': query_main_roads})
        data = response.json()
        nodes = {el['id']: (el['lat'], el['lon']) for el in data['elements'] if el['type'] == 'node'}
        main_roads = []
        for el in data['elements']:
            if el['type'] == 'way' and 'highway' in el['tags']:
                road_name = el['tags'].get('name', '')
                node_ids = el['nodes']
                coords = [nodes[nid] for nid in node_ids if nid in nodes]
                xy_coords = [latlon_to_xy(lat, lon) for lat, lon in coords]
                # Convert np.int64 to int for all coordinates
                xy_coords_clean = [(int(x), int(y)) for x, y in xy_coords]
                main_roads.append({'road_name': road_name, 'xy_coords': xy_coords_clean})
        main_roads = pd.DataFrame(main_roads)
        main_roads.to_csv(f'./OSM/{output_name}_main_roads.csv', index=False)
        print(f'已儲存主幹道至 ./OSM/{output_name}_main_roads.csv')

        # 5. 畫在 200x200 地圖上
        if output_img:
            plt.figure(figsize=(15,15))
            # 畫地鐵站
            plt.scatter(stations_df['x'], stations_df['y'], c='red', s=11, label='Subway Station', zorder=10, alpha=0.5, marker='^')
            # 畫地鐵線
            for line_name, group in subway_lines_df.groupby('line_name'):
                if not group.empty:
                    x_coords = group['x'].tolist()
                    y_coords = group['y'].tolist()
                    plt.plot(x_coords, y_coords, linewidth=2, alpha=0.7, zorder=9)
            # 畫火車站
            plt.scatter(train_stations_df['x'], train_stations_df['y'], c='magenta', s=5, label='Train Station', zorder=5, alpha=0.5, marker='v')
            # 畫大學
            plt.scatter(universities_df['x'], universities_df['y'], c='blue', s=5, label='University', zorder=5, alpha=0.5, marker='s')
            # 畫國高中
            plt.scatter(high_schools_df['x'], high_schools_df['y'], c='aqua', s=2, label='High School', zorder=3, alpha=0.3, marker='o')
            # 畫購物中心
            plt.scatter(shopping_centers_df['x'], shopping_centers_df['y'], c='purple', s=5, label='Shopping Center', zorder=3, alpha=0.7, marker='*')
            # 畫辦公區
            plt.scatter(office_areas_df['x'], office_areas_df['y'], c='orange', s=3, label='Office Area', zorder=3, alpha=0.3, marker='D')
            # 畫公園
            plt.scatter(parks_df['x'], parks_df['y'], c='darkgreen', s=3, label='Park', zorder=2, alpha=0.2, marker='p')
            # 畫酒吧
            plt.scatter(bars_df['x'], bars_df['y'], c='brown', s=3, label='Bar', zorder=2, alpha=0.2, marker='4')
            # 畫高速公路
            for _, road in freeways_roads.iterrows():
                xy_coords = ast.literal_eval(str(road['xy_coords'])) if isinstance(road['xy_coords'], str) else road['xy_coords']
                x_coords = [x for x, y in xy_coords]
                y_coords = [y for x, y in xy_coords]
                plt.plot(x_coords, y_coords, color='turquoise', zorder=2, linewidth=1.5, alpha=0.5,label='Freeway' if 'Freeway' not in plt.gca().get_legend_handles_labels()[1] else "")
            # 畫主幹道 trunk/primary
            for _, road in main_roads.iterrows():
                xy_coords = ast.literal_eval(str(road['xy_coords'])) if isinstance(road['xy_coords'], str) else road['xy_coords']
                x_coords = [x for x, y in xy_coords]
                y_coords = [y for x, y in xy_coords]
                plt.plot(x_coords, y_coords, color='khaki', zorder=1, linewidth=1, alpha=0.3, label='Trunk/Primary' if 'Trunk/Primary' not in plt.gca().get_legend_handles_labels()[1] else "")

            # 6. 輸出圖
            plt.xlim(1, 200)
            plt.ylim(1, 200)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'{output_name} POI Map')
            plt.gca().invert_yaxis()
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.gca().xaxis.set_major_locator(MultipleLocator(10))
            plt.gca().yaxis.set_major_locator(MultipleLocator(10))
            plt.tight_layout()
            os.makedirs('./Animations', exist_ok=True)
            plt.savefig(f'./Animations/{output_name}_poi_map.png', dpi=200)
            print(f'已輸出 ./Animations/{output_name}_poi_map.png')

    def uid_trajectory_animation_with_POI(self, train_df, uid, output_name='Cityname', end_day=10):
            user_df = train_df[(train_df['uid'] == uid ) & (train_df['d'] <= end_day)]
            ani_fig, ani_ax = plt.subplots(figsize=(15,15))
            ani_ax.set_xlim(1, 200)
            ani_ax.set_ylim(1, 200) 
            ani_ax.invert_yaxis()
            ani_ax.grid(True, alpha=0.3)
            ani_ax.xaxis.set_major_locator(MultipleLocator(10))
            ani_ax.yaxis.set_major_locator(MultipleLocator(10))

            # 讀取OSM資料
            stations_df = pd.read_csv(f'./OSM/{output_name}_subway_stations.csv')
            subway_lines_df = pd.read_csv(f'./OSM/{output_name}_subway_lines.csv')
            train_stations_df = pd.read_csv(f'./OSM/{output_name}_train_stations.csv')
            freeways_roads = pd.read_csv(f'./OSM/{output_name}_motorways.csv')
            main_roads = pd.read_csv(f'./OSM/{output_name}_main_roads.csv')
            universities_df = pd.read_csv(f'./OSM/{output_name}_universities.csv')
            office_areas_df = pd.read_csv(f'./OSM/{output_name}_office_areas.csv')

            # --- 靜態底圖繪製 ---
            ani_ax.set_xlim(1, 200)
            ani_ax.set_ylim(1, 200)
            ani_ax.invert_yaxis()
            ani_ax.grid(True, alpha=0.3)
            ani_ax.xaxis.set_major_locator(MultipleLocator(10))
            ani_ax.yaxis.set_major_locator(MultipleLocator(10))
            ani_ax.tick_params(axis='x', labelsize=10)
            ani_ax.tick_params(axis='y', labelsize=10)
            # 地鐵站
            ani_ax.scatter(stations_df['x'], stations_df['y'], c='red', s=11, label='Subway Station', zorder=10, alpha=0.5, marker='^')
            # 地鐵線
            for _, group in subway_lines_df.groupby('line_name'):
                if not group.empty:
                    x_coords = group['x'].to_numpy()
                    y_coords = group['y'].to_numpy()
                    ani_ax.plot(x_coords, y_coords, linewidth=2, alpha=0.7, zorder=9)
            # 火車站
            ani_ax.scatter(train_stations_df['x'], train_stations_df['y'], c='magenta', s=3, label='Train Station', zorder=5, alpha=0.4, marker='v')
            # 畫大學
            ani_ax.scatter(universities_df['x'], universities_df['y'], c='blue', s=5, label='University', zorder=5, alpha=0.5, marker='s')
            # 畫辦公區
            ani_ax.scatter(office_areas_df['x'], office_areas_df['y'], c='orange', s=3, label='Office Area', zorder=3, alpha=0.2, marker='D')
            # 高速公路
            for _, road in freeways_roads.iterrows():
                try:
                    xy_coords = ast.literal_eval(str(road['xy_coords'])) if isinstance(road['xy_coords'], str) else road['xy_coords']
                    xy_coords_np = np.array(xy_coords)
                    x_coords = xy_coords_np[:,0]
                    y_coords = xy_coords_np[:,1]
                    ani_ax.plot(x_coords, y_coords, color='turquoise', zorder=2, linewidth=1.5, alpha=0.5, label='Freeway' if 'Freeway' not in ani_ax.get_legend_handles_labels()[1] else "")
                except Exception as e:
                    print(f"Skipping freeways_roads['xy_coords']: {road['xy_coords']} (Error: {e})")
            # 畫主幹道 trunk/primary
            for _, road in main_roads.iterrows():
                try:
                    xy_coords = ast.literal_eval(str(road['xy_coords'])) if isinstance(road['xy_coords'], str) else road['xy_coords']
                    xy_coords_np = np.array(xy_coords)
                    x_coords = xy_coords_np[:,0]
                    y_coords = xy_coords_np[:,1]
                    ani_ax.plot(x_coords, y_coords, color='khaki', zorder=1, linewidth=1, alpha=0.3, label='Trunk/Primary' if 'Trunk/Primary' not in plt.gca().get_legend_handles_labels()[1] else "")
                except Exception as e:
                    print(f"Skipping main_roads['xy_coords']: {road['xy_coords']} (Error: {e})")
    

            # --- 動畫更新只畫使用者軌跡 ---
            def init():
                pass  # 不清空底圖

            def update(i):
                # 只移除前一幀的使用者軌跡
                if hasattr(update, 'user_scatters'):
                    for sc in update.user_scatters:
                        sc.remove()
                update.user_scatters = []
                window = 3
                for j in range(max(0, i - window + 1), i + 1):
                    alpha = 0.1 + 0.9 * (j - max(0, i - window + 1)) / window
                    sc = ani_ax.scatter(user_df.iloc[j]['x'], user_df.iloc[j]['y'], alpha=alpha, s=30, c='k', marker='*', zorder=20)
                    update.user_scatters.append(sc)
                ani_ax.set_title(f"uid={uid} 第{user_df.iloc[i]['d']}天 {user_df.iloc[i]['t']*0.5}點鐘 ", fontsize=18)
                print(f"單人分時軌跡動畫進度: {i+1}/{user_df.shape[0]}", end='\r')

            ani = anime.FuncAnimation(ani_fig, update, frames=user_df.shape[0], init_func=init, repeat=True)
            start_day = user_df['d'].min()
            end_day = user_df['d'].max()
            ani.save(f'./Animations/{output_name}_uid{uid}_days{start_day}~{end_day}_trajectory_animation_withPOI.gif', fps=2, writer='pillow', dpi=150)
            plt.title(f"UID:{uid} 的軌跡動畫")

    def mode_proportion(self, train_df, std_df, output_name='Cityname', thresholds=[0, 9999], early_stop=3000):
        """
        計算每個uid各個時間下眾數的比例
        """
        for i in range(len(thresholds) - 1):
            lower = thresholds[i]
            upper = thresholds[i + 1]

            # 1、過濾出此區間的人
            filter_std_df = std_df[(std_df['x_std_mean'] >= lower) | (std_df['y_std_mean'] >= lower)]
            valid_uid_list = filter_std_df[(filter_std_df['x_std_mean'] < upper) & (filter_std_df['y_std_mean'] < upper)]['uid'].unique()
            if len(valid_uid_list) > early_stop:
                valid_uid_list = valid_uid_list[:early_stop]
            print(f"x|y std >= {lower},x&y std < {upper} 採用的使用者ID數量: {len(valid_uid_list)}")

            # 2、計算此區間每個人的眾數比例
            result = []
            times = np.arange(0, 48)  
            for idx, uid in enumerate(valid_uid_list):
                user_df = train_df[train_df['uid'] == uid]
                for time in times:
                    time_df = user_df[user_df['t'] == time]
                    if not time_df.empty:
                        mode_value = time_df['x'].mode().values[0]
                        x_mode_count = (time_df['x'] == mode_value).sum()
                        x_total_count = time_df['x'].count()
                        proportion = x_mode_count / x_total_count
                        result.append({'uid': uid, 'time': time, 'proportion': proportion})
                    else:
                        result.append({'uid': uid, 'time': time, 'proportion': None})
                print(f"計算進度: {idx + 1}/{len(valid_uid_list)}", end='\r')
            result_df = pd.DataFrame(result)
            result_df.to_csv(f'./Stability/{output_name}_mode_proportion_{lower}_{upper}.csv', index=False)
            print(f"已儲存眾數比例結果至 ./Stability/{output_name}_mode_proportion_{lower}_{upper}.csv")
            print(f"眾數比例計算完成: {lower} <= x|y std < {upper}, 眾數平均佔有比為: {result_df['proportion'].mean()}")

"""
測試程式碼
"""
if __name__ == "__main__":
    # # 計算眾數比例
    # # Adp = DataPreprocessor()
    # # train_df = pd.read_csv('./Training_Testing_Data/A_x_train.csv')
    # # std_df = pd.read_csv('./Stability/A_xtrain_working_day_stability.csv')
    # thresholds = [0, 1, 2, 3, 4, 5, 10, 9999]  # x|y std 的閾值
    # # Adp.mode_proportion(train_df=train_df,
    # #                     std_df=std_df,
    # #                     output_name='A_x',
    # #                     thresholds=thresholds,
    # #                     early_stop=3000)
    # fig, ax = plt.subplots(figsize=(10, 10))
    # for i in range(len(thresholds) - 1):
    #     lower = thresholds[i]
    #     upper = thresholds[i + 1]
    #     proportion_df = pd.read_csv(f'./Stability/A_x_mode_proportion_{lower}_{upper}.csv')
    #     mean_proportion = proportion_df.groupby('time')['proportion'].mean().reset_index()
    #     ax.plot(mean_proportion['time'], mean_proportion['proportion'], label=f'{lower} <= x|y std < {upper}', marker='o')
    # ax.set_xlim(0, 47)
    # ax.xaxis.set_major_locator(MultipleLocator(2))
    # plt.xlabel('Time (half-hour index)')
    # plt.ylabel('Average Mode Proportion')
    # plt.title('Average Mode Proportion by Time')    
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.show()


    # #測試POI對齊
    # Adp = DataPreprocessor()
    # poi_xy_arr=np.array([[135,77], [129,81], [135,82], [139,88],[53,177],[94,24],[113,38],[168,124]])
    # poi_latlon_arr = np.array([
    #     [35.171686742452046, 136.8819338645865],
    #     [35.136656631816585, 136.9096687302056],
    #     [35.17630485304206, 136.91512169532413],
    #     [35.18615646726879, 136.9473140549638],
    #     [34.758689289657525, 137.4246461958176],
    #     [34.96953043074047, 136.61705699934433],
    #     [35.05893050719316, 136.67898179306147],
    #     [35.329354846749176, 137.13783025649616],
    # ])

    # grid_df = Adp.map_alignment(poi_xy_arr=poi_xy_arr,
    #                   poi_latlon_arr=poi_latlon_arr,
    #                     output_name='A')
    # Adp.mark_POI(grid_df=grid_df,
    #              output_name='A',
    #              output_img=True)
    # train_df = pd.read_csv('./Training_Testing_Data/A_x_train.csv') 
    # Adp.uid_trajectory_animation_with_POI(train_df, 
    #                                       uid=24,
    #                                       output_name='A',
    #                                       end_day=7)
