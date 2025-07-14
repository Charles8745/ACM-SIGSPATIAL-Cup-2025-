import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fastdtw
from DataVisualize_v2 import DataVisualizer as dv
from fastdtw import fastdtw
import time

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

"""
測試程式碼
"""
if __name__ == "__main__":
    # test_city_name = 'D'
    # DataLoader = DataPreprocessor(city_name=test_city_name, data_input=f'./Data./city_{test_city_name}_challengedata.csv')

    # x_train_df,_,y_train_df,_ = DataLoader.get_training_testing_data()
    # _, _=DataLoader.stability_analysis_std(x_train_df, city_name=test_city_name,dataset_prefix='x')
    # _, _=DataLoader.stability_analysis_std(y_train_df, city_name=test_city_name,dataset_prefix='y')
    
    # DataLoader.stability_analysis_trajectories(df_path=f"./Training_Testing_Data/{test_city_name}_x_train.csv", 
    #                                            std_df_path=f"./Stability/{test_city_name}_xtrain_working_day_stability.csv",
    #                                            output_name=f"{test_city_name}_xtrain_working_day_dtw.csv",
    #                                            IsWorkingDay=True
    #                                            )

    # DataLoader.stability_analysis_trajectories(df_path=f"./Training_Testing_Data/{test_city_name}_x_train.csv", 
    #                                         std_df_path=f"./Stability/{test_city_name}_xtrain_non_working_day_stability.csv",
    #                                         output_name=f"{test_city_name}_xtrain_non_working_day_dtw.csv", 
    #                                         IsWorkingDay=False
    #                                         )

    # DataLoader.stability_analysis_trajectories(df_path=f"./Training_Testing_Data/{test_city_name}_y_train.csv", 
    #                                            std_df_path=f"./Stability/{test_city_name}_ytrain_working_day_stability.csv",
    #                                            output_name=f"{test_city_name}_ytrain_working_day_dtw.csv",
    #                                            IsWorkingDay=True
    #                                            )

    # DataLoader.stability_analysis_trajectories(df_path=f"./Training_Testing_Data/{test_city_name}_y_train.csv", 
    #                                         std_df_path=f"./Stability/{test_city_name}_ytrain_non_working_day_stability.csv",
    #                                         output_name=f"{test_city_name}_ytrain_non_working_day_dtw.csv",
    #                                         IsWorkingDay=False
    #                                         )

    # 可視化工具
    # visual_tool = dv(data_input='./Training_Testing_Data/A_x_train.csv')
    # visual_tool.single_user_trajectory(uid=33)
    # visual_tool.single_user_trajectory_animation(uid=33, fps=4, output_each_frame=True)
    # visual_tool.single_user_trajectory(uid=94)
    # visual_tool.single_user_trajectory_animation(uid=94, fps=4, output_each_frame=True)
    # visual_tool.single_user_trajectory(uid=630)
    # visual_tool.single_user_trajectory_animation(uid=630, fps=4, output_each_frame=True)
    # visual_tool.single_user_trajectory_animation(uid=35, fps=4, output_each_frame=False)
    # visual_tool.single_user_trajectory_animation(uid=6, fps=4, output_each_frame=False)
    # visual_tool.single_user_trajectory_animation(uid=283, fps=4, output_each_frame=False)
    # visual_tool.single_user_trajectory_animation(uid=704, fps=4, output_each_frame=False)
    # visual_tool.single_user_trajectory_animation(uid=14, fps=4, output_each_frame=False)
    # plt.show()
    
    # 計算標準差平均值
    # std_df = pd.read_csv('./Stability/A_xtrain_working_day_stability.csv')
    # std_df['x_std'] = std_df['x_std'].replace(0, np.nan)
    # std_df['y_std'] = std_df['y_std'].replace(0, np.nan)
    # uid_std_mean = std_df.groupby('uid')[['x_std', 'y_std']].mean().reset_index()
    # uid_std_mean['x_std'] = uid_std_mean['x_std'].round(1)
    # uid_std_mean['y_std'] = uid_std_mean['y_std'].round(1)
    # uid_std_mean.to_csv('./A_xtrain_working_day_std_mean.csv', index=False)
    # print(uid_std_mean.head())
    
    # 抓出dtw距離大於10的uid
    # dtw_df = pd.read_csv('./Stability/A_xtrain_working_day_dtw.csv')
    # dtw_df = dtw_df[(dtw_df['dtw_mean'] > 10) & (dtw_df['dtw_mean'] < 20)]['uid'].unique()
    # print(dtw_df[:10])

    # # 統計資料 std
    # df_x = pd.read_csv('./Stability/D_xtrain_non_working_day_stability.csv')
    # df_y = pd.read_csv('./Stability/D_ytrain_non_working_day_stability.csv')
    # df = pd.concat([df_x, df_y], axis=0)

    # # 找出 raw_df 比 df 多的 uid
    # raw_df = pd.read_csv('./Data/city_D_challengedata.csv')
    # raw_uids = set(raw_df['uid'].unique())
    # df_uids = set(df['uid'].unique())
    # extra_uids = raw_uids - df_uids
    # print(raw_df['uid'].nunique(), "個uid")
    # print(df['uid'].nunique(), "個uid")
    # print("raw_df 比 df 多的 uid 數量:", len(extra_uids))
    # # print("raw_df 比 df 多的 uid:", sorted(list(extra_uids)))

    # 統計資料 std
    # std_interval = [5, 10, 20, 30]
    # for std in std_interval:
    #     count = df[(df['x_std_mean'] < std) & (df['y_std_mean'] < std)]['uid'].nunique()
    #     print(f"標準差小於{std}的uid數量: {count}, 佔比: {count/df['uid'].nunique():.2%}")

    # count = df[(df['x_std_mean'] >= std_interval[-1]) | (df['y_std_mean'] >= std_interval[-1])]['uid'].nunique()
    # print(f"標準差大於等於{std_interval[-1]}的uid數量: {count}, 佔比: {count/df['uid'].nunique():.2%}")

    # 統計資料 dtw
    # df_x = pd.read_csv('./Stability/A_xtrain_working_day_dtw.csv')
    # df_y = pd.read_csv('./Stability/A_ytrain_working_day_dtw.csv')
    # df = pd.concat([df_x, df_y], axis=0)
    # print(f"有{df['uid'].nunique()}個uid\n")
    # dtw_interval = [50, 100, 150, 200]

    # for dtw in dtw_interval:
    #     count = df[df['dtw_mean'] < dtw]['uid'].nunique()
    #     print(f"DTW小於{dtw}的uid數量: {count}, 佔比: {count/df['uid'].nunique():.2%}")
    
    # count = df[df['dtw_mean'] >= dtw_interval[-1]]['uid'].nunique()
    # print(f"DTW大於等於{dtw_interval[-1]}的uid數量: {count}, 佔比: {count/df['uid'].nunique():.2%}")

    # 比較CityC CityD前後差異是否很大
    # city_name = 'D'
    # train_working_day_std = pd.read_csv(f'./Stability/{city_name}_xtrain_working_day_stability.csv')
    # test_working_day_std = pd.read_csv(f'./Stability/{city_name}_xtest_working_day_stability.csv')

    # train_df = train_working_day_std.set_index(['uid', 't'])
    # test_df = test_working_day_std.set_index(['uid', 't'])

    # uids = test_working_day_std['uid'].unique()
    # hours = test_working_day_std['t'].unique()

    # diff_list = []
    # for i, uid in enumerate(uids):
    #     x_diff = []
    #     y_diff = []
    #     for hour in hours:
    #         try:
    #             train_x_mean = train_df.at[(uid, hour), 'x_mean']
    #             test_x_mean = test_df.at[(uid, hour), 'x_mean']
    #             train_y_mean = train_df.at[(uid, hour), 'y_mean']
    #             test_y_mean = test_df.at[(uid, hour), 'y_mean']
    #         except KeyError:
    #             continue

    #         if train_x_mean != 0 and test_x_mean != 0:
    #             x_diff.append(abs(test_x_mean - train_x_mean))
    #         if train_y_mean != 0 and test_y_mean != 0:
    #             y_diff.append(abs(test_y_mean - train_y_mean))
    #     diff_list.append({
    #         'uid': uid,
    #         'x_diff': np.mean(x_diff) if x_diff else np.nan,
    #         'y_diff': np.mean(y_diff) if y_diff else np.nan
    #     })
    #     print(f"處理進度: {i+1}/{len(uids)} (uid={uid})", end='\r')
    

    # diff_df = pd.DataFrame(diff_list)
    # diff_df.to_csv(f'./Stability/{city_name}_working_day_diff.csv', index=False)

    # x_diff_mean = diff_df['x_diff'].mean()
    # y_diff_mean = diff_df['y_diff'].mean()
    # print(f"\n{city_name}x_diff平均: {x_diff_mean:.2f}, y_diff平均: {y_diff_mean:.2f}")

    # # 用 seaborn 畫 boxplot
    # plt.figure(figsize=(4, 6))
    # sns.boxplot(data=diff_df[['x_diff', 'y_diff']])
    # sns.stripplot(data=diff_df[['x_diff', 'y_diff']], color='violet', jitter=0.2, size=1.2, alpha=0.3)
    

    # # 取得中位數
    # medians = diff_df[['x_diff', 'y_diff']].median()

    # # 在圖上標註中位數
    # for i, col in enumerate(['x_diff', 'y_diff']):
    #     plt.text(i, medians[col], f'Median={medians[col]:.2f}', ha='center', va='bottom', color='red', fontsize=10, fontweight='bold')

    # plt.title(f'{city_name} 1~60天和61~75天工作日差異')
    # plt.show()

    pass