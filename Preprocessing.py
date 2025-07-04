import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataVisualize_v2 import DataVisualizer as dv

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

    def sin_encode(self, df):
        """
        將時間欄位進行正弦編碼，將時間轉換為週期性特徵。
        """
        df['time_in_day_normalize'] = df['t'] / 48  # 正規化為 0~1
        df['time_sin'] = np.sin(2 * np.pi * df['time_in_day_normalize'])

        return df

    def stability_analysis(self, df, city_name, dataset_prefix):
        os.makedirs('./Stability', exist_ok=True)

        # 計算每個uid在每個時間段的x, y標準差
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

        working_day_std_df.to_csv(f'./Stability/{city_name}_{dataset_prefix}train_working_day_stability.csv', index=False)
        print(f"工作日標準差資料已儲存至 ./Stability/{city_name}_{dataset_prefix}train_working_day_stability.csv")

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

        non_working_day_std_df.to_csv(f'./Stability/{city_name}_{dataset_prefix}train_non_working_day_stability.csv', index=False)
        print(f"非工作日標準差資料已儲存至 ./Stability/{city_name}_{dataset_prefix}train_non_working_day_stability.csv")

        return working_day_std_df, non_working_day_std_df


"""
測試程式碼
"""
if __name__ == "__main__":
    # test_city_name = 'D'
    # DataLoader = DataPreprocessor(city_name=test_city_name, data_input=f'./Data./city_{test_city_name}_challengedata.csv')

    # x_train_df,_,y_train_df,_ = DataLoader.get_training_testing_data()
    # _, _=DataLoader.stability_analysis(x_train_df, city_name=test_city_name,dataset_prefix='x')
    # _, _=DataLoader.stability_analysis(y_train_df, city_name=test_city_name,dataset_prefix='y')
    # print(x_train_df.head())

    visual_tool = dv(data_input='./Training_Testing_Data/A_x_train.csv')
    visual_tool.single_user_trajectory(uid=3)
    # visual_tool.single_user_trajectory_animation(uid=3, fps=4, output_each_frame=True)
    # visual_tool.single_user_trajectory_animation(uid=35, fps=4, output_each_frame=False)
    # visual_tool.single_user_trajectory_animation(uid=6, fps=4, output_each_frame=False)
    # visual_tool.single_user_trajectory_animation(uid=283, fps=4, output_each_frame=False)
    # visual_tool.single_user_trajectory_animation(uid=704, fps=4, output_each_frame=False)
    # visual_tool.single_user_trajectory_animation(uid=14, fps=4, output_each_frame=False)
    plt.show()
    
    # 計算標準差平均值
    # std_df = pd.read_csv('./Stability/A_xtrain_working_day_stability.csv')
    # std_df['x_std'] = std_df['x_std'].replace(0, np.nan)
    # std_df['y_std'] = std_df['y_std'].replace(0, np.nan)
    # uid_std_mean = std_df.groupby('uid')[['x_std', 'y_std']].mean().reset_index()
    # uid_std_mean['x_std'] = uid_std_mean['x_std'].round(1)
    # uid_std_mean['y_std'] = uid_std_mean['y_std'].round(1)
    # uid_std_mean.to_csv('./A_xtrain_working_day_std_mean.csv', index=False)
    # print(uid_std_mean.head())