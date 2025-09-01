import os
import re
import numpy as np
import pandas as pd
import validator_InModify as validator
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoLocator
import geobleu
import time
import random
import seaborn as sns
from collections import defaultdict, Counter
import ast  
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 或 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正確顯示負號
"""
1. 改成針對 (x, y) 組合來找眾數
2. 若(x, y) 組合在此時間段只出現低於閥值次，則改成平均
"""

class ModelZoo:
    def __init__(self, train_data_input, test_data_input):
        if isinstance(train_data_input, pd.DataFrame):
            self.train_data = train_data_input
            print(f"直接使用DataFrame資料，共有{self.train_data.shape[0]}筆資料\n",
                  f"train資料範圍: uid={self.train_data['uid'].min()}~{self.train_data['uid'].max()}\n",
                  f"train時間範圍: days={self.train_data['d'].min()}~{self.train_data['d'].max()}\n")
        elif isinstance(train_data_input, str):
            self.train_data = pd.read_csv(train_data_input, header=0, dtype=int)
            print(f"讀取資料成功，共有{self.train_data.shape[0]}筆資料\n",
                f"train資料範圍: uid={self.train_data['uid'].min()}~{self.train_data['uid'].max()}\n",
                f"train時間範圍: days={self.train_data['d'].min()}~{self.train_data['d'].max()}\n")
        else:
            raise ValueError("只能接受DataFrame或資料路徑字串（csv檔）。") 
        
        if isinstance(test_data_input, pd.DataFrame):
            self.test_data = test_data_input
            print(f"直接使用DataFrame資料，共有{self.test_data.shape[0]}筆資料\n",
                  f"test資料範圍: uid={self.test_data['uid'].min()}~{self.test_data['uid'].max()}\n",
                  f"test時間範圍: days={self.test_data['d'].min()}~{self.test_data['d'].max()}\n")
        elif isinstance(test_data_input, str):
            self.test_data = pd.read_csv(test_data_input, header=0, dtype=int)
            print(f"讀取資料成功，共有{self.test_data.shape[0]}筆資料\n",
                f"test資料範圍: uid={self.test_data['uid'].min()}~{self.test_data['uid'].max()}\n",
                f"test時間範圍: days={self.test_data['d'].min()}~{self.test_data['d'].max()}\n")
        else:
            raise ValueError("只能接受DataFrame或資料路徑字串（csv檔）。")        
        
    def Per_User_Per_t_Mode(self, valid_uid_list, output_name, early_stop=None):
        os.makedirs('./Predictions', exist_ok=True)
        os.makedirs('./ckpt', exist_ok=True)
        print(f'Per_User_Per_t_Mode: 使用者數量={len(valid_uid_list)}')

        if early_stop is not None and early_stop < len(valid_uid_list):
            valid_uid_list = sorted(random.sample(list(valid_uid_list), early_stop))
            print(f'隨機抽取 {early_stop} 個 uid 進行訓練/預測')

        start_time = time.time()
        # Train資料中每個使用者在每個時間點的x,y值的全域模式
        result = []
        for i, uid in enumerate(valid_uid_list):
            user_df = self.train_data[self.train_data['uid'] == uid]
            if user_df.empty:
                continue

            x_user_global_mode = user_df['x'].mode()
            y_user_global_mode = user_df['y'].mode()
            for t in np.arange(0, 48):
                t_df = user_df[user_df['t'] == t]
                if not t_df.empty:
                    x_mode = t_df['x'].mode().values[0]
                    y_mode = t_df['y'].mode().values[0]
                else:
                    x_mode = x_user_global_mode.values[0] if not x_user_global_mode.empty else 0 # 如果都沒有值，則設為0
                    y_mode = y_user_global_mode.values[0] if not y_user_global_mode.empty else 0
                result.append({'uid': uid, 't': t, 'x': x_mode, 'y': y_mode})
            print(f'訓練進度: {i+1}/{len(valid_uid_list)} 使用者ID={uid}', end='\r')

        train_mode_df = pd.DataFrame(result)
        train_mode_df.to_csv(f'./ckpt/{output_name}_per_user_per_t_mode.csv', index=False)
        print(f"Train_Mode: 結果已儲存至 ./ckpt/{output_name}_per_user_per_t_mode.csv")

        # 依據測試資料的時間點，將模式應用到測試資料
        before_non_uid =[]
        after_non_uid = []
        result = []
        for i, uid in enumerate(valid_uid_list):
            train_user_df = train_mode_df[train_mode_df['uid'] == uid]
            if train_user_df.empty:
                before_non_uid.append(uid)
                print(f'使用者ID={uid} 在訓練資料中沒有模式，跳過此使用者')
                continue
            test_user_df = self.test_data[self.test_data['uid'] == uid]
            if test_user_df.empty:
                after_non_uid.append(uid)
                print(f'使用者ID={uid} 在測試資料中沒有資料，跳過此使用者')
                continue

            days = np.sort(test_user_df['d'].unique())
            for day in days:
                hours = np.sort(test_user_df[test_user_df['d'] == day]['t'].unique())
                for hour in hours:
                    x_mode = train_user_df[train_user_df['t'] == hour]['x'].values[0]
                    y_mode = train_user_df[train_user_df['t'] == hour]['y'].values[0]
                    result.append({'uid': uid, 'd':day, 't': hour, 'x': x_mode, 'y': y_mode})
            print(f'預測進度: {i+1}/{len(valid_uid_list)} 使用者ID={uid}', end='\r')

        prediction_df = pd.DataFrame(result)
        prediction_df = prediction_df[['uid', 'd', 't', 'x', 'y']].astype(int)

        prediction_df.to_csv(f'./Predictions/{output_name}_per_user_per_t_mode.csv', index=False)
        print(f"Per_User_Per_t_Mode: 結果已儲存至 ./Predictions/{output_name}_per_user_per_t_mode.csv")

        elapsed_time = time.time() - start_time
        print(f"Per_User_Per_t_Mode: 執行時間: {elapsed_time//60:.2f}min")
        return prediction_df

    def Per_User_Per_t_Mode_working_day(self, valid_uid_list, output_name, early_stop=None):
        os.makedirs('./Predictions', exist_ok=True)
        os.makedirs('./ckpt', exist_ok=True)
        print(f'Per_User_Per_t_Mode: 使用者數量={len(valid_uid_list)}')

        if early_stop is not None and early_stop < len(valid_uid_list):
            valid_uid_list = sorted(random.sample(list(valid_uid_list), early_stop))
            print(f'隨機抽取 {early_stop} 個 uid 進行訓練/預測')

        start_time = time.time()
      # Train資料中每個使用者在每個時間點的x,y值的全域模式並且分成工作日和非工作日
        result = []
        for i, uid in enumerate(valid_uid_list):
            user_df = self.train_data[self.train_data['uid'] == uid]
            if user_df.empty: # 如果使用者在訓練資料中沒有資料，則跳過
                continue
            
            # working day 訓練
            user_df_working_day = user_df[user_df['working_day']==1]
            x_user_global_mode = user_df_working_day['x'].mode() # global用於沒有資料的情況補值
            y_user_global_mode = user_df_working_day['y'].mode()
            for t in np.arange(0, 48):
                t_df = user_df_working_day[user_df_working_day['t'] == t]
                if not t_df.empty:
                    x_mode = t_df['x'].mode().values[0]
                    y_mode = t_df['y'].mode().values[0]
                else:
                    x_mode = x_user_global_mode.values[0] if not x_user_global_mode.empty else 0 # 如果都沒有值，則設為0
                    y_mode = y_user_global_mode.values[0] if not y_user_global_mode.empty else 0
                result.append({'uid': uid, 't': t, 'x': x_mode, 'y': y_mode, 'working_day': 1})

            # non-working day 訓練
            user_df_non_working_day = user_df[user_df['working_day']==0]
            x_user_global_mode = user_df_non_working_day['x'].mode() # global用於沒有資料的情況補值
            y_user_global_mode = user_df_non_working_day['y'].mode()
            for t in np.arange(0, 48):
                t_df = user_df_non_working_day[user_df_non_working_day['t'] == t]
                if not t_df.empty:
                    x_mode = t_df['x'].mode().values[0]
                    y_mode = t_df['y'].mode().values[0]
                else:
                    if not x_user_global_mode.empty and not y_user_global_mode.empty:
                        x_mode = x_user_global_mode.values[0]
                        y_mode = y_user_global_mode.values[0] 
                    else:
                        x_mode_working = user_df_working_day[user_df_working_day['t'] == t]['x'].mode()
                        y_mode_working = user_df_working_day[user_df_working_day['t'] == t]['y'].mode()
                        x_mode = x_mode_working.values[0] if not x_mode_working.empty else user_df_working_day['x'].mode().values[0]
                        y_mode = y_mode_working.values[0] if not y_mode_working.empty else user_df_working_day['y'].mode().values[0]
                result.append({'uid': uid, 't': t, 'x': x_mode, 'y': y_mode, 'working_day': 0})

            print(f'訓練進度: {i+1}/{len(valid_uid_list)} 使用者ID={uid}', end='\r')

        train_mode_df = pd.DataFrame(result)
        train_mode_df.to_csv(f'./ckpt/{output_name}_Per_User_Per_t_Mode_working_day.csv', index=False)
        print(f"Train_Mode: 結果已儲存至 ./ckpt/{output_name}_Per_User_Per_t_Mode_working_day.csv")


        # 依據測試資料的時間點，將模式應用到測試資料
        before_non_uid =[]
        after_non_uid = []
        result = []
        for i, uid in enumerate(valid_uid_list):
            train_user_df = train_mode_df[train_mode_df['uid'] == uid]
            test_user_df = self.test_data[self.test_data['uid'] == uid]

            # 若之前之後沒有資料，則跳過
            if train_user_df.empty:
                before_non_uid.append(uid)
                print(f'使用者ID={uid} 在訓練資料中沒有模式，跳過此使用者')
                continue
            if test_user_df.empty:
                after_non_uid.append(uid)
                print(f'使用者ID={uid} 在測試資料中沒有資料，跳過此使用者')
                continue

            # 取得工作日和非工作日的模式
            days = np.sort(test_user_df['d'].unique())
            for day in days:
                hours = np.sort(test_user_df[test_user_df['d'] == day]['t'].unique())
                if test_user_df[test_user_df['d'] == day]['working_day'].values[0] == 1: # 工作日
                    for hour in hours:
                        x_mode = train_user_df[(train_user_df['t'] == hour) & (train_user_df['working_day'] == 1)]['x'].values[0]
                        y_mode = train_user_df[(train_user_df['t'] == hour) & (train_user_df['working_day'] == 1)]['y'].values[0]
                        result.append({'uid': uid, 'd':day, 't': hour, 'x': x_mode, 'y': y_mode})
                else: # 非工作日
                    for hour in hours:
                        x_mode = train_user_df[(train_user_df['t'] == hour) & (train_user_df['working_day'] == 0)]['x'].values[0]
                        y_mode = train_user_df[(train_user_df['t'] == hour) & (train_user_df['working_day'] == 0)]['y'].values[0]
                        result.append({'uid': uid, 'd':day, 't': hour, 'x': x_mode, 'y': y_mode})

            print(f'預測進度: {i+1}/{len(valid_uid_list)} 使用者ID={uid}', end='\r')

        prediction_df = pd.DataFrame(result)
        prediction_df = prediction_df[['uid', 'd', 't', 'x', 'y']].astype(int)

        prediction_df.to_csv(f'./Predictions/{output_name}_Per_User_Per_t_Mode_working_day.csv', index=False)
        print(f"Per_User_Per_t_Mode_working_day: 結果已儲存至 ./Predictions/{output_name}_Per_User_Per_t_Mode_working_day.csv")

        elapsed_time = time.time() - start_time
        print(before_non_uid)
        print(after_non_uid)
        print(f"Per_User_Per_t_Mode_working_day: 執行時間: {elapsed_time//60:.2f}min")
        return prediction_df

    def Per_User_Per_t_Mode_working_day_modify(self, feature_df, valid_uid_list, output_name, early_stop=None):
        """
        針對補值修改:
        1、 若在早上0點到6點及晚上8點到12點，則取home點
        2、 其他時間取上一個點的mode
        """
        os.makedirs('./Predictions', exist_ok=True)
        os.makedirs('./ckpt', exist_ok=True)
        print(f'Per_User_Per_t_Mode: 使用者數量={len(valid_uid_list)}')

        if early_stop is not None and early_stop < len(valid_uid_list):
            valid_uid_list = sorted(random.sample(list(valid_uid_list), early_stop))
            print(f'隨機抽取 {early_stop} 個 uid 進行訓練/預測')

        start_time = time.time()
        # Train資料中每個使用者在每個時間點的x,y值的全域模式並且分成工作日和非工作日
        result = []
        for i, uid in enumerate(valid_uid_list):
            user_df = self.train_data[self.train_data['uid'] == uid]
            if user_df.empty: # 如果使用者在訓練資料中沒有資料，則跳過
                continue

            # 當t<=12 or t >= 40補值邏輯
            if not pd.isnull(feature_df[feature_df['uid'] == uid]['home_x'].values[0]) :
                global_home_x = feature_df[feature_df['uid'] == uid]['home_x'].values[0] 
                global_home_y = feature_df[feature_df['uid'] == uid]['home_y'].values[0] 
            else:
                global_home_x = user_df['x'].mode().values[0]
                global_home_y = user_df['y'].mode().values[0]
            
            # working day 訓練
            repeat_threshold = 2
            user_df_working_day = user_df[user_df['working_day']==1]
            for t in np.arange(0, 48):
                t_df = user_df_working_day[user_df_working_day['t'] == t]
                if not t_df.empty:
                    xy_counts = t_df.groupby(['x', 'y']).size()
                    max_count = xy_counts.max()
                    if max_count < repeat_threshold:
                        # 所有組合都低於閥值，取平均
                        x_mode = int(t_df['x'].mean())
                        y_mode = int(t_df['y'].mean())
                    else:
                        xy_mode = xy_counts.idxmax()
                        x_mode, y_mode = xy_mode
                else: # 此t沒有資料
                    if t<=12 or t >= 40: # 若在早上0點到6點及晚上8點到12點，取home點
                        x_mode = global_home_x
                        y_mode = global_home_y
                    else: # 其他時間取上一個點的mode
                        x_mode = result[-1]['x']
                        y_mode = result[-1]['y']
                result.append({'uid': uid, 't': t, 'x': x_mode, 'y': y_mode, 'working_day': 1})

            # non-working day 訓練
            user_df_non_working_day = user_df[user_df['working_day']==0]
            for t in np.arange(0, 48):
                t_df = user_df_non_working_day[user_df_non_working_day['t'] == t]
                if not t_df.empty:
                    xy_counts = t_df.groupby(['x', 'y']).size()
                    max_count = xy_counts.max()
                    if max_count < repeat_threshold:
                        # 所有組合都低於閥值，取平均
                        x_mode = int(t_df['x'].mean())
                        y_mode = int(t_df['y'].mean())
                    else:
                        xy_mode = xy_counts.idxmax()
                        x_mode, y_mode = xy_mode
                else: # 此t沒有資料
                    if t<=12 or t >= 40: # 若在早上0點到6點及晚上8點到12點，取home點
                        x_mode = global_home_x
                        y_mode = global_home_y
                    else: # 其他時間取上一個點的mode
                        x_mode = result[-1]['x']
                        y_mode = result[-1]['y']
                result.append({'uid': uid, 't': t, 'x': x_mode, 'y': y_mode, 'working_day': 0})

            print(f'訓練進度: {i+1}/{len(valid_uid_list)} 使用者ID={uid}', end='\r')

        train_mode_df = pd.DataFrame(result)
        train_mode_df.to_csv(f'./ckpt/{output_name}_Per_User_Per_t_Mode_working_day_modify.csv', index=False)
        print(f"Train_Mode: 結果已儲存至 ./ckpt/{output_name}_Per_User_Per_t_Mode_working_day_modify.csv")


        # 依據測試資料的時間點，將模式應用到測試資料
        before_non_uid =[]
        after_non_uid = []
        result = []
        for i, uid in enumerate(valid_uid_list):
            train_user_df = train_mode_df[train_mode_df['uid'] == uid]
            test_user_df = self.test_data[self.test_data['uid'] == uid]

            # 若之前之後沒有資料，則跳過
            if train_user_df.empty:
                before_non_uid.append(uid)
                print(f'使用者ID={uid} 在訓練資料中沒有模式，跳過此使用者')
                continue
            if test_user_df.empty:
                after_non_uid.append(uid)
                print(f'使用者ID={uid} 在測試資料中沒有資料，跳過此使用者')
                continue

            # 取得工作日和非工作日的模式
            days = np.sort(test_user_df['d'].unique())
            for day in days:
                hours = np.sort(test_user_df[test_user_df['d'] == day]['t'].unique())
                if test_user_df[test_user_df['d'] == day]['working_day'].values[0] == 1: # 工作日
                    for hour in hours:
                        x_mode = train_user_df[(train_user_df['t'] == hour) & (train_user_df['working_day'] == 1)]['x'].values[0]
                        y_mode = train_user_df[(train_user_df['t'] == hour) & (train_user_df['working_day'] == 1)]['y'].values[0]
                        result.append({'uid': uid, 'd':day, 't': hour, 'x': x_mode, 'y': y_mode})
                else: # 非工作日
                    for hour in hours:
                        x_mode = train_user_df[(train_user_df['t'] == hour) & (train_user_df['working_day'] == 0)]['x'].values[0]
                        y_mode = train_user_df[(train_user_df['t'] == hour) & (train_user_df['working_day'] == 0)]['y'].values[0]
                        result.append({'uid': uid, 'd':day, 't': hour, 'x': x_mode, 'y': y_mode})

            print(f'預測進度: {i+1}/{len(valid_uid_list)} 使用者ID={uid}', end='\r')

        prediction_df = pd.DataFrame(result)
        prediction_df = prediction_df[['uid', 'd', 't', 'x', 'y']].astype(int)

        prediction_df.to_csv(f'./Predictions/{output_name}_Per_User_Per_t_Mode_working_day_modify.csv', index=False)
        print(f"Per_User_Per_t_Mode_working_day: 結果已儲存至 ./Predictions/{output_name}_Per_User_Per_t_Mode_working_day_modify.csv")

        elapsed_time = time.time() - start_time
        print(before_non_uid)
        print(after_non_uid)
        print(f"Per_User_Per_t_Mode_working_day_modify: 執行時間: {elapsed_time//60:.2f}min")
        return prediction_df

    def Per_User_Per_t_Mode_working_day_dynamic(self, feature_df, valid_uid_list, output_name, early_stop=None):
        """
        預測時候下班時間動態調整:
        若在上班日五點之後出現連續3個delta_t=1的情況，則將通勤路線平均分佈上去
        """
        os.makedirs('./Predictions', exist_ok=True)
        os.makedirs('./ckpt', exist_ok=True)
        print(f'Per_User_Per_t_Mode: 使用者數量={len(valid_uid_list)}')

        if early_stop is not None and early_stop < len(valid_uid_list):
            valid_uid_list = sorted(random.sample(list(valid_uid_list), early_stop))
            print(f'隨機抽取 {early_stop} 個 uid 進行訓練/預測')

        start_time = time.time()
        # Train資料中每個使用者在每個時間點的x,y值的全域模式並且分成工作日和非工作日
        result = []
        for i, uid in enumerate(valid_uid_list):
            user_df = self.train_data[self.train_data['uid'] == uid]
            if user_df.empty: # 如果使用者在訓練資料中沒有資料，則跳過
                continue

            # 當t<=12 or t >= 40補值邏輯
            if not pd.isnull(feature_df[feature_df['uid'] == uid]['home_x'].values[0]) :
                global_home_x = feature_df[feature_df['uid'] == uid]['home_x'].values[0] 
                global_home_y = feature_df[feature_df['uid'] == uid]['home_y'].values[0] 
            else:
                global_home_x = user_df['x'].mode().values[0]
                global_home_y = user_df['y'].mode().values[0]
            
            # working day 訓練
            user_df_working_day = user_df[user_df['working_day']==1]
            for t in np.arange(0, 48):
                t_df = user_df_working_day[user_df_working_day['t'] == t]
                if not t_df.empty:
                    x_mode = t_df['x'].mode().values[0]
                    y_mode = t_df['y'].mode().values[0]
                else: # 此t沒有資料
                    if t<=12 or t >= 40: # 若在早上0點到6點及晚上8點到12點，取home點
                        x_mode = global_home_x
                        y_mode = global_home_y
                    else: # 其他時間取上一個點的mode
                        x_mode = result[-1]['x']
                        y_mode = result[-1]['y']
                result.append({'uid': uid, 't': t, 'x': x_mode, 'y': y_mode, 'working_day': 1})

            # non-working day 訓練
            user_df_non_working_day = user_df[user_df['working_day']==0]
            for t in np.arange(0, 48):
                t_df = user_df_non_working_day[user_df_non_working_day['t'] == t]
                if not t_df.empty:
                    x_mode = t_df['x'].mode().values[0]
                    y_mode = t_df['y'].mode().values[0]
                else: # 此t沒有資料
                    if t<=12 or t >= 40: # 若在早上0點到6點及晚上8點到12點，取home點
                        x_mode = global_home_x
                        y_mode = global_home_y
                    else: # 其他時間取上一個點的mode
                        x_mode = result[-1]['x']
                        y_mode = result[-1]['y']
                result.append({'uid': uid, 't': t, 'x': x_mode, 'y': y_mode, 'working_day': 0})

            print(f'訓練進度: {i+1}/{len(valid_uid_list)} 使用者ID={uid}', end='\r')

        train_mode_df = pd.DataFrame(result)
        train_mode_df.to_csv(f'./ckpt/{output_name}_Per_User_Per_t_Mode_working_day_dynamic.csv', index=False)
        print(f"Train_Mode: 結果已儲存至 ./ckpt/{output_name}_Per_User_Per_t_Mode_working_day_dynamic.csv")

       # 依據測試資料的時間點，將模式應用到測試資料
        result = []
        for i, uid in enumerate(valid_uid_list):
            train_user_df = train_mode_df[train_mode_df['uid'] == uid]
            test_user_df = self.test_data[self.test_data['uid'] == uid]
            feature_user_df = feature_df[feature_df['uid'] == uid]

            # 若之前之後沒有資料，則跳過
            if train_user_df.empty:
                print(f'使用者ID={uid} 在訓練資料中沒有模式，跳過此使用者')
                continue
            if test_user_df.empty:
                print(f'使用者ID={uid} 在測試資料中沒有資料，跳過此使用者')
                continue

            # 取得工作日和非工作日的模式
            days = np.sort(test_user_df['d'].unique())
            for day in days:
                user_day_df = test_user_df[test_user_df['d'] == day]
                hours = np.sort(user_day_df['t'].unique())
                # 工作日預測
                # 如果下班時間在五點之後且有連續3個以上delta_t=1的情況，則將通勤路線平均分佈上去
                if user_day_df['working_day'].values[0] == 1: 
                    # 判斷哪些需要mask，只考慮t>=34的區段，並且只處理第一組111...就剪枝
                    after_5pm_df = user_day_df[user_day_df['t'] >= 34].copy()
                    mask = np.zeros(after_5pm_df.shape[0], dtype=bool)
                    count = 0
                    if not after_5pm_df.empty:
                        delta_t_arr = after_5pm_df['delta_t'].values
                        for i in range(len(delta_t_arr)):
                            if delta_t_arr[i] == 1 or delta_t_arr[i] == 2: # 只考慮delta_t=1或2的情況
                                count += 1
                                if count >= 3:
                                    mask[i-2:i+1] = True
                            else:
                                if count >= 3: # 第一組就剪枝
                                    break
                                else:
                                    count = 0
                    # 取得通勤路徑點，第一個點離家最近
                    commute_str = feature_user_df['commute_paths'].values[0]
                    if not isinstance(commute_str, str) or pd.isna(commute_str):
                        commute_points = []
                    else:
                        commute_points = re.findall(r'\((\d+),(\d+)\)', commute_str)
                        commute_points = [(int(x), int(y)) for x, y in commute_points]
                    commute_len = len(commute_points)
                    # 預測
                    total_conut = count 
                    mask_count = count
                    for idx, hour in enumerate(hours):
                        # 如果該時間點為下班之後且有mask則另外處理
                        if hour >= 34:
                            mask_flag = bool(mask[list(after_5pm_df['t'].values).index(hour)])
                            if mask_flag and commute_points: # 有mask的情況且通勤點值存在
                                if mask_count == 1: # 只剩一個點了，快到家了，那就commute_points中第一組點
                                    x_mode = commute_points[0][0]
                                    y_mode = commute_points[0][1]
                                else: # 平均分佈上去
                                    idx = int((mask_count/ total_conut) * (commute_len)) - 1
                                    x_mode = commute_points[idx][0]
                                    y_mode = commute_points[idx][1]
                                mask_count -= 1
                            else: # 正常情況下取模式
                                x_mode = train_user_df[(train_user_df['t'] == hour) & (train_user_df['working_day'] == 1)]['x'].values[0]
                                y_mode = train_user_df[(train_user_df['t'] == hour) & (train_user_df['working_day'] == 1)]['y'].values[0]
                        # 非下班時間
                        else:
                            x_mode = train_user_df[(train_user_df['t'] == hour) & (train_user_df['working_day'] == 1)]['x'].values[0]
                            y_mode = train_user_df[(train_user_df['t'] == hour) & (train_user_df['working_day'] == 1)]['y'].values[0]
                        result.append({'uid': uid, 'd': day, 't': hour, 'x': x_mode, 'y': y_mode})
                
                # 非工作日預測
                else:
                    for hour in hours:
                        x_mode = train_user_df[(train_user_df['t'] == hour) & (train_user_df['working_day'] == 0)]['x'].values[0]
                        y_mode = train_user_df[(train_user_df['t'] == hour) & (train_user_df['working_day'] == 0)]['y'].values[0]
                        result.append({'uid': uid, 'd':day, 't': hour, 'x': x_mode, 'y': y_mode})

            print(f'預測進度: {i+1}/{len(valid_uid_list)} 使用者ID={uid}', end='\r')
        prediction_df = pd.DataFrame(result)
        prediction_df = prediction_df[['uid', 'd', 't', 'x', 'y']].astype(int)

        prediction_df.to_csv(f'./Predictions/{output_name}_Per_User_Per_t_Mode_working_day_dynamic.csv', index=False)
        print(f"Per_User_Per_t_Mode_working_day_dynamic: 結果已儲存至 ./Predictions/{output_name}_Per_User_Per_t_Mode_working_day_dynamic.csv")

        elapsed_time = time.time() - start_time
        print(f"Per_User_Per_t_Mode_working_day_dynamic: 執行時間: {elapsed_time//60:.2f}min")
        return prediction_df

    def Per_User_Per_t_Mode_working_day_top_p(self, feature_df, valid_uid_list, output_name='CityName', early_stop=None, top_p=0.7):
        if len(valid_uid_list) > early_stop:
            valid_uid_list = valid_uid_list[:early_stop]
        
        # 1、訓練，先記錄各uid這個時間段下的超過top_p的眾數和比例，要注意補值方式有更動
        result = []
        for idx, uid in enumerate(valid_uid_list):
            user_df = self.train_data[self.train_data['uid'] == uid]
            # 當t<=12 or t >= 40補值邏輯
            if not pd.isnull(feature_df[feature_df['uid'] == uid]['home_x'].values[0]) :
                global_home_x = feature_df[feature_df['uid'] == uid]['home_x'].values[0] 
                global_home_y = feature_df[feature_df['uid'] == uid]['home_y'].values[0] 
            else:
                global_home_x = user_df['x'].mode().values[0]
                global_home_y = user_df['y'].mode().values[0]

            # working day 訓練
            user_df_working_day = user_df[user_df['working_day']==1]
            for t in np.arange(0, 48):
                t_df = user_df_working_day[user_df_working_day['t'] == t]
                xy_counts = t_df.groupby(['x', 'y']).size().reset_index(name='count')
                xy_counts = xy_counts.sort_values('count', ascending=False).reset_index(drop=True)
                total = xy_counts['count'].sum()
                temp_res = [] # 儲存此t下的top_p的眾數和比例
                if not t_df.empty: # 此t有資料
                    sum = 0
                    i = 0
                    while sum < top_p:
                        row = xy_counts.iloc[i]
                        x_mode = int(row['x'])
                        y_mode = int(row['y'])
                        ratio = round(row['count'] / total, 3)
                        sum += ratio
                        i += 1
                        temp_res.append([int(x_mode), int(y_mode), float(ratio)])

                else: # 此t沒有資料
                    if t<=12 or t >= 40: # 若在早上0點到6點及晚上8點到12點，取home點
                        temp_res.append([int(global_home_x), int(global_home_y), 1.0])
                    else: # 其他時間取上一個點的mode
                        temp_res = result[-1]['mode']
                result.append({'uid': uid, 't': t, 'working_day': 1, 'mode': temp_res})

            # non-working day 訓練
            user_df_non_working_day = user_df[user_df['working_day']==0]
            for t in np.arange(0, 48):
                t_df = user_df_non_working_day[user_df_non_working_day['t'] == t]
                xy_counts = t_df.groupby(['x', 'y']).size().reset_index(name='count')
                xy_counts = xy_counts.sort_values('count', ascending=False).reset_index(drop=True)
                total = xy_counts['count'].sum()
                temp_res = [] # 儲存此t下的top_p的眾數和比例
                if not t_df.empty: # 此t有資料
                    sum = 0
                    i = 0
                    while sum < top_p:
                        row = xy_counts.iloc[i]
                        x_mode = int(row['x'])
                        y_mode = int(row['y'])
                        ratio = round(row['count'] / total, 3)
                        sum += ratio
                        i += 1
                        temp_res.append([int(x_mode), int(y_mode), float(ratio)])

                else: # 此t沒有資料
                    if t<=12 or t >= 40: # 若在早上0點到6點及晚上8點到12點，取home點
                        temp_res.append([int(global_home_x), int(global_home_y), 1.0])
                    else: # 其他時間取上一個點的mode
                        temp_res = result[-1]['mode']
                result.append({'uid': uid, 't': t, 'working_day': 0, 'mode': temp_res})
            print(f'訓練進度: {idx+1}/{len(valid_uid_list)} 使用者ID={uid}', end='\r')

        train_mode_df = pd.DataFrame(result)
        train_mode_df.to_csv(f'./ckpt/{output_name}_Per_User_Per_t_Mode_working_day_top_p.csv', index=False)
        print(f"Per_User_Per_t_Mode_working_day_top_p: 結果已儲存至 ./ckpt/{output_name}_Per_User_Per_t_Mode_working_day_top_p.csv")

        # 2、預測，依據測試資料的時間點，將模式依據機率應用到測試資料
        result = []
        for i, uid in enumerate(valid_uid_list):
            train_user_df = train_mode_df[train_mode_df['uid'] == uid]
            test_user_df = self.test_data[self.test_data['uid'] == uid]

            # 依據ratio當權重選擇mode
            def sample_xy_from_mode(mode_list):
                xs = [item[0] for item in mode_list]
                ys = [item[1] for item in mode_list]
                ratios = [item[2] for item in mode_list]
                idx = random.choices(range(len(mode_list)), weights=ratios, k=1)[0]
                return xs[idx], ys[idx]

            # 取得工作日和非工作日的模式
            days = np.sort(test_user_df['d'].unique())
            for day in days:
                hours = np.sort(test_user_df[test_user_df['d'] == day]['t'].unique())
                if test_user_df[test_user_df['d'] == day]['working_day'].values[0] == 1: # 工作日
                    for hour in hours:
                        mode_series = train_user_df[(train_user_df['t'] == hour) & (train_user_df['working_day'] == 1)]['mode']
                        mode_list = mode_series.values[0]
                        x, y = sample_xy_from_mode(mode_list)
                        result.append({'uid': uid, 'd':day, 't': hour, 'x': x, 'y': y})
                else: # 非工作日
                    for hour in hours:
                        mode_series = train_user_df[(train_user_df['t'] == hour) & (train_user_df['working_day'] == 0)]['mode']
                        mode_list = mode_series.values[0]
                        x, y = sample_xy_from_mode(mode_list)
                        result.append({'uid': uid, 'd':day, 't': hour, 'x': x, 'y': y})

            print(f'預測進度: {i+1}/{len(valid_uid_list)} 使用者ID={uid}', end='\r')
        prediction_df = pd.DataFrame(result)
        prediction_df = prediction_df[['uid', 'd', 't', 'x', 'y']].astype(int)

        prediction_df.to_csv(f'./Predictions/{output_name}_Per_User_Per_t_Mode_working_day_top_p.csv', index=False)
        print(f"Per_User_Per_t_Mode_working_day: 結果已儲存至 ./Predictions/{output_name}_Per_User_Per_t_Mode_working_day_top_p.csv")
        return prediction_df

    def Per_User_Per_t_Mode_day_of_week(self, valid_uid_list, output_name, early_stop=None):
        os.makedirs('./Predictions', exist_ok=True)
        os.makedirs('./ckpt', exist_ok=True)
        print(f'Per_User_Per_t_Mode: 使用者數量={len(valid_uid_list)}')

        if early_stop is not None and early_stop < len(valid_uid_list):
            valid_uid_list = sorted(random.sample(list(valid_uid_list), early_stop))
            print(f'隨機抽取 {early_stop} 個 uid 進行訓練/預測')

        start_time = time.time()
        # Train資料中每個使用者在每個時間點的x,y值的全域模式並且分成禮拜一到禮拜天
        day_of_week_list = [1,2,3,4,5,6,0] # 1:禮拜一, 2:禮拜二, ..., 0:禮拜天
        hours = np.arange(0, 48)
        result = []
        for i, uid in enumerate(valid_uid_list):
            user_df = self.train_data[self.train_data['uid'] == uid]
            if user_df.empty: # 如果使用者在訓練資料中沒有資料，則跳過
                continue
            x_user_global_mode = user_df['x'].mode() # global用於沒有資料的情況補值
            y_user_global_mode = user_df['y'].mode()
            
            # 先計算每個時間點的mode用於沒有資料的情況補值
            hour_mode_list = []
            for hour in hours:
                hour_df = user_df[user_df['t'] == hour]
                if not hour_df.empty: 
                    hour_mode_list.append({
                        't': hour,
                        'x_mode': hour_df['x'].mode().values[0], 
                        'y_mode': hour_df['y'].mode().values[0] 
                    })

                else:
                    hour_mode_list.append({
                        't': hour,
                        'x_mode': x_user_global_mode.values[0],
                        'y_mode': y_user_global_mode.values[0] 
                    })

            # 再計算禮拜一到禮拜天每個時間點的mode
            for day in day_of_week_list:
                day_df = user_df[user_df['day_of_week']==day]
                for hour in hours:
                    hour_df = day_df[day_df['t'] == hour]
                    if not hour_df.empty and len(hour_df) >= 5: # 有資料且至少5筆
                        x_mode = hour_df['x'].mode().values[0]
                        y_mode = hour_df['y'].mode().values[0]
                    else: # 沒有資料的情況用此時間的global補值
                        x_mode = hour_mode_list[hour]['x_mode'] 
                        y_mode = hour_mode_list[hour]['y_mode'] 
                    result.append({'uid': uid, 't': hour, 'x': x_mode, 'y': y_mode, 'day_of_week': day})

            print(f'訓練進度: {i+1}/{len(valid_uid_list)} 使用者ID={uid}', end='\r')

        train_mode_df = pd.DataFrame(result)
        train_mode_df.to_csv(f'./ckpt/{output_name}_Per_User_Per_t_Mode_day_of_week.csv', index=False)
        print(f"Train_Mode: 結果已儲存至 ./ckpt/{output_name}_Per_User_Per_t_Mode_day_of_week.csv")

        # 依據測試資料的時間點，將模式應用到測試資料
        before_non_uid =[]
        after_non_uid = []
        result = []
        for i, uid in enumerate(valid_uid_list):
            train_user_df = train_mode_df[train_mode_df['uid'] == uid]
            test_user_df = self.test_data[self.test_data['uid'] == uid]

            # 若之前之後沒有資料，則跳過
            if train_user_df.empty:
                before_non_uid.append(uid)
                print(f'使用者ID={uid} 在訓練資料中沒有模式，跳過此使用者')
                continue
            if test_user_df.empty:
                after_non_uid.append(uid)
                print(f'使用者ID={uid} 在測試資料中沒有資料，跳過此使用者')
                continue

            # 取得工作日和非工作日的模式
            days = np.sort(test_user_df['d'].unique())
            for day in days:
                user_df = test_user_df[test_user_df['d'] == day]
                day_of_week = user_df['day_of_week'].values[0]
                hours = np.sort(test_user_df[test_user_df['d'] == day]['t'].unique())
                for hour in hours:
                    x_mode = train_user_df[(train_user_df['t'] == hour) & (train_user_df['day_of_week'] == day_of_week)]['x'].values[0]
                    y_mode = train_user_df[(train_user_df['t'] == hour) & (train_user_df['day_of_week'] == day_of_week)]['y'].values[0]
                    result.append({'uid': uid, 'd':day, 't': hour, 'x': x_mode, 'y': y_mode})

            print(f'預測進度: {i+1}/{len(valid_uid_list)} 使用者ID={uid}', end='\r')

        prediction_df = pd.DataFrame(result)
        prediction_df = prediction_df[['uid', 'd', 't', 'x', 'y']].astype(int)

        prediction_df.to_csv(f'./Predictions/{output_name}_Per_User_Per_t_Mode_day_of_week.csv', index=False)
        print(f"Per_User_Per_t_Mode_day_of_week: 結果已儲存至 ./Predictions/{output_name}_Per_User_Per_t_Mode_day_of_week.csv")

        elapsed_time = time.time() - start_time
        print(f'before_non_uid: {before_non_uid}')
        print(f'after_non_uid: {after_non_uid}')
        print(f"Per_User_Per_t_Mode_day_of_week: 執行時間: {elapsed_time//60:.2f}min")
        return prediction_df

    def Per_User_Markov(self, valid_uid_list=None, output_name='markov', early_stop=None, top_p=0.7):
        os.makedirs('./Predictions', exist_ok=True)
        if valid_uid_list is None:
            valid_uid_list = self.test_data['uid'].unique()
        if early_stop is not None and early_stop < len(valid_uid_list):
            valid_uid_list = sorted(random.sample(list(valid_uid_list), early_stop))
            print(f'隨機抽取 {early_stop} 個 uid 進行預測')

        result = []
        start_time = time.time()
        for i, uid in enumerate(valid_uid_list):
            user_train = self.train_data[self.train_data['uid'] == uid].sort_values(['d', 't'])
            user_test = self.test_data[self.test_data['uid'] == uid].sort_values(['d', 't'])
            if user_train.empty or user_test.empty:
                continue

            # 建立馬可夫轉移表：key=(prev_t, curr_t, prev_x, prev_y)，value=Counter of (x, y)
            transitions = defaultdict(Counter)
            prev = None
            prev_t = None
            for _, row in user_train.iterrows():
                curr = (row['x'], row['y'])
                curr_t = row['t']
                if prev is not None and prev_t is not None:
                    transitions[(prev_t, curr_t, prev[0], prev[1])][curr] += 1
                prev = curr
                prev_t = curr_t

            # 取得初始點：train的最後一天的最後一個點
            last_row = user_train.iloc[-1]
            last_t = last_row['t']
            last_xy = (last_row['x'], last_row['y'])

            # 預測時
            for _, row in user_test.iterrows():
                key = (last_t, row['t'], last_xy[0], last_xy[1])
                next_xy = None
                if key in transitions and transitions[key]:
                    # top_p sampling ...
                    items = transitions[key].most_common()
                    total = sum(cnt for _, cnt in items)
                    probs = [cnt / total for _, cnt in items]
                    cum_prob = 0
                    top_items = []
                    for (xy, p) in zip([xy for xy, _ in items], probs):
                        cum_prob += p
                        top_items.append(xy)
                        if cum_prob >= top_p:
                            break
                    next_xy = random.choice(top_items)
                else:
                    # 其餘 fallback 保持原本眾數邏輯
                    t_mode_x = user_train[user_train['t'] == row['t']]['x'].mode()
                    t_mode_y = user_train[user_train['t'] == row['t']]['y'].mode()
                    if not t_mode_x.empty and not t_mode_y.empty:
                        next_xy = (t_mode_x.values[0], t_mode_y.values[0])
                    else:
                        next_xy = (user_train['x'].mode().values[0], user_train['y'].mode().values[0])
                result.append({'uid': uid, 'd': row['d'], 't': row['t'], 'x': next_xy[0], 'y': next_xy[1]})
                last_t = row['t']
                last_xy = next_xy

            print(f'預測進度: {i+1}/{len(valid_uid_list)} 使用者ID={uid}', end='\r')

        prediction_df = pd.DataFrame(result)
        prediction_df = prediction_df[['uid', 'd', 't', 'x', 'y']].astype(int)
        prediction_df.to_csv(f'./Predictions/{output_name}_Per_User_Markov.csv', index=False)
        print(f"\nMarkov預測完成，結果已儲存至 ./Predictions/{output_name}_Per_User_Markov.csv")
        elapsed_time = time.time() - start_time
        print(f"Per_User_Markov: 執行時間: {elapsed_time//60:.2f}min")
        return prediction_df

    def Per_User_Markov_working_day(self, valid_uid_list=None, output_name='markov', early_stop=None, top_p=0.7):
        os.makedirs('./Predictions', exist_ok=True)
        if valid_uid_list is None:
            valid_uid_list = self.test_data['uid'].unique()
        if early_stop is not None and early_stop < len(valid_uid_list):
            valid_uid_list = sorted(random.sample(list(valid_uid_list), early_stop))
            print(f'隨機抽取 {early_stop} 個 uid 進行預測')

        result = []
        start_time = time.time()
        for i, uid in enumerate(valid_uid_list):
            user_train = self.train_data[self.train_data['uid'] == uid].sort_values(['d', 't'])
            user_test = self.test_data[self.test_data['uid'] == uid].sort_values(['d', 't'])
            if user_train.empty or user_test.empty:
                continue

            # 分別建立 working_day=1 和 working_day=0 的轉移表
            transitions_dict = {0: defaultdict(Counter), 1: defaultdict(Counter)}
            for wd in [0, 1]:
                prev = None
                prev_t = None
                for _, row in user_train[user_train['working_day'] == wd].iterrows():
                    curr = (row['x'], row['y'])
                    curr_t = row['t']
                    if prev is not None and prev_t is not None:
                        transitions_dict[wd][(prev_t, curr_t, prev[0], prev[1])][curr] += 1
                    prev = curr
                    prev_t = curr_t

            # 取得初始點
            last_row = user_train.iloc[-1]
            last_t = last_row['t']
            last_xy = (last_row['x'], last_row['y'])
            last_wd = last_row['working_day']

            for _, row in user_test.iterrows():
                wd = row['working_day']
                key = (last_t, row['t'], last_xy[0], last_xy[1])
                next_xy = None
                transitions = transitions_dict.get(wd, defaultdict(Counter))
                if key in transitions and transitions[key]:
                    # top_p sampling
                    items = transitions[key].most_common()
                    total = sum(cnt for _, cnt in items)
                    probs = [cnt / total for _, cnt in items]
                    cum_prob = 0
                    top_items = []
                    for (xy, p) in zip([xy for xy, _ in items], probs):
                        cum_prob += p
                        top_items.append(xy)
                        if cum_prob >= top_p:
                            break
                    next_xy = random.choice(top_items)
                else:
                    # fallback: 該時間下的眾數
                    t_mode_x = user_train[(user_train['t'] == row['t']) & (user_train['working_day'] == wd)]['x'].mode()
                    t_mode_y = user_train[(user_train['t'] == row['t']) & (user_train['working_day'] == wd)]['y'].mode()
                    if not t_mode_x.empty and not t_mode_y.empty:
                        next_xy = (t_mode_x.values[0], t_mode_y.values[0])
                    else:
                        next_xy = (user_train['x'].mode().values[0], user_train['y'].mode().values[0])
                result.append({'uid': uid, 'd': row['d'], 't': row['t'], 'x': next_xy[0], 'y': next_xy[1]})
                last_t = row['t']
                last_xy = next_xy
                last_wd = wd

            print(f'預測進度: {i+1}/{len(valid_uid_list)} 使用者ID={uid}', end='\r')

        prediction_df = pd.DataFrame(result)
        prediction_df = prediction_df[['uid', 'd', 't', 'x', 'y']].astype(int)
        prediction_df.to_csv(f'./Predictions/{output_name}_Per_User_Markov_working_day.csv', index=False)
        print(f"\nMarkov_working_day預測完成，結果已儲存至 ./Predictions/{output_name}_Per_User_Markov_working_day.csv")
        elapsed_time = time.time() - start_time
        print(f"Per_User_Markov_working_day: 執行時間: {elapsed_time//60:.2f}min")
        return prediction_df

    def Evaluation(self, generated_data_input, reference_data_input, valid=False, city_name=None, raw_data_path=None):
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


"""
測試程式碼
"""
if __name__ == "__main__":
    # 檢查同一個cluster分數的那3000人45vs15-->Per_User_Per_t_Mode_working_day_modify
    raw_train_data_df = pd.read_csv('./Training_Testing_Data/A_y_train.csv', header=0)
    feature_df = pd.read_csv('./Stability/A_features.csv', header=0)
    cluster_df = pd.read_csv('./Stability/A_activity_space.csv', header=0)
    
    train_data_df_45 = raw_train_data_df[raw_train_data_df['d'] <= 45]
    test_data_df_15 = raw_train_data_df[raw_train_data_df['d'] > 45]
    cluster = 3


    valid_uid_list = cluster_df[(cluster_df['cluster'] == cluster) & (cluster_df['uid'] > 147000)]['uid'].unique() # !!!!!!!!!!!!!!!!!!!!!!!!!!
    train_data_df_45 = train_data_df_45[train_data_df_45['uid'].isin(valid_uid_list)]
    test_data_df_15 = test_data_df_15[test_data_df_15['uid'].isin(valid_uid_list)]
    print(f'前45天有效的使用者ID數量: {len(train_data_df_45["uid"].unique())}')
    print(f'後15天有效的使用者ID數量: {len(test_data_df_15["uid"].unique())}')
    train_uids = set(train_data_df_45["uid"].unique())
    test_uids = set(test_data_df_15["uid"].unique())
    print(f'有效的使用者ID數量: {len(valid_uid_list)}')


    std_model_zoo = ModelZoo(train_data_df_45, test_data_df_15)
    std_model_zoo.Per_User_Per_t_Mode_working_day_modify(
            feature_df = feature_df,
            valid_uid_list = valid_uid_list,
            output_name=f'A_y_cluster{cluster}_modify',
            early_stop=150000
        )

    final_GEOBLEU_score, final_DTW_score = std_model_zoo.Evaluation(
        generated_data_input = f'./Predictions/A_y_cluster{cluster}_modify_Per_User_Per_t_Mode_working_day_modify.csv',
        reference_data_input = test_data_df_15,
    )
    print(f"最終GEO-BLEU分數: {final_GEOBLEU_score:.4f}, 最終DTW分數: {final_DTW_score:.4f}\n\n")

    # 可視化比較
    target_uid = valid_uid_list[0]
    gen_df = pd.read_csv(f'./Predictions/A_y_cluster{cluster}_modify_Per_User_Per_t_Mode_working_day_modify.csv', header=0)
    user_test_df = test_data_df_15[test_data_df_15['uid'] == target_uid]
    future_traj = gen_df[gen_df['uid'] == target_uid][['x', 'y']].values  
    true_traj = user_test_df[['x', 'y']].values  # 真實第61~75天

    plt.figure(figsize=(8, 6))
    plt.scatter(true_traj[:, 0], true_traj[:, 1], label='True', color='blue', marker='o', alpha=0.5, s=3)
    plt.scatter(future_traj[:, 0], future_traj[:, 1], label='Generated', color='red', marker='x', alpha=0.5, s=3)
    plt.title(f'UID {target_uid} 第61~75天軌跡比較')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()

    # # 檢查同一個cluster分數-->Per_User_Per_t_Mode_working_day_modify
    # raw_train_data_df = pd.read_csv('./Training_Testing_Data/A_x_train.csv', header=0)
    # raw_test_data_df = pd.read_csv('./Training_Testing_Data/A_x_test.csv', header=0)
    # feature_df = pd.read_csv('./Stability/A_features.csv', header=0)
    # cluster_df = pd.read_csv('./Stability/A_activity_space.csv', header=0)

    # valid_uid_list = cluster_df[(cluster_df['cluster'] == 1) & (cluster_df['uid'] <= 147000)]['uid'].unique()
    # print(f'有效的使用者ID數量: {len(valid_uid_list)}')
    # std_model_zoo = ModelZoo(raw_train_data_df, raw_test_data_df)
    # std_model_zoo.Per_User_Per_t_Mode_working_day_modify(
    #         feature_df = feature_df,
    #         valid_uid_list = valid_uid_list,
    #         output_name=f'A_x_cluster1_modify',
    #         early_stop=150000
    #     )

    # final_GEOBLEU_score, final_DTW_score = std_model_zoo.Evaluation(
    #     generated_data_input = f'./Predictions/A_x_cluster1_modify_Per_User_Per_t_Mode_working_day_modify.csv',
    #     reference_data_input = raw_test_data_df,
    # )
    # print(f"最終GEO-BLEU分數: {final_GEOBLEU_score:.4f}, 最終DTW分數: {final_DTW_score:.4f}\n\n")

    # # 可視化比較
    # target_uid = valid_uid_list[0]
    # gen_df = pd.read_csv(f'./Predictions/A_x_cluster1_modify_Per_User_Per_t_Mode_working_day_modify.csv', header=0)
    # user_test_df = raw_test_data_df[raw_test_data_df['uid'] == target_uid]
    # future_traj = gen_df[gen_df['uid'] == target_uid][['x', 'y']].values  
    # true_traj = user_test_df[['x', 'y']].values  # 真實第61~75天

    # plt.figure(figsize=(8, 6))
    # plt.scatter(true_traj[:, 0], true_traj[:, 1], label='True', color='blue', marker='o', alpha=0.5, s=3)
    # plt.scatter(future_traj[:, 0], future_traj[:, 1], label='Generated', color='red', marker='x', alpha=0.5, s=3)
    # plt.title(f'UID {target_uid} 第61~75天軌跡比較')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # # 檢查安旭的3000人中位數
    # generated_df = pd.read_csv('./Predictions/rf_cityA3000_result.csv', header=0)
    # generated_df_sorted = generated_df.sort_values(by='peers_rf_geobleu', ascending=False)
    # median_idx = len(generated_df_sorted) // 2
    # start_idx = max(median_idx - 5, 0)
    # end_idx = min(median_idx + 5, len(generated_df_sorted))
    # median_10_uid = generated_df_sorted.iloc[start_idx:end_idx]['uid'].tolist()
    # print(generated_df_sorted[generated_df_sorted['uid'].isin(median_10_uid)])

    # # mode vs. gt 輸出scatter比較
    # raw_train_data_df = pd.read_csv('./Training_Testing_Data/A_y_train.csv', header=0)
    # train_data_df = raw_train_data_df[raw_train_data_df['d'] <= 45].copy()
    # test_data_df = raw_train_data_df[raw_train_data_df['d'] > 45].copy()   
    
    # # 解析 peers_rf_pred 並展平成 (uid, d, t, x, y)
    # def parse_peers_pred(val):
    #     if isinstance(val, str):
    #         try:
    #             items = ast.literal_eval(val)
    #         except Exception:
    #             return []
    #     elif isinstance(val, (list, tuple)):
    #         items = val
    #     else:
    #         return []
    #     out = []
    #     for it in items:
    #         if isinstance(it, (list, tuple)) and len(it) == 4:
    #             d, t, x, y = it
    #             out.append((int(d), int(t), int(x), int(y)))
    #     return out

    # rows = []
    # for _, r in generated_df_sorted.iterrows():
    #     uid = int(r['uid'])
    #     for d, t, x, y in parse_peers_pred(r['peers_rf_pred']):
    #         rows.append({'uid': uid, 'd': d, 't': t, 'x': x, 'y': y})
    # mode_pred_df = pd.DataFrame(rows)

    # gt_df = test_data_df
    # valid_uid_list = median_10_uid[5:]

    # fig, axes = plt.subplots(2, len(valid_uid_list), figsize=(20,20))
    # for i, uid in enumerate(valid_uid_list):
    #     axes[0, i].scatter(mode_pred_df[mode_pred_df['uid'] == uid]['x'],
    #                     mode_pred_df[mode_pred_df['uid'] == uid]['y'],
    #                     label='Mode', alpha=0.8, s=10, color='red', marker='x')
    #     axes[0, i].scatter(gt_df[gt_df['uid'] == uid]['x'],
    #             gt_df[gt_df['uid'] == uid]['y'],
    #             label='gt', alpha=0.3, s=3, color='green')
    #     axes[0, i].set_title(f'UID {uid} Mode')
    #     axes[0, i].set_xlabel('x')
    #     axes[0, i].set_ylabel('y')
    #     axes[0, i].set_aspect('equal')
    #     axes[0, i].set_xlim(1, 200)
    #     axes[0, i].set_ylim(1, 200)
    #     axes[0, i].grid(True)
    #     axes[0, i].invert_yaxis()
    #     axes[0, i].legend()

    #     axes[1, i].scatter(train_data_df[train_data_df['uid'] == uid]['x'],
    #             train_data_df[train_data_df['uid'] == uid]['y'],
    #             label='45gt', alpha=0.5, s=5, color='blue')
    #     axes[1, i].scatter(gt_df[gt_df['uid'] == uid]['x'],
    #             gt_df[gt_df['uid'] == uid]['y'],
    #             label='15gt', alpha=1, s=5, color='green')
    #     axes[1, i].set_title(f'UID {uid} CVAE')
    #     axes[1, i].set_xlabel('x')  
    #     axes[1, i].set_ylabel('y')
    #     axes[1, i].set_aspect('equal')
    #     axes[1, i].set_xlim(1, 200)
    #     axes[1, i].set_ylim(1, 200)
    #     axes[1, i].grid(True)
    #     axes[1, i].invert_yaxis()
    #     axes[1, i].legend()

    # plt.tight_layout()
    # plt.show()

    # # 看特定人的x,y時間序列圖，把GT, mode 的x,y拉出來看時間線段上的重合性
    # def plot_x_y_sequence_compare(uid, mode_df , gt_df):
    #     # 依照時間排序
    #     mode_user = mode_df[mode_df['uid'] == uid].sort_values(['d', 't'])
    #     gt_user = gt_df[gt_df['uid'] == uid].sort_values(['d', 't'])

    #     fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharex=True)

    #     # 左上：x的mode和gt
    #     axes[0].plot(mode_user['x'].values, '-o', label='Mode', color='red', alpha=0.7)
    #     axes[0].plot(gt_user['x'].values, '-o', label='GT', color='green', alpha=0.7)
    #     axes[0].set_title(f'UID {uid} x 時序 (Mode vs GT)')
    #     axes[0].set_ylabel('x')
    #     axes[0].legend()
    #     axes[0].grid(True)

    #     # 右上：y的mode和gt
    #     axes[1].plot(mode_user['y'].values, '-o', label='Mode', color='red', alpha=0.7)
    #     axes[1].plot(gt_user['y'].values, '-o', label='GT', color='green', alpha=0.7)
    #     axes[1].set_title(f'UID {uid} y 時序 (Mode vs GT)')
    #     axes[1].set_ylabel('y')
    #     axes[1].legend()
    #     axes[1].grid(True)

    #     plt.tight_layout()
    #     plt.show()

    # valid_uid_list = mode_pred_df['uid'].unique().tolist()
    # plot_x_y_sequence_compare(uid=148120,
    #                             mode_df=mode_pred_df,
    #                             gt_df=gt_df)

    # # cityA的3000人前45天和後15天的比較，並計算每個人的geobleu和dtw分數
    # raw_train_data_df = pd.read_csv('./Training_Testing_Data/A_y_train.csv', header=0)
    # feature_df = pd.read_csv('./Stability/A_features.csv', header=0)

    # train_data_df = raw_train_data_df[raw_train_data_df['d'] <= 45].copy()
    # test_data_df = raw_train_data_df[raw_train_data_df['d'] > 45].copy()    

    # print(f'前45天有效的使用者ID數量: {len(train_data_df["uid"].unique())}')
    # print(f'後15天有效的使用者ID數量: {len(test_data_df["uid"].unique())}')
    # train_uids = set(train_data_df["uid"].unique())
    # test_uids = set(test_data_df["uid"].unique())
    # valid_uid_list = sorted(list(train_uids & test_uids))
    # print(f'有效的使用者ID數量: {len(valid_uid_list)}')

    # std_model_zoo = ModelZoo(train_data_df, test_data_df)
    # std_model_zoo.Per_User_Per_t_Mode_working_day_modify(
    #         feature_df = feature_df,
    #         valid_uid_list = valid_uid_list,
    #         output_name=f'A_y_3000_45vs15',
    #         early_stop=150000
    #     )
    
    # # 計算個別分數並輸出
    # score_res = []
    # generated_df = pd.read_csv(f'./Predictions/A_y_3000_45vs15_Per_User_Per_t_Mode_working_day_modify.csv', header=0)
    # reference_df = test_data_df
    # for idx, uid in enumerate(valid_uid_list):
    #     gen_user = generated_df[generated_df['uid'] == uid]
    #     ref_user = reference_df[reference_df['uid'] == uid]

    #     gen_traj = gen_user[['d', 't', 'x', 'y']].to_records(index=False)
    #     ref_traj = ref_user[['d', 't', 'x', 'y']].to_records(index=False)
    #     gen_traj = [tuple(row) for row in gen_traj]
    #     ref_traj = [tuple(row) for row in ref_traj]

    #     # GEOBLEU_score
    #     GEOBLEU_score = geobleu.calc_geobleu_single(gen_traj, ref_traj)

    #     # dtw
    #     DTW_score = geobleu.calc_dtw_single(gen_traj, ref_traj)

    #     score_res.append({
    #         'uid': uid,
    #         'GEOBLEU_score': GEOBLEU_score,
    #         'DTW_score': DTW_score
    #     })

    #     print(f"{idx}/{len(valid_uid_list)}人--uid={uid}", end='\r')
    
    # score_df = pd.DataFrame(score_res)
    # score_df.to_csv(f'./Scores/A_y_3000_45vs15_scores.csv', index=False)

   

    # # 看這3000人的分數及挑出特定幾人看分布與軌跡
    # # 先依據geo_bleu sorted
    # score_df = pd.read_csv(f'./Scores/A_y_3000_45vs15_scores.csv', header=0)
    # score_df = score_df.sort_values(by='GEOBLEU_score', ascending=False)
    # print(score_df.head(10))
    # print(score_df.tail(10))
    # top_10_uid = score_df.head(10)['uid'].tolist()
    # tail_10_uid = score_df.tail(10)['uid'].tolist()  

    # geobleu_sorted = score_df.sort_values(by='GEOBLEU_score', ascending=False).reset_index(drop=True)
    # median_idx = len(geobleu_sorted) // 2
    # start_idx = max(median_idx - 5, 0)
    # end_idx = min(median_idx + 5, len(geobleu_sorted))
    # median_10_uid = geobleu_sorted.iloc[start_idx:end_idx]['uid'].tolist()
    # print(score_df[score_df['uid'].isin(median_10_uid)])

    # # mode vs. gt 輸出scatter比較
    # raw_train_data_df = pd.read_csv('./Training_Testing_Data/A_y_train.csv', header=0)
    # train_data_df = raw_train_data_df[raw_train_data_df['d'] <= 45].copy()
    # test_data_df = raw_train_data_df[raw_train_data_df['d'] > 45].copy()   
    
    # mode_pred_df = pd.read_csv('./Predictions/A_y_3000_45vs15_Per_User_Per_t_Mode_working_day_modify.csv')
    # gt_df = test_data_df
    # valid_uid_list = tail_10_uid[5:]

    # fig, axes = plt.subplots(2, len(valid_uid_list), figsize=(20,20))
    # for i, uid in enumerate(valid_uid_list):
    #     axes[0, i].scatter(mode_pred_df[mode_pred_df['uid'] == uid]['x'],
    #                     mode_pred_df[mode_pred_df['uid'] == uid]['y'],
    #                     label='Mode', alpha=0.8, s=10, color='red', marker='x')
    #     axes[0, i].scatter(gt_df[gt_df['uid'] == uid]['x'],
    #             gt_df[gt_df['uid'] == uid]['y'],
    #             label='gt', alpha=0.3, s=3, color='green')
    #     axes[0, i].set_title(f'UID {uid} Mode')
    #     axes[0, i].set_xlabel('x')
    #     axes[0, i].set_ylabel('y')
    #     axes[0, i].set_aspect('equal')
    #     axes[0, i].set_xlim(1, 200)
    #     axes[0, i].set_ylim(1, 200)
    #     axes[0, i].grid(True)
    #     axes[0, i].invert_yaxis()
    #     axes[0, i].legend()

    #     axes[1, i].scatter(train_data_df[train_data_df['uid'] == uid]['x'],
    #             train_data_df[train_data_df['uid'] == uid]['y'],
    #             label='45gt', alpha=0.5, s=5, color='blue')
    #     axes[1, i].scatter(gt_df[gt_df['uid'] == uid]['x'],
    #             gt_df[gt_df['uid'] == uid]['y'],
    #             label='15gt', alpha=1, s=5, color='green')
    #     axes[1, i].set_title(f'UID {uid} CVAE')
    #     axes[1, i].set_xlabel('x')  
    #     axes[1, i].set_ylabel('y')
    #     axes[1, i].set_aspect('equal')
    #     axes[1, i].set_xlim(1, 200)
    #     axes[1, i].set_ylim(1, 200)
    #     axes[1, i].grid(True)
    #     axes[1, i].invert_yaxis()
    #     axes[1, i].legend()

    # plt.tight_layout()
    # plt.show()

    # # 依據標準差看分數
    # raw_std_df = pd.read_csv('./Stability/A_ytrain_working_day_stability.csv', header=0)
    # thresholds = [0,9999]
    # for i in range(len(thresholds) - 1):
    #     lower = thresholds[i]
    #     upper = thresholds[i + 1]
    #     filter_std_df = raw_std_df[(raw_std_df['x_std_mean'] >= lower) | (raw_std_df['y_std_mean'] >= lower)]
    #     valid_uid_list = filter_std_df[(filter_std_df['x_std_mean'] < upper) & (filter_std_df['y_std_mean'] < upper)]['uid'].unique()
    #     print(f"x|y std >= {lower},x&y std < {upper} 有效的使用者ID數量: {len(valid_uid_list)}")

    #     geobleu_scores = score_df[score_df['uid'].isin(valid_uid_list)]['GEOBLEU_score'].mean()
    #     dtw_scores = score_df[score_df['uid'].isin(valid_uid_list)]['DTW_score'].mean()
    #     print(geobleu_scores, dtw_scores, '\n')

    # # 看特定人的x,y時間序列圖，把GT, mode 的x,y拉出來看時間線段上的重合性
    # def plot_x_y_sequence_compare(uid, mode_df , gt_df):
    #     # 依照時間排序
    #     mode_user = mode_df[mode_df['uid'] == uid].sort_values(['d', 't'])
    #     gt_user = gt_df[gt_df['uid'] == uid].sort_values(['d', 't'])

    #     fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharex=True)

    #     # 左上：x的mode和gt
    #     axes[0].plot(mode_user['x'].values, '-o', label='Mode', color='red', alpha=0.7)
    #     axes[0].plot(gt_user['x'].values, '-o', label='GT', color='green', alpha=0.7)
    #     axes[0].set_title(f'UID {uid} x 時序 (Mode vs GT)')
    #     axes[0].set_ylabel('x')
    #     axes[0].legend()
    #     axes[0].grid(True)

    #     # 右上：y的mode和gt
    #     axes[1].plot(mode_user['y'].values, '-o', label='Mode', color='red', alpha=0.7)
    #     axes[1].plot(gt_user['y'].values, '-o', label='GT', color='green', alpha=0.7)
    #     axes[1].set_title(f'UID {uid} y 時序 (Mode vs GT)')
    #     axes[1].set_ylabel('y')
    #     axes[1].legend()
    #     axes[1].grid(True)

    #     plt.tight_layout()
    #     plt.show()

    # valid_uid_list = mode_pred_df['uid'].unique().tolist()
    # plot_x_y_sequence_compare(uid=149254,
    #                             mode_df=mode_pred_df,
    #                             gt_df=gt_df)

    # 不同std分類對分數影響-->Per_User_Per_t_Mode_working_day_dynamic
    # raw_train_data_df = pd.read_csv('./Training_Testing_Data/A_x_train.csv', header=0)
    # raw_test_data_df = pd.read_csv('./Training_Testing_Data/A_x_test.csv', header=0)
    # raw_std_df = pd.read_csv('./Stability/A_xtrain_working_day_stability.csv', header=0)
    # feature_df = pd.read_csv('./Stability/A_features.csv', header=0)

    # std_model_zoo = ModelZoo(raw_train_data_df, raw_test_data_df)
    # target_uid = 24

    # std_model_zoo.Per_User_Per_t_Mode_working_day_modify(
    #         feature_df = feature_df, 
    #         valid_uid_list = [target_uid], 
    #         output_name=f'A_x_uid{target_uid}_modify',
    #         early_stop=3000
    #     )

    # final_GEOBLEU_score, final_DTW_score = std_model_zoo.Evaluation(
    #     generated_data_input = f'./Predictions/A_x_uid{target_uid}_modify_Per_User_Per_t_Mode_working_day_modify.csv',
    #     reference_data_input = raw_test_data_df,
    #     valid=False,
    #     city_name='a',
    #     raw_data_path='./Data/city_A_challengedata.csv'
    # )
    # print(f"最終GEO-BLEU分數: {final_GEOBLEU_score:.4f}, 最終DTW分數: {final_DTW_score:.4f}\n\n")

    # std_model_zoo.Per_User_Per_t_Mode_working_day_dynamic(
    #         feature_df = feature_df, 
    #         valid_uid_list = [target_uid], 
    #         output_name=f'A_x_uid{target_uid}_dynamic',
    #         early_stop=3000
    #     )
    
    # final_GEOBLEU_score, final_DTW_score = std_model_zoo.Evaluation(
    #     generated_data_input = f'./Predictions/A_x_uid{target_uid}_dynamic_Per_User_Per_t_Mode_working_day_dynamic.csv',
    #     reference_data_input = raw_test_data_df,
    #     valid=False,
    #     city_name='a',
    #     raw_data_path='./Data/city_A_challengedata.csv'
    # )
    # print(f"最終GEO-BLEU分數: {final_GEOBLEU_score:.4f}, 最終DTW分數: {final_DTW_score:.4f}\n\n")

    # thresholds = [0, 1, 2, 3, 4, 5, 10, 9999]
    # thresholds = [0, 9999]
    # for i in range(len(thresholds) - 1):
    #     lower = thresholds[i]
    #     upper = thresholds[i + 1]

    #     filter_std_df = raw_std_df[(raw_std_df['x_std_mean'] >= lower) | (raw_std_df['y_std_mean'] >= lower)]
    #     valid_uid_list = filter_std_df[(filter_std_df['x_std_mean'] < upper) & (filter_std_df['y_std_mean'] < upper)]['uid'].unique()
    #     print(f"x|y std >= {lower},x&y std < {upper} 有效的使用者ID數量: {len(valid_uid_list)}")

    #     std_model_zoo.Per_User_Per_t_Mode_working_day_dynamic(
    #         feature_df = feature_df, 
    #         valid_uid_list = valid_uid_list, 
    #         output_name=f'A_x_std{upper}',
    #         early_stop=10000
    #     )

    #     final_GEOBLEU_score, final_DTW_score = std_model_zoo.Evaluation(
    #         generated_data_input = f'./Predictions/A_x_std{upper}_Per_User_Per_t_Mode_working_day_dynamic.csv',
    #         reference_data_input = raw_test_data_df,
    #         valid=False,
    #         city_name='a',
    #         raw_data_path='./Data/city_A_challengedata.csv'
    #     )
    #     print(f"最終GEO-BLEU分數: {final_GEOBLEU_score:.4f}, 最終DTW分數: {final_DTW_score:.4f}\n\n")

    
    """
    以下用於檢查下午五點之後dunamic生成的資料與modify資料的差異
    這段程式碼會產生動畫圖
    """
   

