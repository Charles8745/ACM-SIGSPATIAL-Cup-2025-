import os
import numpy as np
import pandas as pd
import validator_InModify as validator
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoLocator
import geobleu
import time
import random
import joblib
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import shap
import xgboost as xgb
import lightgbm as lgb
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 或 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

class ModelZoo:
    def __init__(self):
        pass

    def DataPreparation(self,train_df, feature_df, feature_list, valid_uid_list=None):
        """
        df: DataFrame (x or y train)
        feature_list: list of feature names to use, e.g. [uid, d, t, x, y, day_of_week, working_day, ...]
        return: X (features), y (labels, if available)
        """
        if valid_uid_list is not None:
            train_df = train_df[train_df['uid'].isin(valid_uid_list)]
            feature_df = feature_df[feature_df['uid'].isin(valid_uid_list)]

        # dynamic_df 是動態特徵，static_df 是靜態特徵
        dynamic_df = train_df
        static_df = feature_df
        df = dynamic_df.merge(static_df, on='uid', how='left')

        X = df[feature_list].copy()
        y = None
        if 'x' in feature_list and 'y' in feature_list:
            y = pd.DataFrame({
                'uid': df['uid'],
                'd': df['d'],
                't': df['t'],
                'x': df['x'],
                'y': df['y']
            })
            X = X.drop(columns=['x', 'y'])
        return X, y

    def Per_user_RF(self, X, y, n_estimators=100, random_state=42, output_name='CityName'):
        """
        X: 特徵 DataFrame
        y: DataFrame with columns ['uid', 'd', 't', 'x', 'y']
        n_estimators: 樹數量
        每個 uid 都訓練一個 RF regressor 並儲存到 ./ckpt
        """
        os.makedirs(f'./ckpt/RF/{output_name}', exist_ok=True)

        uid_list = X['uid'].unique()
        total = len(uid_list)
        for i, uid in enumerate(uid_list):
            X_uid = X[X['uid'] == uid].copy()
            y_uid = y[y['uid'] == uid][['x', 'y']]
            reg = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, warm_start=True)
            reg.fit(X_uid.drop(columns=['uid']), y_uid)
            joblib.dump(reg, f'./ckpt/RF/{output_name}/rf_uid_{uid}.pkl')
            print(f'訓練進度 {i + 1}/{total}: uid {uid}', end='\r')
        print(f'\n訓練完成，模型已儲存到 ./ckpt/RF/{output_name}/')

    def predict_per_user_RF(self, test_df, feature_df, feature_list, valid_uid_list=None, output_name='CityName'):
        if valid_uid_list is not None:
            test_df = test_df[test_df['uid'].isin(valid_uid_list)]
            feature_df = feature_df[feature_df['uid'].isin(valid_uid_list)]

        # 看有考慮那些特徵
        # dynamic_df 是動態特徵，static_df 是靜態特徵
        dynamic_df = test_df
        static_df = feature_df
        df = dynamic_df.merge(static_df, on='uid', how='left')
        test_df = df[feature_list].copy()
        test_df = test_df.drop(columns=['x', 'y'])

        results = []
        uid_list = test_df['uid'].unique()
        total = len(uid_list)
        for idx, uid in enumerate(uid_list):
            raw_uid = df[df['uid'] == uid].copy()
            X_uid = test_df[test_df['uid'] == uid].copy()
            reg = joblib.load(f'./ckpt/RF/{output_name}/rf_uid_{uid}.pkl')
            X_uid_input = X_uid.drop(columns=['uid'])
            y_pred = reg.predict(X_uid_input)
            for i, row in raw_uid.iterrows():
                results.append({
                    'uid': row['uid'],
                    'd': row['d'],
                    't': row['t'],
                    'x': int(round(y_pred[i - X_uid.index[0], 0])),
                    'y': int(round(y_pred[i - X_uid.index[0], 1]))
                })
            print(f'預測進度 {idx + 1}/{total}: uid {uid}', end='\r')

        results = pd.DataFrame(results)
        output_df = results[['uid', 'd', 't', 'x', 'y']].astype(int)
        os.makedirs(f'./Predictions/RF', exist_ok=True)
        output_df.to_csv(f'./Predictions/RF/{output_name}_Per_user_RF.csv', index=False)
        print(f'\n預測完成，結果已儲存到 ./Predictions/RF/{output_name}_Per_user_RF.csv')
        return output_df

    def Per_user_XGB(self, X, y, n_estimators=100, random_state=42, output_name='CityName', use_gpu=False):
        os.makedirs(f'./ckpt/XGB/{output_name}', exist_ok=True)
        uid_list = X['uid'].unique()
        total = len(uid_list)
        for i, uid in enumerate(uid_list):
            X_uid = X[X['uid'] == uid].copy()
            y_uid = y[y['uid'] == uid][['x', 'y']]
            if use_gpu:
                reg_x = xgb.XGBRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, tree_method='gpu_hist')
                reg_y = xgb.XGBRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, tree_method='gpu_hist')
            else:   
                reg_x = xgb.XGBRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
                reg_y = xgb.XGBRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
            reg_x.fit(X_uid.drop(columns=['uid']), y_uid['x'])
            reg_y.fit(X_uid.drop(columns=['uid']), y_uid['y'])
            joblib.dump((reg_x, reg_y), f'./ckpt/XGB/{output_name}/xgb_uid_{uid}.pkl')
            print(f'訓練進度 {i + 1}/{total}: uid {uid}', end='\r')
        print(f'\n訓練完成，模型已儲存到 ./ckpt/XGB/{output_name}/')

    def plot_importance_per_user_XGB(self, uid, output_name='CityName', max_num_features=10):
        """
        視覺化指定 uid 的 XGBoost 模型特徵重要性
        uid: 指定的使用者 ID
        output_name: 模型存檔名稱
        max_num_features: 顯示前幾個重要特徵
        """
        model_path = f'./ckpt/XGB/{output_name}/xgb_uid_{uid}.pkl'
        if not os.path.exists(model_path):
            print(f"模型檔案不存在: {model_path}")
            return
        reg_x, reg_y = joblib.load(model_path)
        xgb.plot_importance(reg_x, max_num_features=max_num_features)
        plt.title(f"XGB Feature Importance (uid={uid})")
        plt.show()
    
    def predict_per_user_XGB(self, test_df, feature_df, feature_list, valid_uid_list=None, output_name='CityName'):
        if valid_uid_list is not None:
            test_df = test_df[test_df['uid'].isin(valid_uid_list)]
            feature_df = feature_df[feature_df['uid'].isin(valid_uid_list)]
        dynamic_df = test_df
        static_df = feature_df
        df = dynamic_df.merge(static_df, on='uid', how='left')
        test_df = df[feature_list].copy()
        test_df = test_df.drop(columns=['x', 'y'])
        results = []
        uid_list = test_df['uid'].unique()
        total = len(uid_list)
        for idx, uid in enumerate(uid_list):
            raw_uid = df[df['uid'] == uid].copy()
            X_uid = test_df[test_df['uid'] == uid].copy()
            reg_x, reg_y = joblib.load(f'./ckpt/XGB/{output_name}/xgb_uid_{uid}.pkl')
            X_uid_input = X_uid.drop(columns=['uid'])
            y_pred_x = reg_x.predict(X_uid_input)
            y_pred_y = reg_y.predict(X_uid_input)
            for i, row in raw_uid.iterrows():
                results.append({
                    'uid': row['uid'],
                    'd': row['d'],
                    't': row['t'],
                    'x': int(round(y_pred_x[i - X_uid.index[0]])),
                    'y': int(round(y_pred_y[i - X_uid.index[0]]))
                })
            print(f'預測進度 {idx + 1}/{total}: uid {uid}', end='\r')
        results = pd.DataFrame(results)
        output_df = results[['uid', 'd', 't', 'x', 'y']].astype(int)
        os.makedirs(f'./Predictions/XGB', exist_ok=True)
        output_df.to_csv(f'./Predictions/XGB/{output_name}_Per_user_XGB.csv', index=False)
        print(f'\n預測完成，結果已儲存到 ./Predictions/XGB/{output_name}_Per_user_XGB.csv')
        return output_df

    def Per_user_LightGBM(self, X, y, n_estimators=100, random_state=42, output_name='CityName', use_gpu=False):
        os.makedirs(f'./ckpt/LGBM/{output_name}', exist_ok=True)
        uid_list = X['uid'].unique()
        total = len(uid_list)
        for i, uid in enumerate(uid_list):
            X_uid = X[X['uid'] == uid].copy()
            y_uid = y[y['uid'] == uid][['x', 'y']]
            if use_gpu:
                reg_x = lgb.LGBMRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, device='gpu', verbosity=-1)
                reg_y = lgb.LGBMRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, device='gpu', verbosity=-1)
            else:
                reg_x = lgb.LGBMRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, verbosity=-1)
                reg_y = lgb.LGBMRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, verbosity=-1)
            reg_x.fit(X_uid.drop(columns=['uid']), y_uid['x'])
            reg_y.fit(X_uid.drop(columns=['uid']), y_uid['y'])
            joblib.dump((reg_x, reg_y), f'./ckpt/LGBM/{output_name}/lgbm_uid_{uid}.pkl')
            print(f'訓練進度 {i + 1}/{total}: uid {uid}', end='\r')
        print(f'\n訓練完成，模型已儲存到 ./ckpt/LGBM/{output_name}/')

    def predict_per_user_LightGBM(self, test_df, feature_df, feature_list, valid_uid_list=None, output_name='CityName'):
        if valid_uid_list is not None:
            test_df = test_df[test_df['uid'].isin(valid_uid_list)]
            feature_df = feature_df[feature_df['uid'].isin(valid_uid_list)]
        dynamic_df = test_df
        static_df = feature_df
        df = dynamic_df.merge(static_df, on='uid', how='left')
        test_df = df[feature_list].copy()
        test_df = test_df.drop(columns=['x', 'y'])
        results = []
        uid_list = test_df['uid'].unique()
        total = len(uid_list)
        for idx, uid in enumerate(uid_list):
            raw_uid = df[df['uid'] == uid].copy()
            X_uid = test_df[test_df['uid'] == uid].copy()
            reg_x, reg_y = joblib.load(f'./ckpt/LGBM/{output_name}/lgbm_uid_{uid}.pkl')
            X_uid_input = X_uid.drop(columns=['uid'])
            y_pred_x = reg_x.predict(X_uid_input)
            y_pred_y = reg_y.predict(X_uid_input)
            for i, row in raw_uid.iterrows():
                results.append({
                    'uid': row['uid'],
                    'd': row['d'],
                    't': row['t'],
                    'x': int(round(y_pred_x[i - X_uid.index[0]])),
                    'y': int(round(y_pred_y[i - X_uid.index[0]]))
                })
            print(f'預測進度 {idx + 1}/{total}: uid {uid}', end='\r')
        results = pd.DataFrame(results)
        output_df = results[['uid', 'd', 't', 'x', 'y']].astype(int)
        os.makedirs(f'./Predictions/LGBM', exist_ok=True)
        output_df.to_csv(f'./Predictions/LGBM/{output_name}_Per_user_LGBM.csv', index=False)
        print(f'\n預測完成，結果已儲存到 ./Predictions/LGBM/{output_name}_Per_user_LGBM.csv')
        return output_df

    def Evaluation(self, generated_data_input, reference_data_input, validator=False, city_name=None, raw_data_path=None):
        # 檢查生成的資料是否符合規範
        if validator:
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

            print(f"{idx+1}/{len(valid_uid_list)}人--uid={uid}", end='\r')

        final_GEOBLEU_score = sum(GEOBLEU_scores) / len(GEOBLEU_scores) if GEOBLEU_scores else 0.0
        final_DTW_score = sum(DTW_scores) / len(DTW_scores) if DTW_scores else 0.0
        print(f'\nGEOBLEU平均分數: {final_GEOBLEU_score:.4f}, DTW平均分數: {final_DTW_score:.4f}')
        return final_GEOBLEU_score, final_DTW_score

"""
測試程式碼
"""
if __name__ == "__main__":
    model = ModelZoo()
    raw_std_df = pd.read_csv('./Stability/A_xtrain_working_day_stability.csv')
    train_df = pd.read_csv('./Training_Testing_Data/A_x_train.csv')
    test_df = pd.read_csv('./Training_Testing_Data/A_x_test.csv')
    feature_df = pd.read_csv('./Stability/A_activity_space.csv')
    feature_list = ['uid', 't', 'x', 'y','working_day']

    # model.plot_importance_per_user_XGB(uid=216, output_name='A', max_num_features=10)

    # LightGBM
    thresholds = [0,9999]
    for i in range(len(thresholds) - 1):
        lower = thresholds[i]
        upper = thresholds[i + 1]
        filter_std_df = raw_std_df[(raw_std_df['x_std_mean'] >= lower) | (raw_std_df['y_std_mean'] >= lower)]
        valid_uid_list = filter_std_df[(filter_std_df['x_std_mean'] < upper) & (filter_std_df['y_std_mean'] < upper)]['uid'].unique()
        if len(valid_uid_list) > 10000:
            valid_uid_list = valid_uid_list[:10000]
        print(f"x|y std >= {lower},x&y std < {upper} 有效的使用者ID數量: {len(valid_uid_list)}")
        # DataPreparation
        X, y = model.DataPreparation(train_df, feature_df, feature_list, valid_uid_list)
        # Train
        model.Per_user_LightGBM(X, y, n_estimators=100, random_state=42, output_name='A')
        # Predict
        predictions = model.predict_per_user_LightGBM(test_df, 
                                                feature_df=feature_df, 
                                                feature_list=feature_list,
                                                valid_uid_list=valid_uid_list, 
                                                output_name='A')
        # Evaluation
        generated_data_input = predictions
        reference_data_input = pd.read_csv('./Training_Testing_Data/A_x_test.csv')
        final_GEOBLEU_score, final_DTW_score = model.Evaluation(generated_data_input, reference_data_input)


    # # XGB
    # thresholds = [0, 1, 2, 3, 4, 5, 10, 9999]
    # max_len = 3000
    # for i in range(len(thresholds) - 1):
    #     lower = thresholds[i]
    #     upper = thresholds[i + 1]
    #     filter_std_df = raw_std_df[(raw_std_df['x_std_mean'] >= lower) | (raw_std_df['y_std_mean'] >= lower)]
    #     valid_uid_list = filter_std_df[(filter_std_df['x_std_mean'] < upper) & (filter_std_df['y_std_mean'] < upper)]['uid'].unique()
    #     if len(valid_uid_list) > max_len:
    #         valid_uid_list = valid_uid_list[:max_len]
    #     print(f"x|y std >= {lower},x&y std < {upper} 有效的使用者ID數量: {len(valid_uid_list)}")
    #     # DataPreparation
    #     X, y = model.DataPreparation(train_df, feature_df, feature_list, valid_uid_list)
    #     # Train
    #     model.Per_user_XGB(X, y, n_estimators=100, random_state=42, output_name='A')
    #     # Predict
    #     predictions = model.predict_per_user_XGB(test_df, 
    #                                             feature_df=feature_df, 
    #                                             feature_list=feature_list,
    #                                             valid_uid_list=valid_uid_list, 
    #                                             output_name='A')
    #     # Evaluation
    #     generated_data_input = predictions
    #     reference_data_input = pd.read_csv('./Training_Testing_Data/A_x_test.csv')
    #     final_GEOBLEU_score, final_DTW_score = model.Evaluation(generated_data_input, reference_data_input)


    # # RF
    # thresholds = [0, 1, 2, 3, 4, 5, 10, 9999]
    # for i in range(len(thresholds) - 1):
    #     lower = thresholds[i]
    #     upper = thresholds[i + 1]
    #     filter_std_df = raw_std_df[(raw_std_df['x_std_mean'] >= lower) | (raw_std_df['y_std_mean'] >= lower)]
    #     valid_uid_list = filter_std_df[(filter_std_df['x_std_mean'] < upper) & (filter_std_df['y_std_mean'] < upper)]['uid'].unique()
    #     if len(valid_uid_list) > 3000:
    #         valid_uid_list = valid_uid_list[:3000]
    #     print(f"x|y std >= {lower},x&y std < {upper} 有效的使用者ID數量: {len(valid_uid_list)}")
    #     # DataPreparation
    #     X, y = model.DataPreparation(train_df, feature_df, feature_list, valid_uid_list)
    #     # Train
    #     model.Per_user_RF(X, y, n_estimators=100, random_state=42, output_name='A')
    #     # Predict
    #     predictions = model.predict_per_user_RF(test_df, 
    #                                             feature_df=feature_df, 
    #                                             feature_list=feature_list,
    #                                             valid_uid_list=valid_uid_list, 
    #                                             output_name='A')
    #     # Evaluation
    #     generated_data_input = predictions
    #     reference_data_input = pd.read_csv('./Training_Testing_Data/A_x_test.csv')
    #     final_GEOBLEU_score, final_DTW_score = model.Evaluation(generated_data_input, reference_data_input)