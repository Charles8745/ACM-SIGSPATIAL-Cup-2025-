import os
import numpy as np
import pandas as pd
import validator_InModify as validator
import geobleu
import time
import random

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
            x_user_global_mode = user_df['x'].mode()
            y_user_global_mode = user_df['y'].mode()
            for t in np.arange(0, 48):
                t_df = user_df[user_df['t'] == t]
                if not t_df.empty:
                    x_mode = t_df['x'].mode().values[0]
                    y_mode = t_df['y'].mode().values[0]
                else:
                    x_mode = x_user_global_mode.values[0]
                    y_mode = y_user_global_mode.values[0]
                result.append({'uid': uid, 't': t, 'x': x_mode, 'y': y_mode})
            print(f'訓練進度: {i+1}/{len(valid_uid_list)} 使用者ID={uid}', end='\r')

        train_mode_df = pd.DataFrame(result)

        # 依據測試資料的時間點，將模式應用到測試資料
        result = []
        for i, uid in enumerate(valid_uid_list):
            train_user_df = train_mode_df[train_mode_df['uid'] == uid]
            test_user_df = self.test_data[self.test_data['uid'] == uid]
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

        train_mode_df.to_csv(f'./ckpt/{output_name}_per_user_per_t_mode.csv', index=False)
        print(f"Train_Mode: 結果已儲存至 ./ckpt/{output_name}_per_user_per_t_mode.csv")
        prediction_df.to_csv(f'./Predictions/{output_name}_per_user_per_t_mode.csv', index=False)
        print(f"Per_User_Per_t_Mode: 結果已儲存至 ./Predictions/{output_name}_per_user_per_t_mode.csv")

        elapsed_time = time.time() - start_time
        print(f"Per_User_Per_t_Mode: 執行時間: {elapsed_time//60:.2f}min")
        return prediction_df

    def Evaluation(self, generated_data_input, reference_data_input ,validator=False, city_name=None, raw_data_path=None):
        # 檢查生成的資料是否符合規範
        if validator:
            validator.main(city_name, raw_data_path, generated_data_input)

        # 讀取生成與參考資料
        print(f'讀取生成資料: {generated_data_input}')
        print(f'讀取參考資料: {reference_data_input}')
        generated_df = pd.read_csv(generated_data_input, header=0, dtype=int)
        reference_df = pd.read_csv(reference_data_input, header=0, dtype=int)
        valid_uid_list = generated_df['uid'].unique()

        
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

    # 不同std分類對分數影響-->Per_User_Per_t_Mode
    raw_train_data_df = pd.read_csv('./Training_Testing_Data/A_x_train.csv', header=0)
    raw_test_data_df = pd.read_csv('./Training_Testing_Data/A_x_test.csv', header=0)
    raw_std_df = pd.read_csv('./Stability/A_xtrain_working_day_stability.csv', header=0)

    std_model_zoo = ModelZoo(raw_train_data_df, raw_test_data_df)

    thresholds = [0, 1, 2, 3, 4, 5, 10, 9999]
    for i in range(len(thresholds) - 1):
        lower = thresholds[i]
        upper = thresholds[i + 1]

        filter_std_df = raw_std_df[(raw_std_df['x_std_mean'] >= lower) | (raw_std_df['y_std_mean'] >= lower)]
        valid_uid_list = filter_std_df[(filter_std_df['x_std_mean'] < upper) & (filter_std_df['y_std_mean'] < upper)]['uid'].unique()
        print(f"x|y std >= {lower},x&y std < {upper} 有效的使用者ID數量: {len(valid_uid_list)}")

        std_model_zoo.Per_User_Per_t_Mode(valid_uid_list=valid_uid_list, output_name=f'A_std{upper}', early_stop=3000)

        final_GEOBLEU_score, final_DTW_score = std_model_zoo.Evaluation(
            generated_data_input=f'./Predictions/A_std{upper}_per_user_per_t_mode.csv',
            reference_data_input='./Training_Testing_Data/A_x_test.csv',
        )
        print(f"最終GEO-BLEU分數: {final_GEOBLEU_score:.4f}, 最終DTW分數: {final_DTW_score:.4f}\n\n")

    # # 不同dtw分類對分數影響-->Per_User_Per_t_Mode
    # raw_train_data_df = pd.read_csv('./Training_Testing_Data/A_x_train.csv', header=0)
    # raw_test_data_df = pd.read_csv('./Training_Testing_Data/A_x_test.csv', header=0)
    # raw_dtw_df = pd.read_csv('./Stability/A_xtrain_working_day_dtw.csv', header=0)

    # dtw_model_zoo = ModelZoo(raw_train_data_df, raw_test_data_df)

    # thresholds = [0, 10, 30, 50, 9999]
    # for i in range(len(thresholds) - 1):
    #     lower = thresholds[i]
    #     upper = thresholds[i + 1]

    #     filter_dtw_df = raw_dtw_df[(raw_dtw_df['dtw_mean'] >= lower)]
    #     valid_uid_list = filter_dtw_df[(filter_dtw_df['dtw_mean'] < upper)]['uid'].unique()
    #     print(f"dtw_mean >= {lower}, dtw_mean < {upper} 有效的使用者ID數量: {len(valid_uid_list)}")

    #     dtw_model_zoo.Per_User_Per_t_Mode(valid_uid_list=valid_uid_list, output_name=f'A_dtw{upper}', early_stop=3000)

    #     final_GEOBLEU_score, final_DTW_score = dtw_model_zoo.Evaluation(
    #         generated_data_input=f'./Predictions/A_dtw{upper}_per_user_per_t_mode.csv',
    #         reference_data_input='./Training_Testing_Data/A_x_test.csv',
    #     )
    #     print(f"最終GEO-BLEU分數: {final_GEOBLEU_score:.4f}, 最終DTW分數: {final_DTW_score:.4f}\n\n")

    # # 不同Geweke Diagnostic分類對分數影響-->Per_User_Per_t_Mode
    # raw_train_data_df = pd.read_csv('./Training_Testing_Data/A_x_train.csv', header=0)
    # raw_test_data_df = pd.read_csv('./Training_Testing_Data/A_x_test.csv', header=0)
    # raw_geweke_df = pd.read_csv('./Stability/A_uid_geweke_ones_ratio.csv', header=0)
    # raw_geweke_df = raw_geweke_df[raw_geweke_df['uid']<=147000]

    # geweke_model_zoo = ModelZoo(raw_train_data_df, raw_test_data_df)

    # thresholds = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    # for i in range(len(thresholds) - 1):
    #     lower = thresholds[i]
    #     upper = thresholds[i + 1]

    #     filter_geweke_df = raw_geweke_df[(raw_geweke_df['ones_ratio'] >= lower)]
    #     valid_uid_list = filter_geweke_df[(filter_geweke_df['ones_ratio'] < upper)]['uid'].unique()
    #     print(f"ones_ratio >= {lower}, ones_ratio < {upper} 有效的使用者ID數量: {len(valid_uid_list)}")

    #     geweke_model_zoo.Per_User_Per_t_Mode(valid_uid_list=valid_uid_list, output_name=f'A_geweke{upper}', early_stop=5000)

    #     final_GEOBLEU_score, final_DTW_score = geweke_model_zoo.Evaluation(
    #         generated_data_input=f'./Predictions/A_geweke{upper}_per_user_per_t_mode.csv',
    #         reference_data_input='./Training_Testing_Data/A_x_test.csv',
    #     )
    #     print(f"最終GEO-BLEU分數: {final_GEOBLEU_score:.4f}, 最終DTW分數: {final_DTW_score:.4f}\n\n")
