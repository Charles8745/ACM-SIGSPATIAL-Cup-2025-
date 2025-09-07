import os
import numpy as np
import pandas as pd
import geobleu
import validator_InModify as validator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 或 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正確顯示負號
"""
用於比較45vs15的mode和生成geo_bleu分數
"""

cluster_list = [-1, 0, 1, 2, 3, 4, 5]

test_df = pd.read_csv(f'./Training_Testing_Data/A_y_train.csv')
gt_df = test_df[test_df['d'] > 45]
def Evaluation(generated_data_input, reference_data_input, valid=False, city_name=None, raw_data_path=None):
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
Mode3000人分數
"""
mode_result = []
for cluster in cluster_list:
    cluster_df = pd.read_csv(f'./Predictions/A_y_cluster{cluster}_modify_Per_User_Per_t_Mode_working_day_modify.csv')
    mode_result.append(cluster_df)

mode_result_df = pd.concat(mode_result, ignore_index=True).sort_values(by=['uid', 'd', 't']).reset_index(drop=True)
mode_result_df.to_csv(f'./Predictions/A_y_cluster_all_modify_Per_User_Per_t_Mode_working_day_modify.csv', index=False)
valid_uid_list = mode_result_df['uid'].unique()
print(f'要檢查的UID數量: {len(valid_uid_list)}')
gt_df = gt_df[gt_df['uid'].isin(valid_uid_list)]

final_GEOBLEU_score, final_DTW_score = Evaluation(
    generated_data_input=mode_result_df,
    reference_data_input=gt_df,
)
print(f"Mode GEO-BLEU分數: {final_GEOBLEU_score:.4f}, DTW分數: {final_DTW_score:.4f}\n\n")

"""
cluster被拆分的部分整合
"""
# cluster_result = []
# for batch_idx in range(10):
#     batch_df = pd.read_csv(f'./Predictions/CVAE/cvae_model_class_h1400l1400uid180layers1_cluster4_{batch_idx+1}_cityA.csv')
#     cluster_result.append(batch_df)

# cluster_result_df = pd.concat(cluster_result, ignore_index=True).sort_values(by=['uid', 'd', 't']).reset_index(drop=True)
# cluster_result_df.to_csv(f'./Predictions/CVAE/class45vs15/cvae_model_class_cluster4_cityA.csv', index=False)

"""
生成模型3000人分數
"""
gen_result = []
for cluster in cluster_list:
    try:
        cluster_df = pd.read_csv(f'./Predictions/CVAE/class45vs15/cvae_model_class_cluster{cluster}_cityA.csv')
        gen_result.append(cluster_df)
        print(f'cluster{cluster} 使用生成數據', f'占比:{cluster_df.shape[0] / gt_df.shape[0]:.2f}')

    except:
        cluster_df = pd.read_csv(f'./Predictions/A_y_cluster{cluster}_modify_Per_User_Per_t_Mode_working_day_modify.csv')
        gen_result.append(cluster_df)
        print(f'cluster{cluster} 使用眾數數據', f'占比:{cluster_df.shape[0] / gt_df.shape[0]:.2f}')

gen_result_df = pd.concat(gen_result, ignore_index=True).sort_values(by=['uid', 'd', 't']).reset_index(drop=True)
valid_uid_list = gen_result_df['uid'].unique()
print(f'要檢查的UID數量: {len(valid_uid_list)}')
gt_df = gt_df[gt_df['uid'].isin(valid_uid_list)]

final_GEOBLEU_score, final_DTW_score = Evaluation(
    generated_data_input=gen_result_df,
    reference_data_input=gt_df,
)
print(f"生成 GEO-BLEU分數: {final_GEOBLEU_score:.4f}, DTW分數: {final_DTW_score:.4f}\n\n")