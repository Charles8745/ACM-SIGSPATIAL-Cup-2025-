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
1. 採用非線性權重，若大於閥值則趨向生成結果，否則更趨向眾數結果
"""

# 分別讀取眾數結果和生成結果
mode_pred_df = pd.read_csv('./Predictions/CVAE/cvae_model_class_h1024l1024uid40layers1_cluster2_cityA.csv')
gen_pred_df = pd.read_csv('./Predictions/CVAE/cvae_model_reg_h2048l2048uid40layers3_cluster2_cityA.csv')


valid_uid_list = mode_pred_df['uid'].unique().tolist()
test_df = pd.read_csv(f'./Training_Testing_Data/A_y_train.csv')
gt_df = test_df[test_df['d'] > 45]
# gt_df = pd.read_csv('./Training_Testing_Data/A_x_test.csv')
gt_df = gt_df[gt_df['uid'].isin(valid_uid_list)]

# 依據輸出結果做ensemble
def sigmoid_weight(dist, threshold=10, k=1):
    # k 控制曲線斜率，threshold 控制轉折點
    # 當 dist = threshold 時，weight 約為 0.5
    return 1 / (1 + np.exp(-k * (dist - threshold)))

results = []
threshold = 10
k = 15 # k 越大，曲線越陡，權重會在接近 threshold 時快速從 0 跳到 1（類似階梯函數）。
for mode_row, gen_row in zip(mode_pred_df.itertuples(index=False), gen_pred_df.itertuples(index=False)):
    mode_x, mode_y = mode_row.x, mode_row.y
    gen_x, gen_y = gen_row.x, gen_row.y
    dist = np.sqrt((mode_x - gen_x) ** 2 + (mode_y - gen_y) ** 2)
    weight = sigmoid_weight(dist, threshold, k)
    ensemble_x = int(round(mode_x * (1 - weight) + gen_x * weight))
    ensemble_y = int(round(mode_y * (1 - weight) + gen_y * weight))
    result = mode_row._asdict()
    result['x'] = ensemble_x
    result['y'] = ensemble_y
    results.append(result)

ensemble_df = pd.DataFrame(results)

# # 畫出 sigmoid 權重曲線
# distances = np.linspace(0, 30, 300)
# plt.figure(figsize=(8, 5))
# for k_val in [0.5, 1, 2, 5, 10]:
#     weights = [sigmoid_weight(d, threshold=10, k=k_val) for d in distances]
#     plt.plot(distances, weights, label=f'k={k_val}')
# plt.axvline(threshold, color='gray', linestyle='--', label=f'閾值={threshold}')
# plt.xlabel('歐式距離')
# plt.ylabel('生成結果權重')
# plt.title('不同k值的Sigmoid加權曲線')
# plt.legend()
# plt.grid(True)
# plt.show()

# 儲存ensemble結果
os.makedirs('./Predictions/Ensemble', exist_ok=True)
ensemble_df.to_csv('./Predictions/Ensemble/A_x_ensemble_pred_cluster1.csv', index=False)

# 計算 geobleu 分數
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

final_GEOBLEU_score, final_DTW_score = Evaluation(
generated_data_input = f'./Predictions/Ensemble/A_x_ensemble_pred_cluster1.csv',
reference_data_input = gt_df,
)
print(f"最終GEO-BLEU分數: {final_GEOBLEU_score:.4f}, 最終DTW分數: {final_DTW_score:.4f}\n\n")

# 把GT, mode, cvae的x,y拉出來看時間線段上的重合性
def plot_x_y_sequence_compare(uid, mode_df, cvae_df, gt_df):
    # 依照時間排序
    mode_user = mode_df[mode_df['uid'] == uid].sort_values(['d', 't'])
    cvae_user = cvae_df[cvae_df['uid'] == uid].sort_values(['d', 't'])
    gt_user = gt_df[gt_df['uid'] == uid].sort_values(['d', 't'])

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)

    # 左上：x的mode和gt
    axes[0, 0].plot(mode_user['x'].values, '-o', label='Mode', color='red', alpha=0.7)
    axes[0, 0].plot(gt_user['x'].values, '-o', label='GT', color='green', alpha=0.7)
    axes[0, 0].set_title(f'UID {uid} x 時序 (Mode vs GT)')
    axes[0, 0].set_ylabel('x')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 左下：x的cvae和gt
    axes[1, 0].plot(cvae_user['x'].values, '-o', label='CVAE', color='blue', alpha=0.7)
    axes[1, 0].plot(gt_user['x'].values, '-o', label='GT', color='green', alpha=0.7)
    axes[1, 0].set_title(f'UID {uid} x 時序 (CVAE vs GT)')
    axes[1, 0].set_xlabel('時間點')
    axes[1, 0].set_ylabel('x')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 右上：y的mode和gt
    axes[0, 1].plot(mode_user['y'].values, '-o', label='Mode', color='red', alpha=0.7)
    axes[0, 1].plot(gt_user['y'].values, '-o', label='GT', color='green', alpha=0.7)
    axes[0, 1].set_title(f'UID {uid} y 時序 (Mode vs GT)')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 右下：y的cvae和gt
    axes[1, 1].plot(cvae_user['y'].values, '-o', label='CVAE', color='blue', alpha=0.7)
    axes[1, 1].plot(gt_user['y'].values, '-o', label='GT', color='green', alpha=0.7)
    axes[1, 1].set_title(f'UID {uid} y 時序 (CVAE vs GT)')
    axes[1, 1].set_xlabel('時間點')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

valid_uid_list = mode_pred_df['uid'].unique().tolist()
for i in range(20):
    plot_x_y_sequence_compare(uid=valid_uid_list[i],
                                mode_df=mode_pred_df,
                                cvae_df=ensemble_df,
                                gt_df=gt_df)