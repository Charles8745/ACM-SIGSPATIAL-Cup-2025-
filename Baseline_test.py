import pandas as pd
import validator_InModify as validator
import geobleu
import os
import numpy as np

class BaselineTest:
    def __init__(self, test_data_input, train_data_input, output_name):
        os.mkdir('./Predictions') if not os.path.exists('Predictions') else None
        self.output_name = output_name

        # 讀取測試資料
        if isinstance(test_data_input, pd.DataFrame):
            self.test_df = test_data_input
        elif isinstance(test_data_input, str):
            self.test_df = pd.read_csv(test_data_input, header=0, dtype=int)
        else:
            raise ValueError("只能接受DataFrame或資料路徑字串（csv檔）。")

        # 讀取訓練資料
        if isinstance(train_data_input, pd.DataFrame):
            self.train_df = train_data_input
        elif isinstance(train_data_input, str):
            self.train_df = pd.read_csv(train_data_input, header=0, dtype=int)
        else:
            raise ValueError("只能接受DataFrame或資料路徑字串（csv檔）。")

        # 合併所有 uid，若超過3000人則只取後3000人
        all_uids = sorted(set(self.test_df['uid']) | set(self.train_df['uid']))
        if len(all_uids) > 3000:
            selected_uids = set(all_uids[-3000:])
            self.test_df = self.test_df[self.test_df['uid'].isin(selected_uids)].reset_index(drop=True)
            self.train_df = self.train_df[self.train_df['uid'].isin(selected_uids)].reset_index(drop=True)
        else:
            selected_uids = set(all_uids)

        # 印出資料資訊
        print(f"測試資料共有{self.test_df.shape[0]}筆資料\n"
              f"資料範圍: uid={self.test_df['uid'].min()}~{self.test_df['uid'].max()}\n"
              f"時間範圍: days={self.test_df['d'].min()}~{self.test_df['d'].max()}\n")
        print(f"訓練資料共有{self.train_df.shape[0]}筆資料\n"
              f"資料範圍: uid={self.train_df['uid'].min()}~{self.train_df['uid'].max()}\n"
              f"時間範圍: days={self.train_df['d'].min()}~{self.train_df['d'].max()}\n")

    def Global_Mean(self):                  
        """
        Calculate the "global mean" as the average of staypoints in days 1 to 60 of all users' trajectories, 
        and for days 61 to 75, predict the trajectory as if the user continues to stay at that point.
        """
        Global_Mean_test_df = self.test_df.copy()
        Global_Mean_train_df = self.train_df.copy()

        # 計算x和y的Global Mean
        x_mean = Global_Mean_train_df['x'].mean()
        y_mean = Global_Mean_train_df['y'].mean()
        print(f"Global Mean: x={x_mean}, y={y_mean}")

        # Predict days 61 to 75
        Global_Mean_test_df['x'] = x_mean
        Global_Mean_test_df['y'] = y_mean

        # 只輸出指定欄位
        output_df = Global_Mean_test_df[['uid', 'd', 't', 'x', 'y']].astype(int)
        output_df.to_csv(f'./Predictions/{self.output_name}_global_mean.csv', index=False)
        print(f"Global Mean預測完成，結果已儲存至./Predictions/{self.output_name}_global_mean.csv\n")

    def Global_Mode(self):
        """
        Determine the "global mode" as the most frequent staypoint in days 1 to 60 of all users' trajectories, 
        and for days 61 to 75, predict the trajectory as if the user continues to stay at that point.
        """
        Global_Mode_test_df = self.test_df.copy()
        Global_Mode_train_df = self.train_df.copy()

        # 計算x和y的Global Mode
        x_mode = Global_Mode_train_df['x'].mode()[0]
        y_mode = Global_Mode_train_df['y'].mode()[0]
        print(f"Global Mode: x={x_mode}, y={y_mode}")

        # Predict days 61 to 75
        Global_Mode_test_df['x'] = x_mode
        Global_Mode_test_df['y'] = y_mode

        # 儲存預測結果
        output_df = Global_Mode_test_df[['uid', 'd', 't', 'x', 'y']].astype(int)
        output_df.to_csv(f'./Predictions/{self.output_name}_global_mode.csv', index=False)
        print(f"Global Mode預測完成，結果已儲存至./Predictions/{self.output_name}_global_mode.csv\n")

    def Per_User_Mean(self):
        """
        Calculate the "per-user mean" as the average of staypoints in days 1 to 60 of a given user's trajectory, 
        and for days 61 to 75, predict the trajectory as if the user continues to stay at that point.
        """
        Per_User_Mean_test_df = self.test_df.copy()
        Per_User_Mean_train_df = self.train_df.copy()

        # 計算每個使用者的x和y的Per User Mean
        user_means = Per_User_Mean_train_df.groupby('uid')[['x', 'y']].mean().reset_index()

        # Predict days 61 to 75
        for uid in user_means['uid']:
            mean_x = user_means[user_means['uid'] == uid]['x'].values[0]
            mean_y = user_means[user_means['uid'] == uid]['y'].values[0]
            Per_User_Mean_test_df.loc[(Per_User_Mean_test_df['uid'] == uid), 'x'] = mean_x
            Per_User_Mean_test_df.loc[(Per_User_Mean_test_df['uid'] == uid), 'y'] = mean_y

        # 儲存預測結果
        Per_User_Mean_test_df = Per_User_Mean_test_df[['uid', 'd', 't', 'x', 'y']].astype(int)
        Per_User_Mean_test_df.to_csv(f'./Predictions/{self.output_name}_per_user_mean.csv', index=False)
        print(f"Per User Mean預測完成，結果已儲存至./Predictions/{self.output_name}_per_user_mean.csv\n")
    
    def Per_User_Mode(self):
        """
        Determine the "per-user mode" as the most frequent staypoint in days 1 to 60 of a given user's trajectory, 
        and for days 61 to 75, predict the trajectory as if the user continues to stay at that point.
        """
        Per_User_Mode_test_df = self.test_df.copy()
        Per_User_Mode_train_df = self.train_df.copy()

        # 計算每個使用者的x和y的Per User Mode
        user_modes = Per_User_Mode_train_df.groupby('uid').agg({
            'x': lambda x: x.mode()[0],
            'y': lambda y: y.mode()[0]
        }).reset_index()

        # Predict days 61 to 75
        for _, row in user_modes.iterrows():
            uid = row['uid']
            mode_x = row['x']
            mode_y = row['y']
            Per_User_Mode_test_df.loc[Per_User_Mode_test_df['uid'] == uid, 'x'] = mode_x
            Per_User_Mode_test_df.loc[Per_User_Mode_test_df['uid'] == uid, 'y'] = mode_y

        # 儲存預測結果
        Per_User_Mode_test_df = Per_User_Mode_test_df[['uid', 'd', 't', 'x', 'y']].astype(int)
        Per_User_Mode_test_df.to_csv(f'./Predictions/{self.output_name}_per_user_mode.csv', index=False)
        print(f"Per User Mode預測完成，結果已儲存至./Predictions/{self.output_name}_per_user_mode.csv\n")

    def Unigram_model(self):
        """
        Unigram Model: For a given user, create a unigram model of staypoints from days 1 to 60 of their trajectory, 
        and use it to predict the trajectory in days 61 to 75.
        """
        Unigram_test_df = self.test_df.copy()
        Unigram_train_df = self.train_df.copy()

        # For each user, build a unigram distribution of (x, y) from training data
        user_unigrams = {}
        for uid, group in Unigram_train_df.groupby('uid'):
            # Count frequency of each (x, y) pair
            staypoints = group.groupby(['x', 'y']).size().reset_index(name='count')
            staypoints = staypoints.sort_values('count', ascending=False)
            # Save the most frequent (x, y) as the unigram prediction
            user_unigrams[uid] = (staypoints.iloc[0]['x'], staypoints.iloc[0]['y'])

        # Predict for each user in test set
        for uid, (x, y) in user_unigrams.items():
            Unigram_test_df.loc[Unigram_test_df['uid'] == uid, 'x'] = x
            Unigram_test_df.loc[Unigram_test_df['uid'] == uid, 'y'] = y

        # Save prediction
        output_df = Unigram_test_df[['uid', 'd', 't', 'x', 'y']].astype(int)
        output_df.to_csv(f'./Predictions/{self.output_name}_unigram.csv', index=False)
        print(f"Unigram Model預測完成，結果已儲存至./Predictions/{self.output_name}_unigram.csv\n")

    def Bigram_model(self):
        """
        Bigram Model: For a given user, create a bigram model of staypoints from days 1 to 60 of their trajectory,
        and use it to predict the trajectory in days 61 to 75.
        """
        Bigram_test_df = self.test_df.copy()
        Bigram_train_df = self.train_df.copy()

        # For each user, build a bigram model: {(prev_x, prev_y): (next_x, next_y)}
        user_bigrams = {}
        for uid, group in Bigram_train_df.groupby('uid'):
            group = group.sort_values(['d', 't'])
            coords = list(zip(group['x'], group['y']))
            bigram_counts = {}
            for i in range(len(coords) - 1):
                prev = coords[i]
                nxt = coords[i + 1]
                if prev not in bigram_counts:
                    bigram_counts[prev] = {}
                bigram_counts[prev][nxt] = bigram_counts[prev].get(nxt, 0) + 1
            user_bigrams[uid] = bigram_counts

        # For each user, get their first test point as the starting point
        for uid in Bigram_test_df['uid'].unique():
            user_test = Bigram_test_df[Bigram_test_df['uid'] == uid].sort_values(['d', 't'])
            user_train = Bigram_train_df[Bigram_train_df['uid'] == uid].sort_values(['d', 't'])
            # Start from the last train point if available, else use the first test point
            if not user_train.empty:
                last_train = (user_train.iloc[-1]['x'], user_train.iloc[-1]['y'])
            else:
                last_train = (user_test.iloc[0]['x'], user_test.iloc[0]['y'])

            preds = []
            prev = last_train
            bigram = user_bigrams.get(uid, {})
            # For each row in test, predict next using bigram, else fallback to unigram (most frequent)
            for idx in user_test.index:
                if prev in bigram and bigram[prev]:
                    # Pick the most frequent next point
                    next_xy = max(bigram[prev].items(), key=lambda x: x[1])[0]
                else:
                    # Fallback: use most frequent (x, y) in train (unigram)
                    train_counts = user_train.groupby(['x', 'y']).size().reset_index(name='count')
                    if not train_counts.empty:
                        next_xy = tuple(train_counts.sort_values('count', ascending=False).iloc[0][['x', 'y']])
                    else:
                        # If no train data, keep as previous
                        next_xy = prev
                Bigram_test_df.at[idx, 'x'] = int(next_xy[0])
                Bigram_test_df.at[idx, 'y'] = int(next_xy[1])
                prev = next_xy

        # Save prediction
        output_df = Bigram_test_df[['uid', 'd', 't', 'x', 'y']].astype(int)
        output_df.to_csv(f'./Predictions/{self.output_name}_bigram.csv', index=False)
        print(f"Bigram Model預測完成，結果已儲存至./Predictions/{self.output_name}_bigram.csv\n")

    def Bigram_model_top_p07(self):
        """
        For a given user, create a bigram model of staypoints from days 1 to 60 of their trajectory, 
        and use it to predict the trajectory in days 61 to 75, applying a sampling parameter of top_p=0.7.
        """

        Bigram_test_df = self.test_df.copy()
        Bigram_train_df = self.train_df.copy()
        top_p = 0.7

        # Build bigram model for each user: {(prev_x, prev_y): {next_xy: count}}
        user_bigrams = {}
        user_unigrams = {}
        for uid, group in Bigram_train_df.groupby('uid'):
            group = group.sort_values(['d', 't'])
            coords = list(zip(group['x'], group['y']))
            bigram_counts = {}
            unigram_counts = {}
            for i in range(len(coords) - 1):
                prev = coords[i]
                nxt = coords[i + 1]
                if prev not in bigram_counts:
                    bigram_counts[prev] = {}
                bigram_counts[prev][nxt] = bigram_counts[prev].get(nxt, 0) + 1
                unigram_counts[nxt] = unigram_counts.get(nxt, 0) + 1
            user_bigrams[uid] = bigram_counts
            # For unigram fallback
            if unigram_counts:
                user_unigrams[uid] = max(unigram_counts.items(), key=lambda x: x[1])[0]
            else:
                user_unigrams[uid] = None

        rng = np.random.default_rng()

        for uid in Bigram_test_df['uid'].unique():
            user_test = Bigram_test_df[Bigram_test_df['uid'] == uid].sort_values(['d', 't'])
            user_train = Bigram_train_df[Bigram_train_df['uid'] == uid].sort_values(['d', 't'])
            # Start from last train point if available, else use first test point
            if not user_train.empty:
                prev = (int(user_train.iloc[-1]['x']), int(user_train.iloc[-1]['y']))
            else:
                prev = (int(user_test.iloc[0]['x']), int(user_test.iloc[0]['y']))

            bigram = user_bigrams.get(uid, {})
            unigram_fallback = user_unigrams.get(uid, prev)

            for idx in user_test.index:
                # Ensure prev is a tuple of ints (not numpy array)
                if isinstance(prev, np.ndarray):
                    prev = tuple(int(x) for x in prev)
                candidates = bigram.get(prev, {})
                if candidates:
                    # Sort by count descending
                    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
                    if len(sorted_candidates) == 0:
                        next_xy = unigram_fallback if unigram_fallback is not None else prev
                    else:
                        counts = np.array([c[1] for c in sorted_candidates])
                        total = counts.sum()
                        probs = counts / total
                        # Cumulative probability for top_p sampling
                        cum_probs = np.cumsum(probs)
                        cutoff = np.searchsorted(cum_probs, top_p, side='right') + 1
                        top_candidates = sorted_candidates[:cutoff]
                        if len(top_candidates) == 0:
                            next_xy = unigram_fallback if unigram_fallback is not None else prev
                        else:
                            top_probs = np.array([c[1] for c in top_candidates], dtype=float)
                            top_probs /= top_probs.sum()
                            # Sample from top_p candidates
                            next_xy = rng.choice([c[0] for c in top_candidates], p=top_probs)
                else:
                    # Fallback: use unigram
                    next_xy = unigram_fallback if unigram_fallback is not None else prev
                Bigram_test_df.at[idx, 'x'] = int(next_xy[0])
                Bigram_test_df.at[idx, 'y'] = int(next_xy[1])
                prev = next_xy

        output_df = Bigram_test_df[['uid', 'd', 't', 'x', 'y']].astype(int)
        output_df.to_csv(f'./Predictions/{self.output_name}_bigram_top_p07.csv', index=False)
        print(f"Bigram Model (top_p=0.7) 預測完成，結果已儲存至./Predictions/{self.output_name}_bigram_top_p07.csv\n")

    def Evaluation(self, city_name, raw_data_path, generated_data_input, reference_data_input):
        """
        先valid繳交資料是否合法再Evaluate
        """
        try:
            # 檢查生成的資料是否符合規範
            # validator.main(city_name, raw_data_path, generated_data_input)

            # 讀取生成與參考資料
            generated_df = pd.read_csv(generated_data_input, header=0, dtype=int)
            reference_df = pd.read_csv(reference_data_input, header=0, dtype=int)

            # 取得所有 uid
            uids = sorted(set(generated_df['uid']) & set(reference_df['uid']))
            scores = []

            # 計算每個 uid 每天的 GEO-BLEU 分數
            for idx, uid in enumerate(uids):
                gen_user = generated_df[generated_df['uid'] == uid]
                ref_user = reference_df[reference_df['uid'] == uid]
                days = sorted(set(gen_user['d']) & set(ref_user['d']))
                for d in days:
                    gen_traj = gen_user[gen_user['d'] == d][['d', 't', 'x', 'y']].to_records(index=False)
                    ref_traj = ref_user[ref_user['d'] == d][['d', 't', 'x', 'y']].to_records(index=False)
                    gen_traj = [tuple(row) for row in gen_traj]
                    ref_traj = [tuple(row) for row in ref_traj]
                    score = geobleu.calc_geobleu_single(gen_traj, ref_traj)
                    scores.append(score)
                    print(f"{idx}/{len(uids)}人--uid={uid}--第{d}天", end='\r')

            final_score = sum(scores) / len(scores) if scores else 0.0
            return final_score
        except:
            print(f"{generated_data_input}資料驗證失敗，請檢查生成的資料是否符合規範。\n")
            pass

"""
測試程式碼
"""
if __name__ == "__main__":
    city_name = 'D'  
    prefix = 'x'

    
    test_data_input = f'./Training_Testing_Data/{city_name}_{prefix}_test.csv'
    train_data_input = f'./Training_Testing_Data/{city_name}_{prefix}_train.csv'
    output_name = f'{city_name}_{prefix}_baseline'
    BL = BaselineTest(test_data_input, train_data_input, output_name)
    
    #  計算各種基線預測
    BL.Global_Mean()
    BL.Global_Mode()
    BL.Per_User_Mean()
    BL.Per_User_Mode()
    BL.Unigram_model()
    BL.Bigram_model()   
    BL.Bigram_model_top_p07()

    # 對所有7種預測結果進行評估
    prediction_methods = [
        'global_mean',
        'global_mode', 
        'per_user_mean',
        'per_user_mode',
        'unigram',
        'bigram',
        'bigram_top_p07'
    ]
    
    print("開始評估所有預測方法...")
    
    for method in prediction_methods:
        print(f"\n評估 {method} 方法:")
        score = BL.Evaluation(city_name = f'{city_name.lower()}', 
                      raw_data_path = f'./Data/city_{city_name}_challengedata.csv', 
                      generated_data_input = f'./Predictions/{city_name}_{prefix}_baseline_{method}.csv',
                      reference_data_input = f'./Training_Testing_Data/{city_name}_{prefix}_test.csv'
                      )
        print(f"{method} 方法的 GEO-BLEU 分數: {score}")
