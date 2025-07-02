import os
import numpy as np
import pandas as pd

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data_df = pd.read_csv(self.data_path)

    def get_training_testing_data(self):
        """
        依據訓練及測試分類，並儲存成新的csv檔案。
        """
        os.makedirs('./Training_Testing_Data', exist_ok=True)

        # 找到 x=999 的所有 uid
        split_point = self.raw_data_df[self.raw_data_df['x'] == 999]['uid'].unique().min()
        print(f"資料分割點: uid={split_point}\n")

        # x_train;x_test (x,y中無999的uid，完整1-75天資料)
        x_train_df = self.raw_data_df[(self.raw_data_df['d'] <= 60) & (self.raw_data_df['uid'] < split_point)]
        x_test_df = self.raw_data_df[(self.raw_data_df['d'] > 60) & (self.raw_data_df['uid'] < split_point)]
        x_train_df.to_csv('./Training_Testing_Data/x_train.csv', index=False)
        x_test_df.to_csv('./Training_Testing_Data/x_test.csv', index=False)
        print(f"x_train有{x_train_df.shape[0]}筆資料, 有{x_train_df['uid'].nunique()}個uid, 從{x_train_df['uid'].min()}到{x_train_df['uid'].max()}")
        print(f"x_test有{x_test_df.shape[0]}筆資料, 有{x_test_df['uid'].nunique()}個uid, 從{x_test_df['uid'].min()}到{x_test_df['uid'].max()}\n")

        # y_train;y_test (y中有999的uid)
        y_train_df = self.raw_data_df[(self.raw_data_df['d'] <= 60) & (self.raw_data_df['uid'] >= split_point)]
        y_test_df = self.raw_data_df[(self.raw_data_df['d'] > 60) & (self.raw_data_df['uid'] >= split_point)]
        y_train_df.to_csv('./Training_Testing_Data/y_train.csv', index=False)
        y_test_df.to_csv('./Training_Testing_Data/y_test.csv', index=False)
        print(f"y_train有{y_train_df.shape[0]}筆資料, 有{y_train_df['uid'].nunique()}個uid, 從{y_train_df['uid'].min()}到{y_train_df['uid'].max()}")
        print(f"y_test有{y_test_df.shape[0]}筆資料, 有{y_test_df['uid'].nunique()}個uid, 從{y_test_df['uid'].min()}到{y_test_df['uid'].max()}\n")


        print("資料處理完成，已儲存在Training_Testing_Data資料夾中\n")
        return x_train_df, x_test_df, y_train_df, y_test_df


"""
測試程式碼
"""
if __name__ == "__main__":
    DataLoader = DataPreprocessor(data_path='./Data./city_D_challengedata.csv')

    x_train_df,_,_,_ = DataLoader.get_training_testing_data()