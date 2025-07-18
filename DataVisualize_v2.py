import numpy as np
import seaborn as sns
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anime
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoLocator
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 或 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

class DataVisualizer:
    def __init__(self, data_input):
        if isinstance(data_input, pd.DataFrame):
            self.raw_csv_df = data_input
            print(f"直接使用DataFrame資料，共有{self.raw_csv_df.shape[0]}筆資料\n",
                  f"資料範圍: uid={self.raw_csv_df['uid'].min()}~{self.raw_csv_df['uid'].max()}\n",
                  f"時間範圍: days={self.raw_csv_df['d'].min()}~{self.raw_csv_df['d'].max()}\n")
        elif isinstance(data_input, str):
            self.data_path = data_input
            self.raw_csv_df = pd.read_csv(self.data_path, header=0, dtype=int)
            print(f"讀取資料成功，共有{self.raw_csv_df.shape[0]}筆資料\n",
                f"資料範圍: uid={self.raw_csv_df['uid'].min()}~{self.raw_csv_df['uid'].max()}\n",
                f"時間範圍: days={self.raw_csv_df['d'].min()}~{self.raw_csv_df['d'].max()}\n")  
        else:
            raise ValueError("只能接受DataFrame或資料路徑字串（csv檔）。")
        
        if self.raw_csv_df[(self.raw_csv_df['x'] == 999) & (self.raw_csv_df['y'] == 999)].shape[0] > 0:
            print("警告: 資料中包含無效的座標(x=999或y=999)，將被移除。")
            self.raw_csv_df = self.raw_csv_df[(self.raw_csv_df['x'] != 999) & (self.raw_csv_df['y'] != 999)]
            print(f"已移除無效資料，共有{self.raw_csv_df.shape[0]}筆有效資料\n")

    def histogram2d(self):
        xy_uid = self.raw_csv_df.groupby(['x', 'y'])['uid'].nunique().reset_index()
        heatmap_data = np.zeros((200, 200), dtype=float)
        for _, row in xy_uid.iterrows():
            # x, y座標從1開始，需減1對應到陣列索引
            count = row['uid']
            # 只對大於0的數值取log10，否則設為0
            heatmap_data[int(row['y'])-1, int(row['x'])-1] = np.log10(count) if count > 0 else 0

        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        im = ax.imshow(
            heatmap_data, 
            cmap='Reds', 
            origin='lower', 
            extent=[1, 200, 1, 200], 
            aspect='equal'
        )
        ax.set_title(f"每個(x, y)座標的唯一使用者數 Day:{self.raw_csv_df['d'].min()}~{self.raw_csv_df['d'].max()}的軌跡 {xy_uid.shape[0]}點有資料", fontsize=18)
        ax.set_xlim(1, 200)
        ax.set_ylim(1, 200)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('唯一使用者數 (log10)',fontsize=14)

    def histogram2d_animation(self, fps=2, output_each_frame=False, max_days=75):
        """
        依據每一天與每個小時繪製2D histogram動畫，顯示每天每個小時每個(x, y)座標的使用者數
        """
        os.makedirs('./Animations', exist_ok=True)
        if output_each_frame:
            os.makedirs('./Animations/histogram2d_each_frame', exist_ok=True)

        ani_fig, ani_ax = plt.subplots(figsize=(12, 12))
        ani_ax.set_xlim(1, 200)
        ani_ax.set_ylim(1, 200)
        ani_ax.invert_yaxis()
        ani_ax.grid(True, alpha=0.3)
        ani_ax.xaxis.set_major_locator(MultipleLocator(10))
        ani_ax.yaxis.set_major_locator(MultipleLocator(10))

        days = sorted(self.raw_csv_df[self.raw_csv_df['d'] <= max_days]['d'].unique())
        # hours = sorted(self.raw_csv_df['t'].unique())
        hours = np.arange(0, 24)  
        frames = [(d, t) for d in days for t in hours]

        def init():
            ani_ax.clear()
            ani_ax.set_xlim(1, 200)
            ani_ax.set_ylim(1, 200)
            ani_ax.invert_yaxis()
            ani_ax.grid(True, alpha=0.3)
            ani_ax.xaxis.set_major_locator(MultipleLocator(10))
            ani_ax.yaxis.set_major_locator(MultipleLocator(10))

        def update(i):
            ani_ax.clear()
            ani_ax.set_xlim(1, 200)
            ani_ax.set_ylim(1, 200)
            ani_ax.invert_yaxis()
            ani_ax.grid(True, alpha=0.3)
            ani_ax.xaxis.set_major_locator(MultipleLocator(10))
            ani_ax.yaxis.set_major_locator(MultipleLocator(10))

            current_day, current_hour = frames[i]
            # frame_data = self.raw_csv_df[(self.raw_csv_df['d'] == current_day) & (self.raw_csv_df['t'] == current_hour)]
            frame_data = self.raw_csv_df[(self.raw_csv_df['d'] == current_day) & (self.raw_csv_df['t'] >= current_hour*2) & (self.raw_csv_df['t'] < (current_hour + 1) * 2)]  # 每小時兩個時間點
            xy_uid = frame_data.groupby(['x', 'y'])['uid'].count().reset_index()  # 不用nunique要算總數

            heatmap_data = np.zeros((200, 200), dtype=float)
            for _, row in xy_uid.iterrows():
                count = row['uid']
                heatmap_data[int(row['y'])-1, int(row['x'])-1] = np.log10(count) if count > 0 else 0

            im = ani_ax.imshow(
                heatmap_data,
                cmap='Reds',
                origin='lower',
                extent=[1, 200, 1, 200],
            )

            ani_ax.set_title(f"Day:{current_day} Hour:{current_hour}~{current_hour+1} 每個(x, y)座標的使用者數", fontsize=16)
            if output_each_frame:
                ani_fig.savefig(f'./Animations/histogram2d_each_frame/histogram2d_day{current_day}_hour{current_hour}~{current_hour+1}.png')
            print(f"2D histogram動畫進度: {i+1}/{len(frames)}", end='\r')

        ani = anime.FuncAnimation(ani_fig, update, frames=len(frames), init_func=init, repeat=True)
        ani.save('./Animations/histogram2d_by_day_hour_animation.gif', fps=fps, writer='pillow')
        plt.title("2D histogram 每天每小時動畫")

    def single_user_trajectory(self, uid,   output=False):
        user_df = self.raw_csv_df[self.raw_csv_df['uid'] == uid]
        if user_df.empty:
            print(f"UID {uid} 的資料不存在")
            return
        
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.plot(user_df['x'], user_df['y'], marker='o', markersize=2, linestyle='-', alpha=0.2, c="r")
        ax.set_title(f"UID:{uid} Day:{user_df['d'].min()}~{user_df['d'].max()}的軌跡，共有{user_df.shape[0]}筆資料", fontsize=18)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_xlim(1, 200)
        ax.set_ylim(1, 200)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        print(f"UID:{uid} 的資料筆數: {user_df.shape[0]}")
        if output:
            os.makedirs('./Animations', exist_ok=True)
            fig.savefig(f'./Animations/uid_{uid}_trajectory.png')
            print(f"UID:{uid} 的軌跡圖已儲存到 ./Animations/uid_{uid}_trajectory.png")
    
    def single_user_trajectory_animation(self, uid, fps=2, output_each_frame=False):
        user_df = self.raw_csv_df[self.raw_csv_df['uid'] == uid]
        os.makedirs('./Animations', exist_ok=True)
        if output_each_frame: os.makedirs(f'./Animations/uid_{uid}_each_frame', exist_ok=True)
        if user_df.empty:
            print(f"UID {uid} 的資料不存在")
            return
        
        ani_fig, ani_ax = plt.subplots()
        ani_ax.set_xlim(1, 200)
        ani_ax.set_ylim(1, 200)
        ani_ax.invert_yaxis()
        ani_ax.grid(True, alpha=0.3)
        ani_ax.xaxis.set_major_locator(MultipleLocator(10))
        ani_ax.yaxis.set_major_locator(MultipleLocator(10))

        def init():
            ani_ax.clear()
            ani_ax.set_xlim(1, 200)
            ani_ax.set_ylim(1, 200)
            ani_ax.invert_yaxis()
        
        def update(i):
            ani_ax.clear()
            ani_ax.set_xlim(1, 200)
            ani_ax.set_ylim(1, 200)
            ani_ax.invert_yaxis()

            window = 5  # 記憶幾個點
            for j in range(max(0, i - window + 1), i + 1):
                alpha = 0.1 + 0.9 * (j - max(0, i - window + 1)) / window  # 越舊越淡
                ani_ax.scatter(user_df.iloc[j]['x'], user_df.iloc[j]['y'], alpha=alpha, s=30, c='red')

            ani_ax.grid(True, alpha=0.3)
            ani_ax.xaxis.set_major_locator(MultipleLocator(10))
            ani_ax.yaxis.set_major_locator(MultipleLocator(10))
            ani_ax.tick_params(axis='x', labelsize=10)
            ani_ax.tick_params(axis='y', labelsize=10)
            ani_ax.set_title(f"uid={uid} 第{user_df.iloc[i]['d']}天 {user_df.iloc[i]['t']*0.5}點鐘 ", fontsize=18)
            if output_each_frame:  
                ani_fig.savefig(f'./Animations/uid_{uid}_each_frame/uid_{uid}_day{user_df.iloc[i]["d"]}_time{user_df.iloc[i]["t"]*0.5}.png')
            print(f"單人分時軌跡動畫進度: {i+1}/{user_df.shape[0]}", end='\r')

        ani = anime.FuncAnimation(ani_fig, update, frames=user_df.shape[0], init_func=init, repeat=True)
        start_day = user_df['d'].min()
        end_day = user_df['d'].max()
        ani.save(f'./Animations/uid_{uid}_days{start_day}~{end_day}_trajectory_animation.gif', fps=fps, writer='pillow')
        plt.title(f"UID:{uid} 的軌跡動畫")

    def Everytimestamp_User_Count(self):
        """
        繪製每個時間戳記總人數的折線圖
        """
        fig, axs = plt.subplots(1, 2, figsize=(9, 4)) 

        hourly_user_count = self.raw_csv_df.groupby('t')['uid'].count()
        axs[0].plot(hourly_user_count.index, hourly_user_count.values, marker='o')
        axs[0].set_title(f"Days: {self.raw_csv_df['d'].min()}~{self.raw_csv_df['d'].max()} UID: {self.raw_csv_df['uid'].min()}~{self.raw_csv_df['uid'].max()}",fontsize=18)
        axs[0].set_xlabel("小時*0.5", fontsize=14)
        axs[0].set_ylabel("No. Users", fontsize=14)
        axs[0].set_xlim(0, self.raw_csv_df['t'].max())
        axs[0].xaxis.set_major_locator(MultipleLocator(1))
        axs[0].yaxis.set_major_locator(AutoLocator())
        axs[0].grid(True, alpha=0.3)

        hourly_unique_user_count = self.raw_csv_df.groupby('t')['uid'].nunique()
        axs[1].plot(hourly_unique_user_count.index, hourly_unique_user_count.values, marker='o', color='orange')
        axs[1].set_title(f"Days: {self.raw_csv_df['d'].min()}~{self.raw_csv_df['d'].max()} UID: {self.raw_csv_df['uid'].min()}~{self.raw_csv_df['uid'].max()}", fontsize=18)
        axs[1].set_xlabel("小時*0.5", fontsize=14)
        axs[1].set_ylabel("No. Unique Users", fontsize=14)
        axs[1].set_xlim(0, self.raw_csv_df['t'].max())
        axs[1].xaxis.set_major_locator(MultipleLocator(1))
        axs[1].yaxis.set_major_locator(AutoLocator())
        axs[1].grid(True, alpha=0.3)

    def Everyday_User_Count(self, Zoom_mode = False, start_day=1, end_day=75):
        """
        繪製每天活躍使用者數量的折線圖
        """
        fig, axs = plt.subplots(1, 2, figsize=(9, 4)) 

        if Zoom_mode:
            daily_users = self.raw_csv_df.groupby('d')['uid'].count()
            daily_users = daily_users[(daily_users.index >= start_day) & (daily_users.index <= end_day)]
            print(f"daily_users已縮小範圍到第{start_day}天到第{end_day}天，共有{daily_users.shape[0]}筆資料")
            axs[0].xaxis.set_major_locator(MultipleLocator(1))
        else:
            daily_users = self.raw_csv_df.groupby('d')['uid'].count()
            axs[0].xaxis.set_major_locator(MultipleLocator(7))
        
        axs[0].plot(daily_users.index, daily_users.values, color='blue')
        axs[0].set_xlabel("Day", fontsize=14)
        axs[0].set_ylabel("No. Users", fontsize=14)
        axs[0].set_title(f"Days: {daily_users.index.min()}~{daily_users.index.max()} UID: {self.raw_csv_df['uid'].min()}~{self.raw_csv_df['uid'].max()}", fontsize=18)
        axs[0].set_xlim(daily_users.index.min(), daily_users.index.max())
        axs[0].yaxis.set_major_locator(AutoLocator())
        axs[0].grid(True, alpha=0.3)
        
        if Zoom_mode:
            daily_unique_users = self.raw_csv_df.groupby('d')['uid'].nunique()
            daily_unique_users = daily_unique_users[(daily_unique_users.index >= start_day) & (daily_unique_users.index <= end_day)]
            print(f"daily_unique_users已縮小範圍到第{start_day}天到第{end_day}天，共有{daily_unique_users.shape[0]}筆資料")
            axs[1].xaxis.set_major_locator(MultipleLocator(1))
        else:
            daily_unique_users = self.raw_csv_df.groupby('d')['uid'].nunique()
            axs[1].xaxis.set_major_locator(MultipleLocator(7))

        axs[1].plot(daily_unique_users.index, daily_unique_users.values, color='orange')
        axs[1].set_xlabel("Day", fontsize=14)
        axs[1].set_ylabel("No. Unique Users", fontsize=14)
        axs[1].set_title(f"Days: {daily_users.index.min()}~{daily_users.index.max()} UID: {self.raw_csv_df['uid'].min()}~{self.raw_csv_df['uid'].max()}", fontsize=18)
        axs[1].set_xlim(daily_unique_users.index.min(), daily_unique_users.index.max())
        axs[1].yaxis.set_major_locator(AutoLocator())
        axs[1].grid(True, alpha=0.3)

    def User_count_distribution(self):
        """
        繪製每個使用者資料筆數的分布（boxplot），並顯示q1, mean, q2
        """
        user_counts = self.raw_csv_df.groupby('uid').size()
        q1 = int(user_counts.quantile(0.25))
        mean = int(user_counts.mean())
        q3 = int(user_counts.quantile(0.75))
        print(f"最大資料筆數的uid: {user_counts.idxmax()}, 最小資料筆數的uid: {user_counts.idxmin()}")
        print(f"最大值: {user_counts.max()}, 最小值: {user_counts.min()}")
        print(f"Q1: {q1}, Mean: {mean}, Q3: {q3}")

        plt.figure(figsize=(4, 4))
        sns.boxplot(x=user_counts)
        sns.stripplot(x=user_counts, color="orange", jitter=0.2, size=1.5, alpha=0.5)
        plt.xlabel("每個使用者的資料筆數", fontsize=18)
        plt.title(f"每個使用者資料筆數分布: Q1: {q1}, Mean: {mean}, Q3: {q3}", fontsize=18)
        plt.grid(True, axis='x', alpha=0.3)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

"""
測試程式碼
"""
if __name__ == "__main__":
    # 直接使用DataFrame資料
    # x_train_df = pd.read_csv('./Training_Testing_Data/x_train.csv', header=0, dtype=int)
    # x_test_df = pd.read_csv('./Training_Testing_Data/x_test.csv', header=0, dtype=int)
    # y_train_df = pd.read_csv('./Training_Testing_Data/y_train.csv', header=0, dtype=int)
    # y_test_df = pd.read_csv('./Training_Testing_Data/y_test.csv', header=0, dtype=int)
    # test_df = pd.concat([x_train_df, x_test_df, y_train_df, y_test_df], ignore_index=True)
    # DataLoader = DataVisualizer(data_input=test_df)

    # 或者從CSV檔案讀取資料
    DataLoader = DataVisualizer(data_input='./Data/city_D_challengedata.csv')

    # DataLoader.histogram2d()
    # DataLoader.histogram2d_animation(fps=2, output_each_frame=True, max_days=3) # 替換成你想要輸出的天數[1:max_days]
    # DataLoader.single_user_trajectory(uid=3)  # 替換成你想要的uid
    # DataLoader.single_user_trajectory_animation(uid=3, fps=2, output_each_frame=True)  # 替換成你想要的uid
    # DataLoader.Everytimestamp_User_Count()
    DataLoader.Everyday_User_Count(Zoom_mode=True, start_day=28, end_day=40)  # 替換成你想要的天數範圍
    # DataLoader.User_count_distribution()
    plt.show()
