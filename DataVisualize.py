import numpy as np
import seaborn as sns
import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anime
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoLocator
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 或 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

# Global parameters
city = "D"
csv_file = f"./Data/city_{city}_challengedata.csv"  
specific_time = 7 # 設定要顯示時段的全域heatmap，間隔為specific_time~specific_time+specific_time_gap點
specific_time_gap = 1 # 時段間隔
uid = 3 # 單人全時軌跡的uid
plt_show = True  # 是否顯示圖形
Animation_show = False  # 是否顯示動畫
heatmap_gif_day = 14 # 分時heatmap動畫的天數

# Read the CSV file for City 
city_raw_df = pd.read_csv(csv_file, header=0, dtype=int)


# Clean the data by removing x or y = 999
city_clean_df = city_raw_df[(city_raw_df['x'] != 999) & (city_raw_df['y'] != 999)]
print(city_clean_df.tail())
print(f"共有可用{city_clean_df.shape[0]}筆資料\n已移除{city_raw_df.shape[0] - city_clean_df.shape[0]}筆無效資料")

# Create subplot
fig, axs = plt.subplots(2, 3, figsize=(30, 20))

# 200*200 全域全時heatmap
# print("正在繪製全域全時Heatmap...")
# axs[0, 0].scatter(city_clean_df['x'], city_clean_df['y'], alpha=0.05, s=1, c='r')
# axs[0, 0].set_title("全域全時Heatmap")
# axs[0, 0].set_xlim(1, 200)
# axs[0, 0].set_ylim(1, 200)
# axs[0, 0].set_aspect('equal')
# axs[0, 0].invert_yaxis()
# axs[0, 0].grid(True, alpha=0.3)
# axs[0, 0].xaxis.set_major_locator(MultipleLocator(10))
# axs[0, 0].yaxis.set_major_locator(MultipleLocator(10))
# print("全域全時Heatmap 完成")

# 使用2D histogram顯示每個(x, y)座標的唯一使用者數，數值以10為底對數化
xy_uid = city_clean_df.groupby(['x', 'y'])['uid'].nunique().reset_index()
heatmap_data = np.zeros((200, 200), dtype=float)
for _, row in xy_uid.iterrows():
    # x, y座標從1開始，需減1對應到陣列索引
    count = row['uid']
    # 只對大於0的數值取log10，否則設為0
    heatmap_data[int(row['y'])-1, int(row['x'])-1] = np.log10(count) if count > 0 else 0
    
im = axs[0, 0].imshow(
    heatmap_data, 
    cmap='Reds', 
    origin='lower', 
    extent=[1, 200, 1, 200],
    aspect='equal'
)
axs[0, 0].set_title("每個(x, y)座標的唯一使用者數")
axs[0, 0].set_xlim(1, 200)
axs[0, 0].set_ylim(1, 200)
axs[0, 0].invert_yaxis()
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].xaxis.set_major_locator(MultipleLocator(10))
axs[0, 0].yaxis.set_major_locator(MultipleLocator(10))
cbar = plt.colorbar(im, ax=axs[0, 0])
cbar.set_label('唯一使用者數 (log10)')

# 單人全時軌跡
print("正在繪製單人全時軌跡...")
person_df = city_clean_df[(city_clean_df['uid'] == uid)]
axs[0, 1].plot(person_df['x'], person_df['y'], marker='o', markersize=2, linewidth=1, alpha=0.2, c="m")
axs[0, 1].set_title(f"uid={uid} 的軌跡")
x_min = math.floor(person_df['x'].min())
x_max = math.ceil(person_df['x'].max())
x_gap = x_max - x_min
y_min = math.floor(person_df['y'].min())
y_max = math.ceil(person_df['y'].max())
y_gap = y_max - y_min
gap = x_gap if x_gap > y_gap else y_gap
axs[0, 1].set_xlim(1, 200)
axs[0, 1].set_ylim(1, 200)
# axs[0, 1].set_xlim(x_min, x_min + gap)
# axs[0, 1].set_ylim(y_min, y_min + gap)
axs[0, 1].set_aspect('equal')
axs[0, 1].invert_yaxis()
axs[0, 1].grid(True, alpha=0.3)
axs[0, 1].xaxis.set_major_locator(MultipleLocator(10))
axs[0, 1].yaxis.set_major_locator(MultipleLocator(10))
print(f"uid:{uid} 的資料筆數: {person_df.shape[0]}")
print("單人全時軌跡 完成")

# 特定時段的的全域heatmap
print("正在繪製特定時段全域Heatmap...")
specific_time_heatmap_df = city_clean_df[(city_clean_df['t']*0.5 >= specific_time) & (city_clean_df['t']*0.5 < (specific_time + specific_time_gap)%24)]

if specific_time_heatmap_df.shape[0] > 1000:
    print(f"共有{specific_time_heatmap_df.shape[0]}筆資料，超過1000，將隨機抽樣1000筆資料")
    specific_time_heatmap_df = specific_time_heatmap_df.sample(n=1000) 
else:
    specific_time_heatmap_df

print(f"全域Heatmap (time= {specific_time}:00~{specific_time + specific_time_gap}:00)的資料筆數: {specific_time_heatmap_df.shape[0]}")
sns.kdeplot(
    data=specific_time_heatmap_df, x="x", y="y",
    fill=True, cmap="Reds", bw_adjust=1, gridsize=80, levels=25, 
    ax=axs[1, 0], alpha=0.5
)
axs[1, 0].set_xlabel("")  # 不顯示 x 軸名稱
axs[1, 0].set_ylabel("")  # 不顯示 y 軸名稱
axs[1, 0].set_title(f"全域Heatmap (time= {specific_time}:00~{specific_time + specific_time_gap}:00)")
axs[1, 0].set_xlim(1, 200)
axs[1, 0].set_ylim(1, 200)
axs[1, 0].invert_yaxis()
axs[1, 0].set_aspect('equal')
axs[1, 0].grid(True, alpha=0.3)
axs[1, 0].xaxis.set_major_locator(MultipleLocator(10))
axs[1, 0].yaxis.set_major_locator(MultipleLocator(10))
print("特定時段全域Heatmap 完成")

# 每小時平均使用人數
print("正在繪製每小時平均使用人數...")
avg_hourly_users = city_clean_df.groupby('t')['uid'].count()
axs[1, 1].plot(avg_hourly_users.index, avg_hourly_users.values)
axs[1, 1].set_xlabel("Hour")
axs[1, 1].set_ylabel("No. Users")
axs[1, 1].set_aspect('auto')
axs[1, 1].xaxis.set_major_locator(MultipleLocator(2))
axs[1, 1].yaxis.set_major_locator(AutoLocator())
axs[1, 1].grid(True, alpha=0.3)
print("每小時平均使用人數 完成")

# 每日活耀人數變化
less_than_days = 75  # 設定要顯示的天數
print("正在繪製每日活耀人數變化(of unique users)...")
daily_users = city_clean_df[city_clean_df['d'] <= less_than_days].groupby('d')['uid'].nunique()  
axs[0, 2].plot(daily_users.index, daily_users.values)
axs[0, 2].set_xlabel("Day")
axs[0, 2].set_ylabel("No. Unique Users")
axs[0, 2].xaxis.set_major_locator(MultipleLocator(7))
axs[0, 2].grid(True, alpha=0.3)
print("正在繪製每日活耀人數變化(of unique users) 完成")
print("正在繪製每日活耀人數變化(No. of Users)...")
daily_users = city_clean_df[city_clean_df['d'] <= less_than_days].groupby('d')['uid'].count()
axs[1, 2].plot(daily_users.index, daily_users.values)
axs[1, 2].set_xlabel("Day")
axs[1, 2].set_ylabel("No. Users")
axs[1, 2].xaxis.set_major_locator(MultipleLocator(7))
axs[1, 2].grid(True, alpha=0.3)

if Animation_show:
    # 單人分時軌跡動畫
    print("正在繪製單人分時軌跡動畫...")
    ani_fig, ani_ax = plt.subplots()
    x = person_df['x']
    y = person_df['y']
    days = person_df['d']
    time = person_df['t']
    def init():
        ani_ax.clear()
        ani_ax.set_xlim(1, 200)
        ani_ax.set_ylim(1, 200)
        # ani_ax.set_xlim(x_min, x_min + gap)
        # ani_ax.set_ylim(y_min, y_min + gap)
        ani_ax.invert_yaxis()

    def run(i):
        ani_ax.clear()
        ani_ax.set_xlim(1, 200)
        ani_ax.set_ylim(1, 200)
        # ani_ax.set_xlim(x_min, x_min + gap)
        # ani_ax.set_ylim(y_min, y_min + gap)
        ani_ax.invert_yaxis()

        window = 5  # 記憶幾個點
        for j in range(max(0, i - window + 1), i + 1):
            alpha = 0.1 + 0.9 * (j - max(0, i - window + 1)) / window  # 越舊越淡
            ani_ax.scatter(x.iloc[j], y.iloc[j], alpha=alpha, s=30, c='m')

        ani_ax.grid(True, alpha=0.3)
        ani_ax.xaxis.set_major_locator(MultipleLocator(10))
        ani_ax.yaxis.set_major_locator(MultipleLocator(10))
        ani_ax.set_title(f"uid={uid} 第{days.iloc[i]}天 {time.iloc[i]*0.5}點鐘 ")
        print(f"單人分時軌跡動畫進度: {i+1}/{person_df.shape[0]}", end='\r')

    ani = anime.FuncAnimation(ani_fig, run, frames=person_df.shape[0], init_func=init, repeat=True)
    ani.save(f'./uid_{uid}_animation.gif', fps=3, writer='pillow')
    print("單人分時軌跡動畫 完成")

    # 全域分時heatmap動畫
    print("正在繪製全域分時Heatmap動畫...")
    heatmap_ani_fig, heatmap_ani_ax = plt.subplots(figsize=(8, 8))
    def init():
        heatmap_ani_ax.clear()
        heatmap_ani_ax.set_xlim(1, 200)
        heatmap_ani_ax.set_ylim(1, 200)
        heatmap_ani_ax.invert_yaxis()

    def update_heatmap(frame):
        heatmap_ani_ax.clear()
        heatmap_ani_ax.set_xlim(1, 200)
        heatmap_ani_ax.set_ylim(1, 200)
        heatmap_ani_ax.invert_yaxis()
        day = frame // 24 + 1  # 天數從1開始
        hour = frame % 24
        frame_df = city_clean_df[(city_clean_df['d'] == day) & (city_clean_df['t']*0.5 >= hour) & (city_clean_df['t']*0.5 < hour + 1)]
        if frame_df.shape[0] > 50000:
            frame_df = frame_df.sample(n=50000)
        if not frame_df.empty:
            sns.kdeplot(
                data=frame_df, x="x", y="y",
                fill=True, cmap="Reds", bw_adjust=1, gridsize=80, levels=25, 
                ax=heatmap_ani_ax, alpha=0.5, 
            )
        heatmap_ani_ax.set_xlabel("")
        heatmap_ani_ax.set_ylabel("")
        heatmap_ani_ax.set_title(f"全域Heatmap (day={day}, hour={hour}:00~{hour+1}:00)")
        heatmap_ani_ax.set_xlim(1, 200)
        heatmap_ani_ax.set_ylim(1, 200)
        heatmap_ani_ax.invert_yaxis()
        heatmap_ani_ax.set_aspect('equal')
        heatmap_ani_ax.grid(True, alpha=0.3)
        heatmap_ani_ax.xaxis.set_major_locator(MultipleLocator(10))
        heatmap_ani_ax.yaxis.set_major_locator(MultipleLocator(10))
        print(f"分時Heatmap動畫進度: {frame+1}/{heatmap_gif_day*24} (day={day}, hour={hour}:00~{hour+1}:00)", end='\r')

    heatmap_ani = anime.FuncAnimation(heatmap_ani_fig, update_heatmap, frames=heatmap_gif_day*24, init_func=init, repeat=False)
    heatmap_ani.save(f'./global_time_heatmap_day_hour.gif', fps=2, writer='pillow')
    print("全域分時Heatmap動畫 完成")

if plt_show:
    plt.tight_layout()
    plt.show()
