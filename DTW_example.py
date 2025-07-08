import numpy as np
import matplotlib.pyplot as plt

x = np.round(np.random.normal(5, 1, 11),2)
y = np.round(x[1:] * np.random.uniform(0.9, 1.1, size=len(x[1:])))
x = x[0:10]

dtw_matrix = np.zeros(shape=(len(x[1:])+2,len(x[1:])+2))
for i in range(len(x[1:])+2):
  for j in range(len(x[1:])+2):
    dtw_matrix[i][j] = np.inf

dtw_matrix[0][0] = 0

for i in range(1, len(x[1:])+2):
  for j in range(1, len(x[1:])+2):
    cost = np.absolute(x[i-1]-y[j-1])
    dtw_matrix[i][j] = cost + min(dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1])

# 用dtw_matrix計算dtw距離，從右下角開始
i = len(x[1:])+1; j = len(x[1:])+1 # 10,10
while i > 1:
  minmums = [dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1]] # [上,左,左上]
  minmum = min(minmums)

  if minmum == dtw_matrix[i-1][j-1]:
    i -= 1
    j -= 1
    print('左上',i,j,minmum)
  elif minmum == dtw_matrix[i-1][j]:
    i -= 1
    print('上',i,j,minmum)
  elif minmum == dtw_matrix[i][j-1]: 
    j -= 1
    print('左',i,j,minmum) 

print(f'Distance: {dtw_matrix[-1][-1]}')

# 這邊是方便輸出x和y的原始數值到dtw_matrix上
for i in range(1,len(x[1:])+2): dtw_matrix[i][0] = np.round(x[i-1],2)
for j in range(1,len(x[1:])+2): dtw_matrix[0][j] = np.round(y[j-1],2)

print(dtw_matrix)
plt.plot(x, label='x')
plt.plot(y, label='y')
plt.legend()

plt.show()
np.savetxt("dtw_matrix.csv", dtw_matrix, delimiter=",")