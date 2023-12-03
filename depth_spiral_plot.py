# plot the boxplot of the depths on spiral data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

spiral_our_depth = pd.read_csv("tmp/result/spiral_our_depth.csv", header=0, index_col=0)
spiral_our_acc = pd.read_csv("tmp/result/spiral_our_acc.csv", header=0, index_col=0)
n = spiral_our_depth.shape[-1]
res_depth = []
res_acc = []
fig, ax1 = plt.subplots()
# for i in range(n):
#     res_depth.append(spiral_our_depth.iloc[:,i].values)
#     res_acc.append(spiral_our_acc.iloc[:,i].values / 100)
ax1 = spiral_our_depth.boxplot(figsize=(7,5), grid=False, )

ax1.set_xlabel(r'Complexity $\omega$')
ax1.set_ylabel('depth selected')
ax1.set_title('VROOM: selected depth')
plt.show()

fig, ax2 = plt.subplots()
ax2 = spiral_our_acc.boxplot(figsize=(7, 5), grid=False)
ax2.set_title('VROOM: test accuracy')
ax2.set_xlabel(r'Complexity $\omega$')
ax2.set_ylabel('test acc')
plt.show()
