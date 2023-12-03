# plot the boxplot of the depths on spiral data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

result_depth = {}
result_acc = {}
for omega in range(0, 21, 2):
    
    depth = []
    accuracy = []
    for i in range(5):
        file = pd.read_csv("tmp_log-spiral/log-spiral/" + "tmp.UDN-inf.spiral.v2.O-%d.seed-%d.lr0.0100iter%d.csv" % (omega, omega * i ,i), header=0, index_col=0)
        depth.append(file['depth'].iloc[-1])
        accuracy.append(file['test_accuracy'].iloc[-1])
    result_depth[str(omega)] = depth
    result_acc[str(omega)] = accuracy
    
fig = plt.figure(figsize=(7, 5))
plt.boxplot(result_depth.values())
plt.xticks(range(1, len(result_depth.keys()) + 1), result_depth.keys())
plt.xlabel(r"Complexity $\omega$")
plt.ylabel("depth selected")
plt.title("UDN: selected depth")
plt.show()

fig = plt.figure(figsize=(7, 5))
plt.boxplot(result_acc.values())
plt.xticks(range(1, len(result_acc.keys()) + 1), result_acc.keys())
plt.xlabel(r"Complexity $\omega$")
plt.ylabel("test accuracy")
plt.title("UDN: test accuracy")
plt.show()