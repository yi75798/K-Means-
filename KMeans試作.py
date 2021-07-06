#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Means法貼文分類試做
Created on Mon Jul  5 16:42:15 2021

@author: liang-yi
"""
# 0.載入套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn import cluster, metrics

### 1.讀取原始檔
data = pd.read_csv("/Users/liang-yi/Desktop/Covid19與團結效應/tsai.csv")

## 僅留下日期、貼文內容、總互動、讚、留言數、分享數、愛心、Wow、哈哈、傷心、怒、Care、貼文連結
df = data[["Post Created Date", "Message", "Total Interactions", "Likes", 
           "Comments","Shares", "Love", "Wow", "Haha", "Sad", "Angry", "Care",
           "URL"]]

## 轉換總互動數type為int
for i in df["Total Interactions"].index:
    df["Total Interactions"].iloc[i] = df["Total Interactions"].iloc[i].replace(",", "")
    df["Total Interactions"].iloc[i] = int(df["Total Interactions"].iloc[i])


### 2.變項操作化
## Reaction =  總互動－留言數－分享數
df["Reaction"] = df["Total Interactions"] - df["Comments"] - df["Shares"]
## 各項互動比例 = 互動／總互動數
df["p_Likes"] = df["Likes"] / df["Reaction"]
df["p_Love"] = df["Love"] / df["Reaction"]
df["p_Wow"] = df["Wow"] / df["Reaction"]
df["p_Haha"] = df["Haha"] / df["Reaction"]
df["p_Sad"] = df["Sad"] / df["Reaction"]
df["p_Angry"] = df["Angry"] / df["Reaction"]
df["p_Care"] = df["Care"] / df["Reaction"]

### 3.K-Means分群
## 挑出各項互動比例轉換成array
dx = df[["p_Likes", "p_Love", "p_Wow", "p_Haha", "p_Sad", "p_Angry", "p_Care"]]
dx = dx.values

# KMeans 演算法
kmeans_fit = cluster.KMeans(n_clusters = 2).fit(dx)
cluster_labels = kmeans_fit.labels_

# 印出分群結果
print("分群結果：")
print(cluster_labels)
print("---")

# 印出績效
silhouette_avg = metrics.silhouette_score(dx, cluster_labels)
print(silhouette_avg)

## 製作績效圖
# 迴圈
silhouette_avgs = []
ks = range(2, 11)
for k in ks:
    kmeans_fit = cluster.KMeans(n_clusters = k).fit(dx)
    cluster_label = kmeans_fit.labels_
    silhouette_avg = metrics.silhouette_score(dx, cluster_label)
    silhouette_avgs.append(silhouette_avg)

# 作圖並印出 k = 2 到 10 的績效
plt.plot(ks, silhouette_avgs)
plt.title("Silhouette score")
plt.xlabel("nums_K")
plt.ylabel("score")
plt.show()
print(silhouette_avgs)

### 4.合併、輸出資料
df_label = pd.DataFrame(cluster_labels, columns=["label"])
output = df.merge(df_label, how="inner", left_index=True, right_index=True)
output.to_csv("/Users/liang-yi/Desktop/Covid19與團結效應/tsai_dealed.csv", index=False, encoding="utf_8_sig")
