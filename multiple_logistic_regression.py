# 【目標】iris データからロジスティック回帰を行う
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# 目的変数（target）と説明変数（data）の用意
iris = load_iris()
# print(iris.DESCR)
tmp_data = pd.DataFrame(iris.data, columns=iris.feature_names)
tmp_data['target'] = iris.target

# 今回はtarget 0, 1, 2 のとりうる値のうち、0 or 1のもののみを抽出
data_iris = tmp_data[tmp_data['target'] <= 1]
print(data_iris.head())
print(data_iris.shape)

# 散布図に可視化
plt.scatter(data_iris.iloc[:, 2],
            data_iris.iloc[:, 3],
            c=data_iris["target"])
# lt.show()


# ロジスティック回帰分析
logit = LogisticRegression()
x_column_list = ['sepal length (cm)']  # 列名を修正
y_column_list = ['target']

x = data_iris[x_column_list]
y = data_iris[y_column_list]

# 2D arrayにreshape
x = x.values.reshape(-1, 1)

# yを1次元配列に変換
y = y.values.ravel()

logit.fit(x, y)
print(logit.coef_)
print(logit.intercept_)

# 複数の説明変数でロジスティック回帰
logit_multi = LogisticRegression()
x_column_list_multi = [data_iris.columns[0],
                       data_iris.columns[1],
                       data_iris.columns[2],
                       data_iris.columns[3]]

x_multi = data_iris[x_column_list_multi]

logit_multi.fit(x_multi, y)
print(logit.coef_)
print(logit.intercept_)

