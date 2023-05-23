# 【目標】iris データからロジスティック回帰を行う
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

"""
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
plt.show()


# ロジスティック回帰分析
logit = LogisticRegression()
x_column_list = ['sepal length (cm)']
y_column_list = ['target']

x = data_iris[x_column_list]
y = data_iris[y_column_list]

print(x.shape)
print(y.shape)


# logit.fit(x, y)
# print(logit.coef_)
# print(logit.intercept_)
"""


iris_df = sns.load_dataset('iris') # データセットの読み込み
iris_df = iris_df[(iris_df['species']=='versicolor') | (iris_df['species']=='virginica')] # 簡単のため、2品種に絞るデータを覗いてみると、こんな感じです。

print(iris_df.head())

# sns.pairplot(iris_df, hue='species')
# plt.show()

X = iris_df[['petal_length']] # 説明変数
Y = iris_df['species'].map({'versicolor': 0, 'virginica': 1}) # versicolorをクラス0, virginicaをクラス1とする
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) # 80%のデータを学習データに、20%を検証データにする

lr = LogisticRegression() # ロジスティック回帰モデルのインスタンスを作成
lr.fit(X_train, Y_train) # ロジスティック回帰モデルの重みを学習

print("coefficient = ", lr.coef_)
print("intercept = ", lr.intercept_)

Y_pred = lr.predict(X_test)
# print(Y_pred)
print(accuracy_score(Y_test, Y_pred))

