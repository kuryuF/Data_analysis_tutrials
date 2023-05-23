import numpy as np
import pandas as pd
import sklearn.datasets
import matplotlib.pyplot as plt
import seaborn as sns

# データセット読み込み
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

# 説明変数
X = pd.DataFrame(housing.data, columns=housing.feature_names)
# print(X.head())

# 目的変数
Y = housing.target
Y = housing_target = pd.DataFrame(Y, columns=["Population"])
# print(Y.head())

# 変数結合
df = pd.concat([X, Y], axis=1)
print(df.head())

sns.jointplot(x=df.columns[0], y=df.columns[2],data=df)
# print(df.columns[2])

plt.tight_layout()
plt.show()

