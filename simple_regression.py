import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression

# データセット読込
iris = datasets.load_iris()

# 説明変数
X = iris.data
X = pd.DataFrame(X, columns=iris.feature_names)
# print(X.head())

# 目的変数
Y = iris.target
Y = iris_target = pd.DataFrame(Y, columns = ["Species"])

# 変数結合
df = pd.concat([X,Y],axis=1)
# print(df.head())

sns.jointplot(x=df.columns[0],y=df.columns[2],data=df)

# 単回帰分析
lr = LinearRegression()
x_column_list = ['sepal length (cm)']
y_column_list = ['petal length (cm)']

data_iri_x = X[x_column_list]
data_iri_y = X[y_column_list]

mod = lr.fit(data_iri_x, data_iri_y)
y_lin_fit = mod.predict(data_iri_x)
plt.plot(data_iri_x, y_lin_fit, color = '#000000', linewidth=0.5)


print(lr.coef_)			# w0（比例定数）
print(lr.intercept_)	# W1（切片）
# => y = w0 * x + w1

plt.tight_layout()
plt.show()
# plt.savefig('jointplot.jpg')
