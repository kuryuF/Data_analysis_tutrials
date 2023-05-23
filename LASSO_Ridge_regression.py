import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

housing = fetch_california_housing(as_frame=True)
# print(housing.DESCR)
data_housing = pd.DataFrame(housing.data, columns=housing.feature_names)
# print(data_housing.head())
data_housing['Price'] = housing.target
print(data_housing.head())

lr_multi = LinearRegression()

x_column_list_for_multi = [data_housing.columns[0],
                           data_housing.columns[1],
                           data_housing.columns[2],
                           data_housing.columns[3],
                           data_housing.columns[4],
                           data_housing.columns[5],
                           data_housing.columns[6],
                           data_housing.columns[7]]

y_column_list_for_multi = [data_housing.columns[8]]

lr_multi.fit(data_housing[x_column_list_for_multi], data_housing[y_column_list_for_multi])

print(lr_multi.coef_)
print(lr_multi.intercept_)

# 予測処理とMAE計算
x_train, x_test, y_train, y_test = train_test_split(
    data_housing[x_column_list_for_multi],
    data_housing[y_column_list_for_multi],
    test_size=0.3
)
lr_multi2 = LinearRegression()

lr_multi2.fit(x_train, y_train)
print(lr_multi2.coef_)
print(lr_multi2.intercept_)

y_pred = lr_multi2.predict(x_test)

# 残差
# print(y_pred - y_test)

# MAE
print(mean_absolute_error(y_pred, y_test))

# LASSO
print('============LASSO=================')
lasso = Lasso(alpha=0.1, max_iter=1000)
lasso.fit(x_train, y_train)
print(lasso.coef_)
print(lasso.intercept_)

y_pred_lasso = lasso.predict(x_test)
print(mean_absolute_error(y_pred_lasso, y_test))

# Ridge
print('============RIDGE=================')
ridge = Ridge(alpha=0.01, max_iter=10000)
ridge.fit(x_train, y_train)
print(ridge.coef_)
print(ridge.intercept_)

y_ridge_pred = ridge.predict(x_test)
print(mean_absolute_error(y_ridge_pred, y_test))



