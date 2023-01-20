import os
import requests
import itertools

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape

# checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# download data if it is unavailable
if 'data.csv' not in os.listdir('../Data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/data.csv', 'wb').write(r.content)

# read data
data = pd.read_csv('../Data/data.csv')

# write your code here

# STAGE-1
# X, y = data.rating, data.salary
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
#
# reg = LinearRegression()
# reg.fit(X_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))
#
# err = mape(y_test, reg.predict(X_test.values.reshape(-1, 1)))
# print(round(reg.intercept_[0], 5), round(reg.coef_[0, 0], 5), round(err, 5))


## STAGE-2
# errs = []
# for i in range(2, 5):
#     X_train, X_test, y_train, y_test = train_test_split(X**i, y, test_size=0.3, random_state=100)
#     reg = LinearRegression()
#     reg.fit(X_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))
#     errs.append(mape(y_test, reg.predict(X_test.values.reshape(-1, 1))))

# print(round(min(errs), 5))

# STAGE-3
# X, y = data.drop('salary', axis=1), data.salary
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
#
# reg = LinearRegression()
# reg.fit(X_train, y_train)
#
# err = mape(y_test, reg.predict(X_test))
# print(*reg.coef_, sep=', ')


# STAGE-4
# X, y = data.drop('salary', axis=1), data.salary
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
#
# mcorr = list()
# corr = X.corr()
# for col in corr.columns:
#     z = corr[col]
#     if not z.loc[(z > 0.2) & (z < 1)].empty:
#         mcorr.append(col)
#
# errs = []
# perms = sum([list(itertools.combinations(mcorr, i)) for i in range(1, len(mcorr))], [])
# reg = LinearRegression()
# for i in perms:
#     reg.fit(X_train.drop(list(i), axis=1), y_train)
#     errs.append(mape(y_test, reg.predict(X_test.drop(list(i), axis=1))))
#
# print(round(min(errs), 5))


# STAGE-5
X, y = data.drop('salary', axis=1), data.salary
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

mcorr = list()
corr = X.corr()

for col in corr.columns:
    z = corr[col]
    if not z.loc[(z > 0.2) & (z < 1)].empty:
        mcorr.append(col)

errs = []
perms = sum([list(itertools.combinations(mcorr, i)) for i in range(1, len(mcorr))], [])
reg = LinearRegression()

for i in perms:
    reg.fit(X_train.drop(list(i), axis=1), y_train)
    errs.append(mape(y_test, reg.predict(X_test.drop(list(i), axis=1))))

drop_cols = list(perms[(errs.index(min(errs)))])
reg.fit(X_train.drop(drop_cols, axis=1), y_train)

preds = reg.predict(X_test.drop(drop_cols, axis=1))
preds[preds < 0] = 0

zero_err = mape(y_test, preds)
preds[preds == 0] = y_train.median()
median_err = mape(y_test, preds)

print(round(zero_err, 5) if zero_err < median_err else round(median_err, 5))