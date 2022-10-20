import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_error

import catboost as cb
from xgboost import XGBRegressor

import tensorflow as tf
import gc

# CatBoostRegressor
from tabnanny import verbose

# #pip install pytorch-tabnet
#from pytorch_tabnet.tab_model import TabNetRegressor

print("Ingesting data..")
df = pd.read_csv('data/df_original_model_clean.csv',dtype={'DatePageLink': 'Int64'}, parse_dates=['Date'], dayfirst=True)

# Train test split
X = df.drop(columns=['Turnover', 'Date', 'Game', 'Location'])
y = df['Turnover']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1994)

# Dummy regressor to establish baseline
dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train, y_train)

pred = dummy_regr.predict(X_train)

print(f"The MAE score for the baseline dummy is: {mean_absolute_error(y_train, pred):.2f}")

# Ensembling and stacking

# ridge 
ridgemodel = Ridge(alpha=26)

# xgbmodel
xgbmodel = XGBRegressor(objective='reg:squarederror', 
                        n_estimators=1000, 
                        learning_rate=0.02)

# SVR 
svrmodel = SVR(C=8, 
               epsilon=0.00005, 
               gamma=0.0008)

# Huber 
hubermodel = HuberRegressor(alpha=30,
                            epsilon=3,
                            fit_intercept=True,
                            max_iter=2000)

# CatBoost
cbmodel = cb.CatBoostRegressor(loss_function='MAE',
                               colsample_bylevel=0.3, 
                               depth=2,
                               l2_leaf_reg=20, 
                               learning_rate=0.005, 
                               n_estimators=15000, 
                               subsample=0.3,
                               verbose=False)

estimators = [('ridgemodel', Ridge(alpha=26)), 
              ('svrmodel', SVR(C=8, epsilon=0.00005, gamma=0.0008)), 
              ('hubermodel', HuberRegressor(alpha=30,epsilon=3,fit_intercept=True,max_iter=10000)), 
              ('cbmodel', cb.CatBoostRegressor(loss_function='MAE',colsample_bylevel=0.3, depth=2,
                          l2_leaf_reg=20, learning_rate=0.005, n_estimators=15000, subsample=0.3,verbose=False))]

stackmodel = StackingRegressor(estimators=estimators, final_estimator=XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.02))

print("-----------------------------")
print("Overview of model performance")
print("-----------------------------")
for i in [ridgemodel,hubermodel,cbmodel,svrmodel,xgbmodel,stackmodel]:
    i.fit(X_train, y_train)

    print(f"For model {i}")
    print(f"Train MAE: {mean_absolute_error(y_train,i.predict(X_train)):.2f}")
    print(f"Test MAE: {mean_absolute_error(y_test,i.predict(X_test)):.2f}")
    print("-----------------------------")
print("-----------------------------")
print("Average Test MAE for ensemble model: ")
fit = (svrmodel.predict(X_test) + xgbmodel.predict(X_test) + stackmodel.predict(X_test) + ridgemodel.predict(X_test) + hubermodel.predict(X_test) + cbmodel.predict(X_test)) / 6
print(f"{mean_absolute_error(y_test,fit):.2f}")
print("-----------------------------")
final_prediction = (3 * xgbmodel.predict(X_test) \
+  5 * stackmodel.predict(X_test) + 3 * cbmodel.predict(X_test)) / 11

print(f"Test MAE for ensemble model: {mean_absolute_error(y_test,final_prediction):.2f}")
print("-----------------------------")
# XGBoost regressor 
reg = XGBRegressor().fit(X_train, y_train)
preds = reg.predict(X_train)
print(f"MAE on training data is: {mean_absolute_error(y_train, preds):.2f}")

preds = reg.predict(X_test)
print(f"MAE on test data is: {mean_absolute_error(y_test, preds):.2f}")
print("-----------------------------")
# Test results into bins
def calc_bin_mae(bin_test):
    test_1 = bin_test[bin_test['Turnover']<=5000]
    y_test_1 = test_1['Turnover']
    X_test_1 = test_1.drop(columns=['Turnover'])
    preds = reg.predict(X_test_1)
    print(f"MAE on test data 1 [0,5000] Turnover is: {mean_absolute_error(y_test_1, preds):.2f}")

    test_2 = bin_test[(bin_test['Turnover']>5000) & (bin_test['Turnover']<=10000)]
    y_test_2 = test_2['Turnover']
    X_test_2 = test_2.drop(columns=['Turnover'])
    preds = reg.predict(X_test_2)
    print(f"MAE on test data 2 [5000,10000] Turnover is: {mean_absolute_error(y_test_2, preds):.2f}")

    test_3 = bin_test[(bin_test['Turnover']>10000) & (bin_test['Turnover']<=15000)]
    y_test_3 = test_3['Turnover']
    X_test_3 = test_3.drop(columns=['Turnover'])
    preds = reg.predict(X_test_3)
    print(f"MAE on test data 3 [10000,15000] Turnover is: {mean_absolute_error(y_test_3, preds):.2f}")

    test_4 = bin_test[(bin_test['Turnover']>15000) & (bin_test['Turnover']<=25000)]
    y_test_4 = test_4['Turnover']
    X_test_4 = test_4.drop(columns=['Turnover'])
    preds = reg.predict(X_test_4)
    print(f"MAE on test data 4 [15000,25000] Turnover is: {mean_absolute_error(y_test_4, preds):.2f}")

    test_5 = bin_test[(bin_test['Turnover']>25000) & (bin_test['Turnover']<=35000)]
    y_test_5 = test_5['Turnover']
    X_test_5 = test_5.drop(columns=['Turnover'])
    preds = reg.predict(X_test_5)
    print(f"MAE on test data 5 [25000,35000] Turnover is: {mean_absolute_error(y_test_5, preds):.2f}")

    test_6 = bin_test[bin_test['Turnover']>35000]
    y_test_6 = test_6['Turnover']
    X_test_6 = test_6.drop(columns=['Turnover'])
    preds = reg.predict(X_test_6)
    print(f"MAE on test data 6 [35000,+] Turnover is: {mean_absolute_error(y_test_6, preds):.2f}")


bin_test = pd.concat([X_test, y_test], axis=1)
calc_bin_mae(bin_test)
print("-----------------------------")

# TabNet Regressor 

# X_train = X_train.to_numpy()
# y_train = y_train.to_numpy().reshape(-1, 1)
# X_test = X_test.to_numpy()

# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=1994)


# regressor = TabNetRegressor(verbose=1,seed=42)
# regressor.fit(X_train=X_train, y_train=y_train,
#               eval_set=[(X_valid, y_valid)],
#               patience=300, max_epochs=2000,
#               eval_metric=['mae'])

# preds = regressor.predict(X_test)
# print(f"TabNet Regressor MAE: \n{mean_absolute_error(y_test,preds):.2f}")

#print("Best performing algorithm: XGBoost Regressor")
print("-----------------------------")
print("Training a neural network..")
# Tensorflow neural net

X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8, random_state=12)
test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5, random_state=12)

tf.keras.backend.clear_session()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units = 128, activation = tf.nn.relu, input_shape = [X_train.shape[1]]),
    tf.keras.layers.Dense(units = 64, activation = tf.nn.relu),
    tf.keras.layers.Dense(units = 32, activation = tf.nn.relu),
    tf.keras.layers.Dense(units = 32, activation = tf.nn.relu),
    tf.keras.layers.Dense(units = 1)
    ])

model.compile(loss = 'mean_absolute_error', optimizer = tf.keras.optimizers.Adam(0.1), metrics = ['mae'])

early_st = tf.keras.callbacks.EarlyStopping(monitor='mae', patience=10)

tf.data.experimental.enable_debug_mode()

history = model.fit(X_train, 
                    y_train, 
                    epochs = 500, 
                    validation_data=(X_valid, y_valid),
                    callbacks=[early_st],
                    verbose=0) 

bin_test = pd.concat([X_test, y_test], axis=1)
calc_bin_mae(bin_test)

print("-----------------------------")