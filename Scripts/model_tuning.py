import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split,KFold

print("Ingesting data..")
df = pd.read_csv('data/df_original_model_clean.csv',dtype={'DatePageLink': 'Int64'}, parse_dates=['Date'], dayfirst=True)

# Train test split
X = df.drop(columns=['Turnover', 'Date', 'Game', 'Location'])
y = df['Turnover']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1994)

print("Performing hyperparameter tuning..")

# Various hyper-parameters to tune
xgb1 = XGBRegressor(eval_metric="mae")
hyperparameter_grid = {
    'n_estimators': [100, 400, 800],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.05, 0.1, 0.20],
    'min_child_weight': [1, 10, 100]
    }

xgb_grid = GridSearchCV(xgb1,
                        hyperparameter_grid,
                        cv = 2,
                        n_jobs = -1,
                        verbose=True)

xgb_grid.fit(X_train, y_train)

print(xgb_grid.best_params_)

print("Results after hyperparameter tuning:")
preds = xgb_grid.predict(X_test)
print(f"MAE on test data is: {mean_absolute_error(y_test, preds):.2f}")

# Using Hyperopt, did not yield better results. Keeping it inside the script for reference
# commented out because it takes time.

# import packages for hyperparameters tuning
# from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
#         'gamma': hp.uniform ('gamma', 1,9),
#         'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
#         'reg_lambda' : hp.uniform('reg_lambda', 0,1),
#         'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
#         'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
#         'n_estimators': 180,
#         'seed': 0
#     }

# # Defining objective function 

# def objective(space):
#     reg=xgb.XGBRegressor(
#                     n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
#                     reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
#                     colsample_bytree=int(space['colsample_bytree']),eval_metric="mae",early_stopping_rounds=10)
    
#     evaluation = [( X_train, y_train), ( X_test, y_test)]
    
#     reg.fit(X_train, y_train,
#             eval_set=evaluation,verbose=False)
    

#     pred = reg.predict(X_test)
#     mae = mean_absolute_error(y_test, pred)
#     print ("MAE:", round(mae,2))
#     return {'loss': round(-mae,2), 'status': STATUS_OK }

# # Optimization algorithm

# trials = Trials()

# best_hyperparams = fmin(fn = objective,
#                         space = space,
#                         algo = tpe.suggest,
#                         max_evals = 200,
#                         trials = trials)

# print("The best hyperparameters are : ","\n")
# print(best_hyperparams)

# The best hyperparameters are :  

# {'colsample_bytree': 0.91153505811149, 'gamma': 2.844197638289029, 'max_depth': 18.0, 
# 'min_child_weight': 0.0, 'reg_alpha': 69.0, 'reg_lambda': 0.6438592624312813}