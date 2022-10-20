import pandas as pd 
import numpy as np
import math 
import matplotlib.pyplot as plt 
import datetime
from scipy.stats import norm, skew, kurtosis 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,KFold
from sklearn.linear_model import Ridge, HuberRegressor, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

import catboost as cb
from catboost import Pool, CatBoostRegressor
from xgboost import XGBRegressor, plot_importance, cv
import xgboost as xgb

# CatBoostRegressor
from tabnanny import verbose

import tensorflow as tf
import dice_ml
from dice_ml.utils import helpers # helper functions

print("Ingesting data..")
df = pd.read_csv('data/df_original_model_clean.csv',dtype={'DatePageLink': 'Int64'}, parse_dates=['Date'], dayfirst=True)

X = df.drop(columns=['Turnover', 'Date', 'Game', 'Location'])
y = df['Turnover']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1994)

# Regressor to later use for the counterfactuals 

reg = XGBRegressor().fit(X_train, y_train)
preds = reg.predict(X_train)

# Dice - Counterfactual explanations 

df = pd.read_csv('data/df_original_model_clean.csv',dtype={'DatePageLink': 'Int64'}, parse_dates=['Date'], dayfirst=True)

X = df.drop(columns=['Date', 'Game', 'Location'])
y = df['Turnover']

continuous_features = X.drop(['Turnover'], axis=1).columns.tolist()

d_position = dice_ml.Data(dataframe=X, continuous_features=continuous_features, outcome_name="Turnover")
# We provide the type of model as a parameter (model_type)
m_position = dice_ml.Model(model=reg, backend="sklearn", model_type='regressor')

exp_genetic_position = dice_ml.Dice(d_position, m_position, method="genetic")

query_instances = X[0:2].drop(['Turnover'], axis=1)

genetic_position = exp_genetic_position.generate_counterfactuals(query_instances,
                                                               total_CFs=2,
                                                               desired_range=[20000.0, 25000.0], verbose=0, features_to_vary=['Column', 'Row'])

genetic_position.visualize_as_dataframe(show_only_changes=False)