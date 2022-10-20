import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import shap

print("Ingesting data..")
df = pd.read_csv('data/df_original_model_clean.csv',dtype={'DatePageLink': 'Int64'}, parse_dates=['Date'], dayfirst=True)

X = df.drop(columns=['Turnover', 'Date', 'Game', 'Location'])
y = df['Turnover']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1994)

print("Fitting into XGBoost regressor")
reg = XGBRegressor().fit(X_train, y_train)
preds = reg.predict(X_train)


X100 = shap.utils.sample(X, 100)

print("Computing explainer..")
# compute the SHAP values for the linear model
explainer = shap.Explainer(reg.predict, X100)
shap_values = explainer(X)

# the waterfall_plot shows how we get from shap_values.base_values to model.predict(X)[sample_ind]
shap.plots.waterfall(shap_values[2238], max_display=14)

# the waterfall_plot shows how we get from shap_values.base_values to model.predict(X)[sample_ind]
shap.plots.waterfall(shap_values[28], max_display=14)

shap.partial_dependence_plot(
    "Column", reg.predict, X100, ice=False,
    model_expected_value=True, feature_expected_value=True
)

X100 = shap.utils.sample(X, 100)

shap.partial_dependence_plot(
    "Row", reg.predict, X100, ice=False,
    model_expected_value=True, feature_expected_value=True
)