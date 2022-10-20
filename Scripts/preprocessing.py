import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler

data_preproc_path = "data/df_preproc.csv"

print("Ingesting data..")
df = pd.read_csv('data/df_preproc.csv',dtype={'DatePageLink': 'Int64'}, parse_dates=['Date'], dayfirst=True)

print("Feature engineering..")

# Dropping duplicates with same date, game, turnover but different POSITIONS
df['RowTimesColumn'] = (df['Column']*df['Row']).astype(int)
df = df.drop_duplicates(subset=['Date','Game'],keep='first')
df = df.sort_values(['Date'], ascending= [True])
df.reset_index(drop=True, inplace=True)

df.to_csv('data/df_original.csv', index=False)

df = pd.read_csv('data/df_original.csv',dtype={'DatePageLink': 'Int64'}, parse_dates=['Date'], dayfirst=True)

# Field Feature Engineering
def assign_to_field(x):
    row = x['Row']
    col = x['Column']
    if (row <= 4) & (col <= 4):
        return "Field_1"
    if (row <= 8) & (col <= 8):
        return "Field_2"
    if (row <= 12) & (col <= 12):
        return "Field_3"
    else:
        return "Field_4"

df['Field'] = df.apply(lambda x: assign_to_field(x), axis=1)
dummies = pd.get_dummies(df['Field'])
df = pd.concat([df.drop(['Field'],axis=1), dummies],axis=1)

# Game occurence feature 
game_occurence = df['Game'].value_counts().rename_axis('Game').reset_index(name='Occurence_Control')
df = pd.merge(df,game_occurence,how='left',on='Game')

# RTP feature engineering
df['RTP'] = (df['RTPHigh'] + df['RTPLow'])/2 
df.drop(['RTPHigh', 'RTPLow'], axis=1, inplace=True)

# Date feature engineering
def date_to_year(x):
    # Extract year
    date = x['Date']
    year = "Year_"+str(date.year)
    return year

def date_to_month(x):
    # Extract month
    date = x['Date']
    datetime_object = datetime.datetime.strptime(str(date.month), "%m")
    month = datetime_object.strftime("%B")
    return month 

def date_to_day(x):
    # Extract day 
    date = x['Date']
    datetime_object = datetime.datetime.strptime(str(date.month), "%m")
    day = datetime_object.strftime("%A")
    return day

df['Year'] = df.apply(lambda x: date_to_year(x), axis=1)
df['Month'] = df.apply(lambda x: date_to_month(x), axis=1)
df['Day'] = df.apply(lambda x: date_to_day(x), axis=1)

# One hot encoding on cat variables
dummies = pd.get_dummies(df[['Brand', 'Gametype','Page', 'Page_Type', 'Provider_Party']])
df = pd.concat([df.drop(['Brand', 'Gametype','Page', 'Page_Type', 'Provider_Party'],axis=1), dummies],axis=1)

# One hot encoding on date variables
dummies = pd.get_dummies(df[['Year', 'Month','Day']])
df = pd.concat([df.drop(['Year', 'Month','Day'],axis=1), dummies],axis=1)

# One hot encoding on theme variables
themes_dummies = df['Theme'].str.get_dummies(sep=",")
themes_dummies.columns = themes_dummies.columns.map(lambda x: 'Theme_' + str(x))
df = pd.concat([df.drop(['Theme'],axis=1), themes_dummies],axis=1)

# Scaling numerical variables
scaler = StandardScaler()
numerical_cols = ['RTP', 'Lines', 'Reels']
df_scaled = scaler.fit_transform(df[numerical_cols])
df[numerical_cols] = df_scaled

# Dropping highly correlated with turnover variables 
df.drop(['Num_Bets','GGR','Distinct_Accounts','RowTimesColumn'], axis=1, inplace=True)

# Saving to csv 
df.to_csv('data/df_original_model_clean.csv', index=False)

print("Preprocessing done, df_original_model_clean.csv was created, data ready for modelling.")