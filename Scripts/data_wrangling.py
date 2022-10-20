import pandas as pd 
import numpy as np
from fuzzywuzzy import process

print('Data_wrangling script running...')

# CSV's paths
daily_tracking_path = 'data/iGamingtracker/igamingtracker_DailyTracking.csv'
games_path = 'data/iGamingtracker/igamingtracker_Games.csv'
themes_path = 'data/iGamingtracker/igamingtracker_GameThemes.csv'
location_path = 'data/iGamingtracker/igamingtracker_LocationGrid.csv'
pages_path = 'data/iGamingtracker/igamingtracker_Pages.csv'
performance_daily_path = 'data/performance/performance_daily.csv'
performance_metadata_path = 'data/performance/performance_metadata.csv'

print("Ingesting data..")
# Tracking dataframe has the information of: For a specific date, this game, had that position on the operator
tracking_df = pd.read_csv(daily_tracking_path,dtype={'DatePageLink': 'Int64'})

# Games dataframe
games_df = pd.read_csv(games_path)

# Game Themes 
themes_df = pd.read_csv(themes_path)

# Location Grid
locationgrid_df = pd.read_csv(location_path)

# Pages 
pages_df = pd.read_csv(pages_path)

# Performance data
performance_df = pd.read_csv(performance_daily_path)

# Performance metadata 
perf_meta_df = pd.read_csv(performance_metadata_path)
print("Done")


# COMBINING PERFORMANCE DATA 
# Get rid of any duplicates before joining
print("Combining performance data..")
new_performance_df = performance_df.drop_duplicates(
  subset = ['BEGINTIME', 'GAME_GROUP_ID'],
  keep = 'first').reset_index(drop = True)

new_perf_meta_df = perf_meta_df.drop_duplicates(
    subset = ['GAME_GROUP_ID', 'GAME_GROUP_NAME','GAME_PROVIDER_NAME','PROVIDER_PARTY'],
    keep = 'first').reset_index(drop = True)

perf_df = pd.merge(new_performance_df, new_perf_meta_df, on='GAME_GROUP_ID', how='left')

print("Dropping NAs..")
# Dropping NAs
perf_df.dropna(subset=['GAME_GROUP_NAME'], inplace=True)

# Saving all the performance dataframe for later use
perf_df.to_csv('data/performance/full_perf.csv',index=False)
print("Done, full_perf.csv was created")

# COMBINING iGamingTracker data 
print("Combining iGamingTracker data..")
# Themes - changing the representation, this way, a game has multiple themes in the same row
new_themes = themes_df.groupby(['GameID'])['Theme'].apply(','.join).reset_index()

# Combining games and themes 
games_w_themes_df = pd.merge(games_df, new_themes, on='GameID', how='left')

# Wider representation for themes - for later use 
themes_dummies = games_w_themes_df['Theme'].str.get_dummies(sep=",")
themes_dummies.columns = themes_dummies.columns.map(lambda x: 'theme_' + str(x))
wide_games_w_themes_df = games_w_themes_df.join(themes_dummies)

# Combine Location Grid with Pages 
pages_df = pages_df.rename(columns={"Page ID": "PageID"})
location_w_page_df = pd.merge(locationgrid_df, pages_df, on='PageID', how='left')

# Combine Daily Traclking with Games

# If I join tracking with games on GameID, we will end up with a wrong dataframe because
# we have a lot of records with the same gameID. However, SearchNameID is indeed unique 
# on our game's dataframe with 301445 unique values on 301445 records. 

games_w_themes_df = games_w_themes_df.rename(columns={"SearchstringID": "SearchStringID"})
daily_games_df = pd.merge(tracking_df, games_w_themes_df, on='SearchStringID', how='left')

# Combining daily_games_df with location_w_page_df

# Dropping column that already exist on Daily Tracking
location_w_page_df = location_w_page_df.drop(columns=['Location', 'column', 'Row', 'Active_x', 'PageID', 'Weighting_Tile'])
daily_tracking_df = pd.merge(daily_games_df, location_w_page_df, on='locationID', how='left')

# Saving daily_tracking_df which is the dataframe that has all the iGamingTracker data
daily_tracking_df.to_csv('data/iGamingtracker/full_tracking.csv', index=False)
print("Done, full_tracking.csv was created")


# Combining Performance data and iGamingTracker data
print("Combining performance and iGamingTracker data...")
full_perf_df = pd.read_csv('data/performance/full_perf.csv', parse_dates=['BEGINTIME'], dayfirst=True)
full_tracking_df = pd.read_csv('data/iGamingtracker/full_tracking.csv',dtype={'DatePageLink': 'Int64'}, parse_dates=['Date'], dayfirst=True)

# Matching Provider - Supplier 
common_provs = np.intersect1d(games_df['Brand'], perf_df['GAME_PROVIDER_NAME'])

unique_brands = games_df['Brand'].unique()
unique_perf_brands = perf_df['GAME_PROVIDER_NAME'].unique()

# However, we have some missing brands, because the names of the brands are not Exactly the same, 
# we have to find those brands by edit distance criteria and not just exact matching.

def match_lists_2(list1, list2):
    matches = []
    probabilities = []
    correct_matches = {}

    for i in list1:
        ratios = process.extractOne(i, list2)

        if int(ratios[1]) > 86 and int(ratios[1]) < 100:
            matches.append(ratios[0])
            probabilities.append(ratios[1])
            correct_matches[i] = ratios[0]
        else:
            matches.append('no match')
            probabilities.append('probability was too low')

    df = pd.DataFrame({
        'originals': list1,
        'potential matches': matches,
        'probability of match': probabilities
    }
    )
    

    return df, correct_matches

print("Applying Levenshtein distance to match brands..")
match_brands, correct_brands = match_lists_2(unique_brands, unique_perf_brands)
print("Done")
correct_brands

common_provs = ['Hacksaw Games','Core Gaming','Playzido','NextGen','Blueprint','Lightning Box Games','DWG','Fantasma Games','Gamevy','Push Gaming','Thunderkick']

# With this matching, we get 17687 records of TRACKING data (out of 422928 records) by games that we have PERFORMANCE
tracking_w_performance = full_tracking_df[full_tracking_df['Brand'].isin(common_provs)]

# Matching names between the tracking data and performance data

new_df = pd.merge(tracking_w_performance, full_perf_df,  how='left', left_on=['Date','Game'], right_on = ['BEGINTIME','GAME_GROUP_NAME'])
no_matches_df = new_df[new_df['TURNOVER'].isna()].copy()

# FuzzyWuzzy is a python package that utilizes Levenshtein distance but also has some ready to use functionality to process and compare lists of strings and find matches. 
# The above is very usefull for our case where we want to match mized lists of game names.

def match_lists(list1, list2):
    matches = []
    probabilities = []
    correct_matches = {}

    for i in list1:
        ratios = process.extractOne(i, list2)

        if int(ratios[1]) > 86 and int(ratios[1]) < 100:
            matches.append(ratios[0])
            probabilities.append(ratios[1])
            correct_matches[i] = ratios[0]
        else:
            matches.append('no match')
            probabilities.append('probability was too low')

    df = pd.DataFrame({
        'originals': list1,
        'potential matches': matches,
        'probability of match': probabilities
    }
    )
    

    return df, correct_matches

tracked_games = np.array(tracking_w_performance['Game'].values)
perf_games = np.array(full_perf_df['GAME_GROUP_NAME'].values)

games = tracking_w_performance['Game'].unique()
perf_games = full_perf_df['GAME_GROUP_NAME'].unique()

print("Applying Levenshtein distance to match games..")
matching_df, correct_matches = match_lists(games, perf_games)
print("Done")
correct_track_w_perf = tracking_w_performance.replace({'Game': correct_matches})

test_df = pd.merge(correct_track_w_perf, full_perf_df,  how='left', left_on=['Date','Game'], right_on = ['BEGINTIME','GAME_GROUP_NAME'])

tracking_w_no_name = test_df[test_df['TURNOVER'].isna()].copy()

final_df = test_df[test_df['TURNOVER'].notna()]

final_df.to_csv('data/final_df.csv', index=False)

# Dropping columns of no use and Handling missing values
df = pd.read_csv('data/final_df.csv',dtype={'DatePageLink': 'Int64'}, parse_dates=['Date'], dayfirst=True)

ids_cols = ['DatePageLink', 'ID', 'LocationID', 'SearchStringID', 
            'GameID_x', 'locationID', 'PageID', 'SearchNameID'  , 
            'GameID_y', 'BrandID', 'Supplier ID', 'gametypeid'  ,
            'SiteID', 'CountryID', 'PagetypeID', 'ManufacturerID',
            'Exclusive Page Manufacturer ID','ScreenshotURL'    ,
            'SiteGroupID', 'SiteTypeID', 'SiteTierID', 'Active_y', 
            'ProxyID', 'PageTypeID','GAME_GROUP_ID']

df.drop(ids_cols, axis=1, inplace=True)

# Dropping cols with no information gain
print("Dropping columns with no information gain..")
no_IG_cols = ['Country (States Combined)', 'Country','HitRate','BVGameName','Name',
              'Site', 'Continent', 'Site Group', 'URL', 'LoadType', 'Mobile/Desktop', 
              'Page Last Tracked', 'Page First Tracked', 'Site Type', 'SiteCountryTier', 
              'Weighting_Country', 'Regulated', 'Proxy', 'GAME_GROUP_NAME', 'GAME_PROVIDER_NAME',
              'SearchName', 'Supplier', 'BEGINTIME','Weighting_Page', 'Weighting_Site'
]

df.drop(no_IG_cols, axis=1, inplace=True)

renamed_cols = ['Date', 'Location_Weighting', 'Weighted_Days', 'Location', 'Column',
                'Row', 'Active', 'Weighting_Tile', 'Game', 'Brand', 'Gametype',
                'Branded_Game', 'RTPHigh', 'RTPLow', 'Lines', 'Reels', 'Free_Game',
                'Achievements', 'MinBet', 'MaxBet', 'Volatility', 'MaxExposure',
                'Theme', 'Page', 'Page_Type', 'Num_Bets', 'Turnover', 
                'GGR','Distinct_Accounts', 'Provider_Party']

df.columns = renamed_cols

print("Imputing missing values..")
df['RTPLow'].fillna(df['RTPHigh'], inplace=True)
df.drop(['MaxExposure'], axis=1, inplace=True)

# One way to fill those values and keep the same proportions is with random.choice

df['Achievements'] = df['Achievements'].fillna(pd.Series(np.random.choice([0, 1], 
                                                      p=[0.338, 0.662], size=len(df))))
df['Branded_Game'] = df['Branded_Game'].fillna(pd.Series(np.random.choice([0, 1], 
                                                      p=[0.90, 0.10], size=len(df))))
df['Free_Game'] = df['Free_Game'].fillna(pd.Series(np.random.choice([0, 1], 
                                                      p=[0.05, 0.95], size=len(df))))
df['Lines'] = df['Lines'].fillna(pd.Series(np.random.choice([10.0, 15.0, 15625.0, 20.0, 117649.0], 
                                                      p=[0.37, 0.18, 0.16, 0.15, 0.14], size=len(df))))
df['Reels'] = df['Reels'].fillna(pd.Series(np.random.choice([5.0, 6.0, 3.0, 8.0, 11.0, 7.0], 
                                                      p=[0.66, 0.27, 0.0295, 0.0295, 0.01, 0.001], size=len(df))))
df['MinBet'] = df['MinBet'].fillna(pd.Series(np.random.choice([0.10, 0.20, 0.01, 0.25, 0.30, 0.40, 0.50], 
                                                      p=[0.65, 0.26, 0.035, 0.025, 0.02, 0.006, 0.004], size=len(df))))
df['MaxBet'] = df['MaxBet'].fillna(pd.Series(np.random.choice([100.0, 10.0, 500.0, 1000.0, 20.0, 125.0, 200.0], 
                                                      p=[0.59, 0.21, 0.075, 0.065, 0.03, 0.019, 0.011], size=len(df))))
df['Volatility'] = df['Volatility'].fillna(pd.Series(np.random.choice([5.0, 4.0, 3.0, 1.0], 
                                                      p=[0.36, 0.31, 0.30, 0.03], size=len(df))))

# I will impute weighted_days with the mean, as it is a continous variable with only 0.1% missing
mean = df['Weighted_Days'].mean()
df['Weighted_Days'].fillna(mean, inplace=True)

# Imputing RTP
mean_high = df['RTPHigh'].mean()
df['RTPHigh'].fillna(mean_high, inplace=True)

mean_low = df['RTPLow'].mean()
df['RTPLow'].fillna(mean_low, inplace=True)

def calc_missing_values(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
    missing_value_df.sort_values('percent_missing', inplace=True, ascending=False)
    #print(missing_value_df.head(15))

print("Missing values: ")
calc_missing_values(df)
print("Done")

num_cols = df._get_numeric_data().columns
cat_cols = list(set(df.columns) - set(num_cols))

# Saving for later use
df.to_csv('data/df_preproc.csv', index=False)

print("Data wrangling completed, df_preproc.csv created and the preprocessing can begin.")