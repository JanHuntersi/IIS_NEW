import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
import matplotlib.pyplot as plt

def prepare_train_data(data):

    data['date'] = pd.to_datetime(data['date'])
    data.sort_values(by='date', inplace=True)
    features = ['available_bike_stands', 'temperature', 'relative_humidity',
                'apparent_temperature', 'dew_point', 'precipitation_probability',
                'surface_pressure','bike_stands', 'rain']
    data = data[['date'] + features]

    data.isnull().sum()

    data = data.copy()
    missing_values_cols = data.columns[data.isnull().any()].tolist()
    complete_values_cols = data.drop(missing_values_cols + ["date"], axis=1).columns.tolist()

    missing_data = data[data.isnull().any(axis=1)]
    complete_data = data.dropna()

    for column in missing_values_cols:
        X = complete_data[complete_values_cols]
        y = complete_data[column]
        
        model = RandomForestRegressor()
        model.fit(X, y)
        
        missing_X = missing_data[complete_values_cols]
        predictions = model.predict(missing_X)
        
        data.loc[missing_data.index, column] = predictions


    #AGGREGATING DATA TO INTERVAL

    data.set_index('date', inplace=True)

    agg_data = data.resample('H').mean()

    agg_data.reset_index(inplace=True)

    data = agg_data
    data.dropna(inplace=True)


    left_skew_columns = ["surface_pressure"]
    for col in left_skew_columns:
        data[col] = np.square(data[col])

    right_skew_columns = ["rain", "precipitation_probability"]
    for col in right_skew_columns:
        data[col] = np.log(data[col]+1 )

    #INFO GAINS
    #target = data['available_bike_stands']
    #features = data.drop(columns=['available_bike_stands', 'date'])

    #info_gains = mutual_info_regression(features, target)

    #info_gains = pd.Series(info_gains, index=features.columns)
    #info_gains.sort_values(ascending=False, inplace=True)

    #threshold = 0.1

    #selected_features = info_gains[info_gains > threshold].index.tolist()

    selected_features = ['temperature',
        'apparent_temperature',
        'surface_pressure',
        'dew_point',
        'precipitation_probability',
        'relative_humidity', "rain"]

    learn_features = data[ ['available_bike_stands']+ list(selected_features)]
    learn_features = learn_features.values
    return learn_features, data
