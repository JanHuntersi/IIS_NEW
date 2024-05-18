import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

def info_gains(data):
    target = data['available_bike_stands']
    features = data.drop(columns=['available_bike_stands', 'date'])

    info_gains = mutual_info_regression(features, target)

    info_gains = pd.Series(info_gains, index=features.columns)
    info_gains.sort_values(ascending=False, inplace=True)

    return info_gains

def prepare_train_data(data):

    pipeline = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
        ('model', RandomForestRegressor())
    ])

    data['date'] = pd.to_datetime(data['date'])
    data.sort_values(by='date', inplace=True)

    #print("data features: ", data.columns)

    features = ['available_bike_stands', 'temperature_2m', 'relative_humidity_2m',
                'apparent_temperature', 'dew_point_2m', 'precipitation_probability',
                'surface_pressure','available_bikes', 'precipitation']
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
        
        pipeline.fit(X,y)

        
        missing_X = missing_data[complete_values_cols]
        predictions = pipeline.predict(missing_X)
        
        data.loc[missing_data.index, column] = predictions

    #print(data.isnull().sum())

    #AGGREGATING DATA TO INTERVAL

    data.set_index('date', inplace=True)

    agg_data = data.resample('H').mean()

    agg_data.reset_index(inplace=True)

    data = agg_data
    data.dropna(inplace=True)


    left_skew_columns = ["surface_pressure"]
    for col in left_skew_columns:
        data[col] = np.square(data[col])

    right_skew_columns = [ "precipitation_probability"]
    for col in right_skew_columns:
        data[col] = np.log(data[col]+1 )

    #INFO GAINS
    #info_gains = info_gains(data)

    selected_features = ['temperature_2m',
        'apparent_temperature',
        'surface_pressure',
        'dew_point_2m',
        'precipitation_probability',
        'relative_humidity_2m', "precipitation"]

    learn_features = data[ ['available_bike_stands']+ list(selected_features)]
    learn_features = learn_features.values
    return learn_features, data, pipeline
