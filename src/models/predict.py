import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from datetime import timedelta
import os
import src.data.fetch_weather as weather_logic
import requests



# Get the directory of the current Python script
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', '..', 'models')
models_dir = os.path.join(current_dir, '..', '..', 'models')

processed_dataset_dir = os.path.join(current_dir, '..', '..', 'data', 'processed')
og_dataeset_dir = os.path.join(current_dir, '..', '..', 'data', 'raw', 'og_dataset.csv')

def get_lang_long_from_station(station_number):
    url = "https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b"
    response = requests.get(url)
    data = response.json()
    for station in data:
        if station["number"] == station_number:
            return station["position"]["lat"], station["position"]["lng"]
    print("Station not found in the list of stations.")
    return None, None



# Path: src/models/predict.py
# For testing purposes
def predict(data, station_name = "test"):

    model_path = os.path.join(models_dir, f'{station_name}_model.h5')
    stands_path = os.path.join(models_dir, f'{station_name}_scaler.pkl')
    other_scaler_path = os.path.join(models_dir, f'{station_name}_other_scaler.pkl')


    model = tf.keras.models.load_model(model_path)
    stands_scaler = joblib.load(stands_path)
    other_scaler = joblib.load(other_scaler_path)

    data = pd.DataFrame(data)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(by=['date'])


    left_skew_columns = ["surface_pressure"]
    for col in left_skew_columns:
        data[col] = np.square(data[col])

    right_skew_columns = ["precipitation_probability", "rain"]
    for col in right_skew_columns:
        data[col] = np.log(data[col]+1 )


    selected_features = ['temperature',
        'apparent_temperature',
        'surface_pressure',
        'dew_point',
        'precipitation_probability',
        'relative_humidity', "rain"]

    learn_features = data[['available_bike_stands'] + list(selected_features) ]
    learn_features = learn_features.values

    

    stands = np.array(learn_features[:,0])

    print("we are at stands scaler")


    stands_normalized = stands_scaler.transform(stands.reshape(-1, 1))

    print("we are at other scaler")

    other = np.array(learn_features[:,1:])
    other_normalized = other_scaler.transform(other)


    normalized_data = np.column_stack([stands_normalized, other_normalized])

    X_predict = normalized_data   	
    
    X_predict = X_predict.reshape(1, X_predict.shape[1], X_predict.shape[0])
    

    prediction = model.predict(X_predict)
    prediction =  stands_scaler.inverse_transform(prediction)
    return prediction


def only_predict(data, model, stands_scaler, other_scaler):
    selected_features = ['temperature',
        'apparent_temperature',
        'surface_pressure',
        'dew_point',
        'precipitation_probability',
        'relative_humidity', "rain"]

    learn_features = data[['available_bike_stands'] + list(selected_features) ]
    learn_features = learn_features.values


    stands = np.array(learn_features[:,0])


    stands_normalized = stands_scaler.transform(stands.reshape(-1, 1))

    other = np.array(learn_features[:,1:])
    other_normalized = other_scaler.transform(other)


    normalized_data = np.column_stack([stands_normalized, other_normalized])

    X_predict = normalized_data   	
    
    X_predict = X_predict.reshape(1, X_predict.shape[1], X_predict.shape[0])
    

    prediction = model.predict(X_predict)
    prediction =  stands_scaler.inverse_transform(prediction)
    return prediction

def predict_station(station_name,windowsize=24):

    print("hello from predict_station", station_name)

    model_path = os.path.join(models_dir, f'{station_name}_model.h5')
    stands_path = os.path.join(models_dir, f'{station_name}_scaler.pkl')
    other_scaler_path = os.path.join(models_dir, f'{station_name}_other_scaler.pkl')

    model = tf.keras.models.load_model(model_path)
    stands_scaler = joblib.load(stands_path)
    other_scaler = joblib.load(other_scaler_path)


    #TODO use proccessed when its big enough
    processed_dataset_dir = os.path.join(current_dir, '..', '..', 'data', 'processed',f"{station_name}.csv")

    #use og for now
    data = pd.read_csv(og_dataeset_dir)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(by=['date'])

    #TODO REMOVE BECAUSE OF USING OG_DATASET
    print("data", data.isna().sum())
    data = data.dropna()


    data = data.tail(windowsize)

    print("data", data)

#fix skewness
    left_skew_columns = ["surface_pressure"]
    for col in left_skew_columns:
        data[col] = np.square(data[col])

    right_skew_columns = ["precipitation_probability", "rain"]
    for col in right_skew_columns:
        data[col] = np.log(data[col]+1 )

    #get latitud and longitude
    latitude , longitude = get_lang_long_from_station(station_name)
    weather_data = weather_logic.get_forecast_data(latitude, longitude, station_name)

    print("weather_data", weather_data)

    predctions = []
    for i in range(7):

        prediction = only_predict(data.copy(), model, stands_scaler, other_scaler)
        predctions.append(float(prediction[0][0]))

        if(i == 6):
            break


        last_data = data.tail(1)

        station_data = {
            "temperature": weather_data["temperature"][i], 
            "relative_humidity": weather_data["relative_humidity"][i],
            "dew_point": weather_data["dew_point"][i],
            "apparent_temperature": weather_data["apparent_temperature"][i],
            "precipitation_probability":  np.log(weather_data["precipitation_probability"][i]+1 ),
            "rain": np.log( weather_data["rain"][i]+1 ),
            "surface_pressure": np.square(weather_data["surface_pressure"][i]),
            "bike_stands": last_data["bike_stands"].values[0],
            "available_bike_stands": prediction[0][0]
        }

        station_df = pd.DataFrame(station_data, index=[0])

        data = pd.concat([data, station_df], ignore_index=True)

        if len(data) > windowsize:
            data = data.iloc[1:]

    return predctions
    