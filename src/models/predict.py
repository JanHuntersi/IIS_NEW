import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from datetime import timedelta
import os
import src.data.fetch_weather as weather_logic
import requests
import onnxruntime as ort



# Get the directory of the current Python script
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', 'serve', 'models')

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

    model_path = os.path.join(models_dir,station_name, f'{station_name}_production_model.onnx')
    stands_path = os.path.join(models_dir, f'{station_name}_scaler.pkl')
    other_scaler_path = os.path.join(models_dir, f'{station_name}_production_other_scaler.pkl')


    model =  ort.InferenceSession(model_path)
    stands_scaler = joblib.load(stands_path)
    other_scaler = joblib.load(other_scaler_path)

    data = pd.DataFrame(data)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(by=['date'])


    left_skew_columns = ["surface_pressure"]
    for col in left_skew_columns:
        data[col] = np.square(data[col])

    right_skew_columns = ["precipitation_probability"]
    for col in right_skew_columns:
        data[col] = np.log(data[col]+1 )


    selected_features = ['available_bike_stands', 'temperature_2m', 'relative_humidity_2m',
                'apparent_temperature', 'dew_point_2m', 'precipitation_probability',
                'surface_pressure','available_bikes', 'precipitation']

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
    

    prediction = model.run(["output"],{"input":X_predict})[0]
    prediction =  stands_scaler.inverse_transform(prediction)
    return prediction


def make_prediction(data, model, stands_scaler, other_scaler):
    selected_features = ['temperature_2m',
        'apparent_temperature',
        'surface_pressure',
        'dew_point_2m',
        'precipitation_probability',
        'relative_humidity_2m', 'precipitation']

    learn_features = data[['available_bike_stands'] + list(selected_features) ]
    learn_features = learn_features.values


    stands = np.array(learn_features[:,0])


    stands_normalized = stands_scaler.transform(stands.reshape(-1, 1))

    other = np.array(learn_features[:,1:])
    other_normalized = other_scaler.transform(other)


    normalized_data = np.column_stack([stands_normalized, other_normalized])

    X_predict = normalized_data   	
    
    X_predict = X_predict.reshape(1, X_predict.shape[1], X_predict.shape[0])
    

    prediction = model.run(["output"], {"input":X_predict})[0]
    prediction =  stands_scaler.inverse_transform(prediction)
    return prediction

def predict_station(station_name,models_scalers_dict,windowsize=24):

    print("hello from predict_station", station_name)

    #models_station_dir = os.path.join(models_dir,f"{station_name}")
    #model_path = os.path.join(models_station_dir, f'{station_name}_production_model.onnx')
    #stands_path = os.path.join(models_station_dir, f'{station_name}_scaler.pkl')
    #other_scaler_path = os.path.join(models_station_dir, f'{station_name}_production_other_scaler.pkl')

    model_scaler_station = models_scalers_dict[station_name]

    model =  ort.InferenceSession(model_scaler_station["model"].SerializeToString())
    #stands_scaler = joblib.load(stands_path)
    #other_scaler = joblib.load(other_scaler_path) 
    stands_scaler = model_scaler_station["stands_scaler"]
    other_scaler = model_scaler_station["other_scaler"]


    processed_dataset_dir = os.path.join(current_dir, '..', '..', 'data', 'processed',f"{station_name}.csv")

    #use og for now
    data = pd.read_csv(processed_dataset_dir)

    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(by=['date'])

    #TODO REMOVE BECAUSE OF USING OG_DATASET
    data = data.dropna()


    data = data.tail(windowsize)

   # print("data", data)

#fix skewness
    left_skew_columns = ["surface_pressure"]
    for col in left_skew_columns:
        data[col] = np.square(data[col])

    right_skew_columns = ["precipitation_probability"]
    for col in right_skew_columns:
        data[col] = np.log(data[col]+1 )

    #get latitud and longitude
    latitude , longitude = get_lang_long_from_station(station_name)
    weather_data = weather_logic.get_forecast_data(latitude, longitude, station_name)

    print("weather_data", weather_data)

    predctions = []
    for i in range(7):
        try:
            prediction = make_prediction(data.copy(), model, stands_scaler, other_scaler)
            predctions.append(float(prediction[0][0]))

            if(i == 6):
                break

            last_data = data.tail(1)

            station_data = {
                "temperature_2m": weather_data["temperature_2m"][i], 
                "relative_humidity_2m": weather_data["relative_humidity_2m"][i],
                "dew_point_2m": weather_data["dew_point_2m"][i],
                "apparent_temperature": weather_data["apparent_temperature"][i],
                "precipitation_probability":  np.log(weather_data["precipitation_probability"][i]+1 ),
                "surface_pressure": np.square(weather_data["surface_pressure"][i]),
                "available_bikes": last_data["available_bikes"].values[0],
                "available_bike_stands": prediction[0][0]
            }

            station_df = pd.DataFrame(station_data, index=[0])

            data = pd.concat([data, station_df], ignore_index=True)

            if len(data) > windowsize:
                data = data.iloc[1:]
        except Exception as e:
            print("Error in prediction", e)
            break
    return predctions
    