from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from flask_cors import CORS
from pydantic import BaseModel
import os
import src.models.predict as predict_logic
import requests


# Get the directory of the current Python script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'models' directory relative to the script's directory
models_dir = os.path.join(current_dir, '..', '..', 'models', 'default')

# Construct paths to individual model files within the 'models' directory
scaler_path = os.path.join(models_dir, 'scaler.pkl')
model_path = os.path.join(models_dir, 'base_data_model.h5')

# Load models using the constructed paths
minmax_scaler = joblib.load(scaler_path)
base_model = tf.keras.models.load_model(model_path)


def get_all_stations():
    url = "https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b"
    response = requests.get(url)
    data = response.json()
    return data




app = Flask(__name__)
CORS(app)

class BikeFeaturesRecurrent(BaseModel):
    date:str
    available_bike_stands:int

def datetime_columns(df):
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df.drop(columns=['date'], inplace=True)
    return df

def bike_prediction(bike_features):
    df = pd.DataFrame(bike_features)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    # Data transformation
    target = 'available_bike_stands'
    bikes = np.array(df[target].values.reshape(-1,1))
    bikes = minmax_scaler.transform(bikes)
    bikes = np.reshape(bikes, (bikes.shape[1], 1, bikes.shape[0]))

    # Prediction
    prediction = base_model.predict(bikes)
    prediction = minmax_scaler.inverse_transform(np.array(prediction).reshape(-1,1))

    return {'prediction': prediction.tolist()}



@app.route("/mbajk/stations", methods=['GET'])
def stations():
    return jsonify(get_all_stations())

# get info for specific station
@app.route("/mbajk/station/<int:station_id>", methods=['GET'])
def station(station_id):
    stations = get_all_stations()
    for station in stations:
        if station["number"] == station_id:
            return jsonify(station)
    return jsonify({"error": "Station not found."}), 404


@app.route("/mbajk/predict/station/<int:station_id>", methods=['GET'])
def predict_station(station_id):
    print("Get call for station id:", station_id)
    return jsonify({"predictions": predict_logic.predict_station(station_id,windowsize=8)})




@app.route("/mbajk/predict", methods=['POST'])
def predict():
    try:
        bike_features = request.json
        validate = [BikeFeaturesRecurrent(**bike_feature) for bike_feature in bike_features]
    except:
        return jsonify({'error': 'Bad request.'}), 400
    res = bike_prediction(bike_features)
    return jsonify(res)

@app.route("/health")
def health():
    return "API is alive"

@app.route("/")
def root():
    return jsonify({"message": "Hi, welcome to the bike prediction API!"})

#UNCOMMENT FOR IMAGE BUILD
#def main():
#    app.run(host='0.0.0.0', port=5000)

#if __name__ == '__main__':
#    main()


if __name__ == "__main__":
    app.run(debug=True)
