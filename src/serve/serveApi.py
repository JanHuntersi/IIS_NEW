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
import src.models.mlflow_helper as mlfflow_helper

def make_dir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path 


# Get the directory of the current Python script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'models' directory relative to the script's directory
models_dir = os.path.join(current_dir,  'models')

# Construct paths to individual model files within the 'models' directory
#scaler_path = os.path.join(models_dir, 'scaler.pkl')
#model_path = os.path.join(models_dir, 'base_data_model.h5')

#

def download_all_models():
    print("downloading all models--")
    for i in range(1,30):
        #make dir for station
        station_dir = make_dir_if_not_exist(os.path.join(models_dir, f'station_{i}'))

        station_dir = os.path.join(models_dir, f'station_{i}')

        model = mlfflow_helper.download_latest_model(i, "production")
        stands_scaler = mlfflow_helper.download_scaler(i, "scaler", "production")   
        other_scaler = mlfflow_helper.download_scaler(i, "other_scaler", "production")

        #save model and scalers
    print("All models downloaded")



def get_all_stations():
    url = "https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b"
    response = requests.get(url)
    data = response.json()
    return data




app = Flask(__name__)
CORS(app)

download_all_models()



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
