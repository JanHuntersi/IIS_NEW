from datetime import datetime
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
import onnx
from src.database.db_connector import insert_predictions
import mlflow
import dagshub.auth
import src.environment_settings as settings


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


models_scalers = {}

def download_all_models():
    dagshub.auth.add_app_token(settings.mlflow_tracking_password)
    dagshub.init(repo_owner='JanHuntersi', repo_name='IIS_NEW', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/JanHuntersi/IIS_NEW.mlflow")

    print("downloading all models--")
    for i in range(1,30):
        #make dir for station
        
        model = mlfflow_helper.download_latest_model(i, "production")
        stands_scaler = mlfflow_helper.download_scaler(i, "scaler", "production", return_scaler=True)   
        other_scaler = mlfflow_helper.download_scaler(i, "other_scaler", "production", return_scaler=True)

        models_scalers[i] = {"model": model, "stands_scaler": stands_scaler, "other_scaler": other_scaler}
        print("Downloaded model, scalers for station:", i)

        #save model and scalers
        make_dir_if_not_exist(os.path.join(models_dir, str(i)))
        #joblib.dump(stands_scaler, os.path.join(models_dir, str(i), f'{i}_scaler.pkl'))
        #joblib.dump(other_scaler, os.path.join(models_dir, str(i), f'{i}_production_other_scaler.pkl'))
        #onnx.save_model(model, os.path.join(models_dir, str(i), f'{i}_production_model.onnx'))
        #model.save_model(os.path.join(models_dir, str(i), f'{i}_production_model.onnx'))

    print("All models downloaded")
    #print("models_scalers structure is:", models_scalers)



def get_all_stations():
    url = "https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b"
    response = requests.get(url)
    data = response.json()
    return data




app = Flask(__name__)
CORS(app)





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

    if station_id not in range(1,30):
        return jsonify({"error": "Station not found."}), 404

    print("Get call for station id:", station_id)
    
    prediction = predict_logic.predict_station(station_id,models_scalers[station_id],windowsize=8)

    prediction = [abs(x) for x in prediction]
    final_predictions = [max(0,round(x)) for x in prediction]
    print("Predictions for station:", station_id, "are:", final_predictions)

    #add prediction to mongo
    insert_predictions(station_id, {"prediction": final_predictions,"station_id":station_id,"date":datetime.now()})

    ret_pred = {"prediction": final_predictions}
    return jsonify(ret_pred)



@app.route("/health")
def health():
    return "API is alive"

@app.route("/")
def root():
    return jsonify({"message": "Hi, welcome to the bike prediction API!"})

#UNCOMMENT FOR IMAGE BUILD
def main():
    download_all_models()
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()


#if __name__ == "__main__":
#    #download_all_models()
#    app.run(debug=True)
    
