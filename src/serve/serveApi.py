from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from flask_cors import CORS
from pydantic import BaseModel
import os


# Define the directory where the models are stored
models_directory = "../../models"

# Get the absolute path of the models directory
absolute_models_directory = os.path.abspath(models_directory)

# Construct the absolute paths for scaler.pkl and base_data_model.h5
scaler_path = os.path.join(absolute_models_directory, "scaler.pkl")
model_path = os.path.join(absolute_models_directory, "base_data_model.h5")

# Load the models
minmax_scaler = joblib.load(scaler_path)
base_model = tf.keras.models.load_model(model_path)
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

if __name__ == "__main__":
    app.run(debug=True)
