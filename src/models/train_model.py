import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(units=32, return_sequences=True, input_shape=input_shape))
    model.add(GRU(units=32))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1))
    return model

def calculate_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    return mae, mse, evs

def create_dataset_with_steps(time_series, look_back=1, step=1):
    X, y = [], []
    for i in range(0, len(time_series) - look_back, step):
        X.append(time_series[i:(i + look_back), 0])
        y.append(time_series[i + look_back, 0])
    return np.array(X), np.array(y)

data = pd.read_csv("../data/raw/og_dataset.csv")

# sortiranje 
data['date'] = pd.to_datetime(data['date'])

target_feature = 'available_bike_stands'
data = data[['date', target_feature]].dropna()  # Izberemo samo zapise z znanimi vrednostmi ciljne značilnice
bike_series = np.array(data[target_feature].values.reshape(-1, 1))

bike_series

train_size = len(bike_series) - 1302 -186
train_data, test_data = bike_series[:train_size], bike_series[train_size:]

scaler = MinMaxScaler()
train_data_normalized = scaler.fit_transform(train_data)
test_data_normalized = scaler.transform(test_data)


look_back = 186  # Velikost okna
step = 1  # Korak pomika

X_train, y_train = create_dataset_with_steps(train_data_normalized, look_back, step)
X_test, y_test = create_dataset_with_steps(test_data_normalized, look_back, step)

# Oblika vhodnih učnih podatkov
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

input_shape = (X_train.shape[1], X_train.shape[2])

gru_model = build_gru_model(input_shape)

gru_model.compile(optimizer='adam', loss='mean_squared_error')
gru_model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2, verbose=1)

gru_model.save('../models/base_data_model.h5')


# Evaluate the model on the TEST DATA
y_pred = gru_model.predict(X_test)


# get mae, mse, evs for test data
gru_mae_test, gru_mse_test, gru_evs_test = calculate_metrics(y_test, y_pred)

 
print("\nGRU Model Metrics:")
print(f"MAE: {gru_mae_test}, MSE: {gru_mse_test}, EVS: {gru_evs_test}")

with open('../reports/metrics.txt', 'w', encoding='utf-8', errors='replace') as f:
        f.write(f'Mean average error: {gru_mae_test}\nMean square error: {gru_mse_test}\nExplained variance score: {gru_evs_test}\n')



#TRAIN DATA

y_test_pred_gru = gru_model.predict(X_test)

y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1))

y_test_pred_gru = scaler.inverse_transform(y_test_pred_gru)

gru_mae, gru_mse, gru_evs = calculate_metrics(y_test_true, y_test_pred_gru)


print("\nGRU Model Metrics:")
print(f"MAE: {gru_mae}, MSE: {gru_mse}, EVS: {gru_evs}")

with open('../reports/train_metrics.txt', 'w', encoding='utf-8', errors='replace') as f:
        f.write(f'Mean average error: {gru_mae}\nMean square error: {gru_mse}\nExplained variance score: {gru_evs}\n')

