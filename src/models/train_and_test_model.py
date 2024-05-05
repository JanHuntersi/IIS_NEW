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

import src.models.prepare_train_data as tm


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
        X.append(time_series[i:(i + look_back), :])
        y.append(time_series[i + look_back, 0]) 
    return np.array(X), np.array(y)

def histogram_plot(data):
    numeric_col = data.select_dtypes(include=[np.number])
    numeric_col.hist(bins=50, figsize=(20, 20))
    plt.hist(data, bins=50, alpha=0.75)
    plt.title('Histogram of the data')
    plt.show()

def save_train_metrics(history, file_path):
    with open(file_path, 'w') as file:
        file.write("Epoch\tTrain Loss\tValidation Loss\n")
        for epoch, (train_loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss']), start=1):
            file.write(f"{epoch}\t{train_loss}\t{val_loss}\n")

def save_test_metrics(mae, mse, evs, file_path):
    with open(file_path, 'w') as file:
        file.write("Model Metrics\n")
        file.write(f"MAE: {mae}\n")
        file.write(f"MSE: {mse}\n")
        file.write(f"EVS: {evs}\n")

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=32, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=32))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1))
    return model

def train_lstm_model(model, X_train, y_train, epochs=50, station_name = "default",test=False) :
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1)
    save_train_metrics(history, f"../../reports/{station_name}_train_metrics.txt")
    


def train_model(data_path,station_name, window_size=16, test_size_multiplier=5, test=False):


    data = pd.read_csv(data_path)

    print("FOR STATION: ",station_name)

    learn_features, data = tm.prepare_train_data(data)

    train_size = len(learn_features) - window_size * test_size_multiplier
    train_data, test_data = learn_features[:train_size], learn_features[train_size:]


    print(train_data.shape, test_data.shape)

    train_stands = np.array(train_data[:,0])
    test_stands = np.array(test_data[:,0])
    
    stands_scaler = MinMaxScaler()
    train_stands_normalized = stands_scaler.fit_transform(train_stands.reshape(-1, 1))
    test_stands_normalized = stands_scaler.transform(test_stands.reshape(-1, 1))

    train_final_stands = np.array(learn_features[:, 0])
    train_final_stands_normalized = stands_scaler.fit_transform(train_final_stands.reshape(-1, 1))

    train_other = np.array(train_data[:,1:])
    test_other = np.array(test_data[:,1:])
    other_scaler = MinMaxScaler()
    train_other_normalized = other_scaler.fit_transform(train_other)
    test_other_normalized = other_scaler.transform(test_other)

    train_final_other = np.array(learn_features[:, 1:])
    train_final_other_normalized = other_scaler.fit_transform(train_final_other)


    train_normalized = np.column_stack([train_stands_normalized, train_other_normalized])
    test_normalized = np.column_stack([test_stands_normalized, test_other_normalized])

    train_final_normalized = np.column_stack([train_final_stands_normalized, train_final_other_normalized])

    
    look_back = window_size
    step = 1

    X_train, y_train = create_dataset_with_steps(train_normalized, look_back, step)
    X_test, y_test = create_dataset_with_steps(test_normalized, look_back, step)

    X_final, y_final = create_dataset_with_steps(train_final_normalized, look_back, step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])

    X_final = X_final.reshape(X_final.shape[0], X_final.shape[2], X_final.shape[1])


    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    

    input_shape = (X_train.shape[1], X_train.shape[2])


    if(test):
        lstm_model_adv = build_lstm_model(input_shape)
        train_lstm_model(lstm_model_adv, X_train, y_train, epochs=30)




        y_test_pred_lstm_adv = lstm_model_adv.predict(X_test)

        y_test_true = stands_scaler.inverse_transform(y_test.reshape(-1, 1))

        y_test_pred_lstm_adv = stands_scaler.inverse_transform(y_test_pred_lstm_adv)

        
        lstm_mae_test, lstm_mse_test, lstm_evs_test = calculate_metrics(y_test_true, y_test_pred_lstm_adv)
        print("\nLSTM Model Metrics:")
        print(f"MAE: {lstm_mae_test}, MSE: {lstm_mse_test}, EVS: {lstm_evs_test}")

        
        with open(f'../../reports/{station_name}_metrics.txt', 'w', encoding='utf-8', errors='replace') as f:
            f.write(f'Mean average error: {lstm_mae_test}\nMean square error: {lstm_mse_test}\nExplained variance score: {lstm_evs_test}\n')



    lstm_model_final = build_lstm_model(input_shape)
    train_lstm_model(lstm_model_final, X_final, y_final, epochs=30)


    lstm_model_final.save(f'../../models/{station_name}_model.h5')
    joblib.dump(stands_scaler, f'../../models/{station_name}_scaler.pkl')
    joblib.dump(other_scaler, f'../../models/{station_name}_other_scaler.pkl')


def main():
    train_model("../../data/raw/og_dataset.csv","test",window_size=8,test=True)

if __name__ == '__main__':
    main()
