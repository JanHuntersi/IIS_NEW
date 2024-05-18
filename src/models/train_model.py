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
import mlflow
import dagshub.auth
import dagshub
import src.environment_settings as settings
import src.models.prepare_train_data as tm
import onnxruntime as ort
import tf2onnx
from mlflow import MlflowClient
import tensorflow as tf
from mlflow.models import infer_signature
import src.models.mlflow_helper as mlfflow_helper
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

test_metrics_dir = os.path.join(current_dir, '..', '..', 'reports', 'train_metrics')

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
    #create new file or overwrite existing
    
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

def train_lstm_model(model, X_train, y_train, epochs=2, station_name = "default",test=False) :
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1)
    path = os.path.join(test_metrics_dir, f"{station_name}_train_metrics.txt")
    save_train_metrics(history, path)
    return model

def actually_train_model(data_path, station_name, window_size=16, test_size_multiplier=5, test=False,client=mlflow.MlflowClient()):
    data = pd.read_csv(data_path)

    print("FOR STATION: ",station_name)

    learn_features, data, pipeline = tm.prepare_train_data(data)

    #disabled saving pipeline
    #mlfflow_helper.save_pipeline(client,pipeline,station_name)

    train_size = len(learn_features)
    #print(learn_features.shape)

    train_stands = np.array(learn_features[:,0])
    
    stands_scaler = MinMaxScaler()
    train_stands_normalized = stands_scaler.fit_transform(train_stands.reshape(-1, 1))

    train_final_stands = np.array(learn_features[:, 0])
    train_final_stands_normalized = stands_scaler.fit_transform(train_final_stands.reshape(-1, 1))

    train_other = np.array(learn_features[:,1:])
    other_scaler = MinMaxScaler()
    train_other_normalized = other_scaler.fit_transform(train_other)

    train_final_other = np.array(learn_features[:, 1:])
    train_final_other_normalized = other_scaler.fit_transform(train_final_other)

    train_normalized = np.column_stack([train_stands_normalized, train_other_normalized])
    train_final_normalized = np.column_stack([train_final_stands_normalized, train_final_other_normalized])

    look_back = window_size
    step = 1

    X_train, y_train = create_dataset_with_steps(train_normalized, look_back, step)
    X_final, y_final = create_dataset_with_steps(train_final_normalized, look_back, step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])
    X_final = X_final.reshape(X_final.shape[0], X_final.shape[2], X_final.shape[1])

    #print(f"X_train shape: {X_train.shape}")

    input_shape = (X_train.shape[1], X_train.shape[2])

    return input_shape, X_final, y_final, train_size, stands_scaler, other_scaler

def save_scalar(mlf_flow_client,scaler, station_name, scalar_type):
    metadata = {
        "station_name": station_name,
        "scaler_type": scalar_type,
        "expected_features": scaler.n_features_in_,
        "feature_range": scaler.feature_range,
    }

    scaler = mlflow.sklearn.log_model(
        sk_model=scaler,
        artifact_path=f"models/{station_name}/{scalar_type}",
        registered_model_name=f"{scalar_type}={station_name}",
        metadata=metadata,
    )

    scaler_version = mlf_flow_client.create_model_version(
        name=f"{scalar_type}={station_name}",
        source=scaler.model_uri,
        run_id=scaler.run_id
    )

    mlf_flow_client.transition_model_version_stage(
        name=f"{scalar_type}={station_name}",
        version=scaler_version.version,
        stage="staging",
    )
    
def save_model_onnx(model,station_name,X_test,window_size=16,mlflow_client=mlflow.MlflowClient()):
 # SAVE MODEL
    model.output_names = ['output']

    input_signature = [tf.TensorSpec(shape=(None, window_size, 8), dtype=tf.double, name="input")]

    #convert model to onnx
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)

    # Log the model
    onnx_model = mlflow.onnx.log_model(
        onnx_model=onnx_model,
        artifact_path=f"models/{station_name}/model", 
        signature=infer_signature(X_test, model.predict(X_test)),
        registered_model_name=f"model={station_name}"
    )
    # Create model version

    model_version = mlflow_client.create_model_version(
        name=f"model={station_name}",
        source=onnx_model.model_uri,
        run_id=onnx_model.run_id
    )

    # Transition model version to staging
    mlflow_client.transition_model_version_stage(
        name=f"model={station_name}",
        version=model_version.version,
        stage="staging",
    )

    print(f"Saved model for {station_name}")


def train_model(data_path, station_name, window_size=16, test_size_multiplier=5, test=False):
   
    dagshub.auth.add_app_token(settings.mlflow_tracking_password)
    dagshub.init(repo_owner='JanHuntersi', repo_name='IIS_NEW', mlflow=True)
   
    print("starting run")
    print("station_name: ",station_name)
    #print("station_name: ",settings.mlflow_tracking_password)

    ml_flow_client = MlflowClient()

    experiment_name = f"station_{station_name}_train"
    mlflow.set_experiment(experiment_name)

    mlflow.set_tracking_uri("https://dagshub.com/JanHuntersi/IIS_NEW.mlflow")

    

    with mlflow.start_run(run_name=f"run={station_name}_train") as run:
        print("starting autolog")
        mlflow.tensorflow.autolog()


        input_shape, X_final, y_final, train_size, stands_scalar, other_scarlar = actually_train_model(data_path, station_name, window_size, test_size_multiplier, test, ml_flow_client)
        
        lstm_model_final = build_lstm_model(input_shape)
        lstm_model_final = train_lstm_model(lstm_model_final, X_final, y_final, epochs=2,station_name=station_name)

        mlflow.log_param("train_size", train_size)

        #save moel
        #mlfflow_helper.save_model_onnx(lstm_model_final, station_name,X_final,window_size, ml_flow_client)
        save_model_onnx(lstm_model_final,station_name,X_final,window_size,ml_flow_client)
        # SAVE SCALAR
        
        save_scalar(ml_flow_client,stands_scalar, station_name, "scaler")
        save_scalar(ml_flow_client,other_scarlar, station_name, "other_scaler")

        #mlfflow_helper.save_scalar(ml_flow_client,stands_scalar, station_name, "scaler","staging")
        #mlfflow_helper.save_scalar(ml_flow_client,other_scarlar, station_name, "other_scaler","staging")

    mlflow.end_run()
    print("ending run")

    #lstm_model_final.save(f'../../models/{station_name}_model.h5')
    #joblib.dump(stands_scaler, f'../../models/{station_name}_scaler.pkl')
    #joblib.dump(other_scaler, f'../../models/{station_name}_other_scaler.pkl')


def main():
    train_model("../../data/test_train/train_1.csv", "test", window_size=8, test=False)

if __name__ == '__main__':
    main()
