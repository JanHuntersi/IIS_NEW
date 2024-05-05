import src.environment_settings as settings
import mlflow
import dagshub.auth
import dagshub
import os
import os
import mlflow
import dagshub.auth
import dagshub
import pandas as pd
import src.environment_settings as settings
import src.models.prepare_train_data as tm
import tensorflow as tf
import joblib
import numpy as np
import src.models.train_model as tm
import src.models.prepare_train_data as ptd

# Get the directory of the current Python script
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', '..', 'models')
test_metrics_dir = os.path.join(current_dir, '..', '..', 'reports', 'test_metrics')
processed_path = os.path.join(current_dir, '..', '..', 'data', 'processed')

def evaluate_model(data_path, station_name, window_size):
    dagshub.init(repo_owner='JanHuntersi', repo_name='IIS_NEW', mlflow=True)
    print("starting run")
    print("station_name: ",settings.mlflow_tracking_password)

    experiment_name = f"Station_{station_name}_test"
    mlflow.set_experiment(experiment_name)

    mlflow.set_tracking_uri("https://dagshub.com/JanHuntersi/IIS_NEW.mlflow")

    with mlflow.start_run(run_name=f"run={station_name}_test"):
        mlflow.tensorflow.autolog()

        model_path = os.path.join(models_dir, f'{station_name}_model.h5')
        stands_path = os.path.join(models_dir, f'{station_name}_scaler.pkl')
        other_scaler_path = os.path.join(models_dir, f'{station_name}_other_scaler.pkl')

        model = tf.keras.models.load_model(model_path)
        stands_scaler = joblib.load(stands_path)
        other_scaler = joblib.load(other_scaler_path)

        data = pd.read_csv(data_path)

        print("FOR STATION: ",station_name)

        learn_features, data = ptd.prepare_train_data(data)

        stands_data = np.array(learn_features[:, 0])
        stands_normalized = stands_scaler.transform(stands_data.reshape(-1, 1))

       

        other_data = np.array(learn_features[:, 1:])
        other_normalized = other_scaler.transform(other_data)

        data_normalized = np.column_stack([stands_normalized, other_normalized])

        look_back = window_size
        step = 1


        X_final, y_final = tm.create_dataset_with_steps(data_normalized, look_back, step)

        
        X_final = X_final.reshape(X_final.shape[0], X_final.shape[2], X_final.shape[1])

        print(f"X_final shape: {X_final.shape}")

        y_test_predicitons = model.predict(X_final)

        y_test_true = stands_scaler.inverse_transform(y_final.reshape(-1, 1))

        y_test_predicitons = stands_scaler.inverse_transform(y_test_predicitons)

        lstm_mae_adv, lstm_mse_adv, lstm_evs_adv = tm.calculate_metrics(y_test_true, y_test_predicitons)

        print(f"MAE: {lstm_mae_adv}, MSE: {lstm_mse_adv}, EVS: {lstm_evs_adv}")

        mlflow.log_metric("MAE", lstm_mae_adv)
        mlflow.log_metric("MSE", lstm_mse_adv)
        mlflow.log_metric("EVS", lstm_evs_adv)

        # file path
        test_metrics_path = os.path.join(test_metrics_dir, f"{station_name}_test_metrics.csv")

        tm.save_test_metrics(lstm_mae_adv, lstm_mse_adv, lstm_evs_adv, test_metrics_path)

    mlflow.end_run()

def main():
    for i in range(1,30):
        print(f"Evaluating model for station {i}")
        file_path = os.path.join(processed_path, f"{i}.csv")
        evaluate_model(file_path,i, 8)
    
if __name__ == '__main__':
    main()
