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
import src.models.mlflow_helper as mlfflow_helper
import onnxruntime

# Get the directory of the current Python script
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', '..', 'models')
test_metrics_dir = os.path.join(current_dir, '..', '..', 'reports', 'test_metrics')
processed_path = os.path.join(current_dir, '..', '..', 'data', 'test_train')

def evaluate_model(data_path, station_name, window_size):
    dagshub.auth.add_app_token(settings.mlflow_tracking_password)
    dagshub.init(repo_owner='JanHuntersi', repo_name='IIS_NEW', mlflow=True)
    print("starting run")
    print("station_name: ",station_name)

    experiment_name = f"station_{station_name}_test"
    mlflow.set_experiment(experiment_name)

    mlflow.set_tracking_uri("https://dagshub.com/JanHuntersi/IIS_NEW.mlflow")

    with mlflow.start_run(run_name=f"run={station_name}_test"):
        mlflow.tensorflow.autolog()

        """  
        #Load the model and scalers
        model_path = os.path.join(models_dir, f'{station_name}_model.h5')
        stands_path = os.path.join(models_dir, f'{station_name}_scaler.pkl')
        other_scaler_path = os.path.join(models_dir, f'{station_name}_other_scaler.pkl')

        model = tf.keras.models.load_model(model_path)
        stands_scaler = joblib.load(stands_path)
        other_scaler = joblib.load(other_scaler_path)
        """

        # Load the models via mlflow

        print(f"Trying to download latest staging model  for station {station_name}") 
        
        staging_model = mlfflow_helper.download_latest_model(station_name, "staging")
        staging_scaler = mlfflow_helper.download_scaler(station_name, "scaler", "staging")
        staging_other_scaler = mlfflow_helper.download_scaler(station_name, "other_scaler", "staging")

        if staging_model is None or staging_scaler is None or staging_other_scaler is None:
            print(f"Could not download model for station {station_name}, skipping evaluation..")
            mlflow.end_run()
            return
        
        print(f"Trying to download latest production model  for station {station_name}") 
        
        prod_model = mlfflow_helper.download_latest_model(station_name, "production")
        prod_scaler = mlfflow_helper.download_scaler(station_name, "scaler", "production")
        prod_other_scaler = mlfflow_helper.download_scaler(station_name, "other_scaler", "production")

        if prod_model is None or prod_scaler is None or prod_other_scaler is None:
            print(f"Production model does not exist for station {station_name}, using staging model..")
            mlfflow_helper.update_to_prod_model(station_name)
            mlflow.end_run()
            return
        

        #https://stackoverflow.com/questions/71279968/getting-a-prediction-from-an-onnx-model-in-python

        #print("inference session", staging_model)

        staging_model = onnxruntime.InferenceSession(staging_model.SerializeToString())
        prod_model = onnxruntime.InferenceSession(prod_model.SerializeToString())

        data = pd.read_csv(data_path)

        print("FOR STATION: ",station_name)

        learn_features, data, pipeline = ptd.prepare_train_data(data)

        stands_data = np.array(learn_features[:, 0])
        stands_normalized = staging_scaler.transform(stands_data.reshape(-1, 1))
        stands_prod_normalized = prod_scaler.transform(stands_data.reshape(-1, 1))

       

        other_data = np.array(learn_features[:, 1:])
        other_normalized = staging_other_scaler.transform(other_data)
        prod_other_normalized = prod_other_scaler.transform(other_data)

        data_normalized = np.column_stack([stands_normalized, other_normalized])
        prod_data_normalized = np.column_stack([stands_prod_normalized, prod_other_normalized])

        look_back = window_size
        step = 1


        X_final, y_final = tm.create_dataset_with_steps(data_normalized, look_back, step)
        prod_X_final, prod_y_final = tm.create_dataset_with_steps(prod_data_normalized, look_back, step)

        
        X_final = X_final.reshape(X_final.shape[0], X_final.shape[2], X_final.shape[1])
        prod_X_final = prod_X_final.reshape(prod_X_final.shape[0], prod_X_final.shape[2], prod_X_final.shape[1])

        print(f"X_final shape: {X_final.shape}")

        y_test_predicitons = staging_model.run(["output"], {"input": X_final})[0]
        prod_y_test_predicitions = prod_model.run(["output"], {"input": prod_X_final})[0]

        y_test_true = staging_scaler.inverse_transform(y_final.reshape(-1, 1))
        prod_y_test_true = prod_scaler.inverse_transform(prod_y_final.reshape(-1, 1))

        y_test_predicitons = staging_scaler.inverse_transform(y_test_predicitons)
        prod_y_test_predicitions = prod_scaler.inverse_transform(prod_y_test_predicitions)

        lstm_mae_adv, lstm_mse_adv, lstm_evs_adv = tm.calculate_metrics(y_test_true, y_test_predicitons)
        prod_mae_adv, prod_mse_adv, prod_evs_adv = tm.calculate_metrics(prod_y_test_true, prod_y_test_predicitions)

        print(f"STAGING MODEL METRICS: MAE: {lstm_mae_adv}, MSE: {lstm_mse_adv}, EVS: {lstm_evs_adv}")
        print(f"PRODUCTION MODEL METRICS: MAE: {prod_mae_adv}, MSE: {prod_mse_adv}, EVS: {prod_evs_adv}")


        mlflow.log_metric("MAE_staging", lstm_mae_adv)
        mlflow.log_metric("MSE_staging", lstm_mse_adv)
        mlflow.log_metric("EVS_staging", lstm_evs_adv)

        mlflow.log_metric("MAE_production", prod_mae_adv)
        mlflow.log_metric("MSE_production", prod_mse_adv)
        mlflow.log_metric("EVS_production", prod_evs_adv)



        # file path
        test_metrics_path = os.path.join(test_metrics_dir, f"{station_name}_test_metrics.csv")

        tm.save_test_metrics(lstm_mae_adv, lstm_mse_adv, lstm_evs_adv, test_metrics_path)

        if  lstm_mse_adv < prod_mse_adv:
            print(f"Production model is better than staging model for station {station_name}, updating to production..")
            mlfflow_helper.update_to_prod_model(station_name)

    mlflow.end_run()

def main():
    for i in range(1,3):
        print(f"Evaluating model for station {i}")
        file_path = os.path.join(processed_path, f"test_{i}.csv")
        evaluate_model(file_path,i, 8)
        #return
    
if __name__ == '__main__':
    main()
