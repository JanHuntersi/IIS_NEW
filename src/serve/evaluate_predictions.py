
import os
from definitions import ROOT_DIR
import mlflow
import dagshub
from src.database.db_connector import get_prediction_today_station
import src.environment_settings as settings
import pandas as pd
from datetime import datetime,timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

path_to_data = os.path.join(ROOT_DIR, 'data', 'processed')

def assign_true_and_predicted_values(predictions, station_df):
    station_df['date'] = pd.to_datetime(station_df['date'])  # Pretvorimo stolpec 'date' v datetime format

    station_df.drop_duplicates('date', inplace=True)
    station_df.reset_index(inplace=True)

    station_df = station_df.set_index(['date'])

    mapped_mapped_prediction = []
    for prediction_with_hours in predictions:
        mapped_prediction = []
        for prediction in prediction_with_hours:
            prediction['date'] = pd.to_datetime(prediction['date'])  # Pretvorimo datum v datetime format

            #if prediction is after the last timestamp in station_df, we skip it
            last_timestamp = station_df.index[-1]
            if prediction['date'] > last_timestamp:
                print(f"SKIPPING Prediction date {prediction['date']} is after the last timestamp in station_df {last_timestamp}")
                continue


            # Poiščemo najbližji časovni žig v station_df
            nearest_timestamp = station_df.index.get_indexer([prediction['date']], method='nearest')[0]

            # Izvlečemo dejansko vrednost iz station_df
            row_with_nearest_timestamp = station_df.iloc[nearest_timestamp].to_dict()

            prediction['true'] = row_with_nearest_timestamp['available_bike_stands']
            mapped_prediction.append(prediction)
            print("mapped_prediction: ", mapped_prediction)
        if len(mapped_prediction) == 0:
            continue

        mapped_mapped_prediction.append(mapped_prediction)

    return mapped_mapped_prediction


def format_time_in_predictions(predictions):
    """
    predictions is type
    {'_id': ObjectId('664a19a62fafeabf68a965bb'), 'prediction': [16, 10, 0, 0, 0, 0, 0], 'station_id': 1, 'date': datetime.datetime(2024, 5, 19, 17, 24, 22, 109000)}
    """

    predictions_formatted = []
    hours_append = []
    for prediction in predictions:
        base_date = prediction['date']
        for i, pred_entry in enumerate(prediction['prediction']):
            new_date = base_date + timedelta(hours=i)
            hours_append.append({
                "date": new_date.strftime('%Y-%m-%d %H:%M:%S+00:00'),
                "prediction": pred_entry
            })
        predictions_formatted.append(hours_append)
        return predictions_formatted
    return predictions_formatted

def main():

    dagshub.auth.add_app_token(settings.mlflow_tracking_password)
    dagshub.init(repo_owner='JanHuntersi', repo_name='IIS_NEW', mlflow=True)
    
    
    experiment_name = f"evaluate_predictions"
    mlflow.set_experiment(experiment_name)
    
    for i in range(1, 30):
        station_name = f"station_{i}"
        print(f"starting run for station {station_name}") 

        with mlflow.start_run(run_name=f"run={station_name}_eval_pred"):
            mlflow.tensorflow.autolog()


            todays_predictions = get_prediction_today_station(station_name)
            #print("todays_predictions: ", todays_predictions)

            if not todays_predictions:
                print(f"No predictions for station {station_name} today")
                continue

            #print("todays_predictions: ", todays_predictions)

            todays_predictions = format_time_in_predictions(todays_predictions)

        # print("formated todays_predictions: ", todays_predictions)   

            if not todays_predictions:	
                print(f"ERROR: No predictions for station {station_name} today")
                continue
            station_data_path = os.path.join(path_to_data, f"{i }.csv")
            station_df = pd.read_csv(station_data_path)

            predict_true_and_predict = assign_true_and_predicted_values(todays_predictions,station_df)

            #print("Mapped predictions are : ", predict_true_and_predict)

            #print("predict_true_and_predict: ", predict_true_and_predict.count)

            # get true and predicted values
            true_values = [entry['true'] for prediction_entry in predict_true_and_predict for entry in prediction_entry]
            predicted_values = [entry['prediction'] for prediction_entry in predict_true_and_predict for entry in prediction_entry]
            
            # calculate metrics
            mae = mean_absolute_error(true_values, predicted_values)
            mse = mean_squared_error(true_values, predicted_values)
            evs = explained_variance_score(true_values, predicted_values)

            print(f"MAE: {mae}, MSE: {mse}, EVS: {evs}")

            print("Nummber of values in true_values: ", len(true_values))   

            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("evs", evs)

    mlflow.end_run()
    print(f"ending run for station {station_name}")


if __name__ == '__main__':
    main()
