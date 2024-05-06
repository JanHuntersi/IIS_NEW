import pandas as pd
import os

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_MBAJK_FILE = os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'data', 'preprocessed', 'preprocessed_mbajk.csv'))
INPUT_WEATHER_FILE = os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'data', 'preprocessed', 'preprocessed_weather.csv'))
OUTPUT_FILE = os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'data', 'processed'))

def merge_and_save_data(path_mbajk,path_weather, path_output):

    #Load mbajk and weather data
    mbajk = pd.read_csv(path_mbajk)
    weather = pd.read_csv(path_weather)

    #mbajk = mbajk.tail(29)
    #weather = weather.tail(29)

    #Merge mbajk and weather data
    merged = pd.merge(mbajk, weather, on=['date','number'])

    #Aggregate data
    aggregated_data = merged.groupby(['number','date']).agg({
        'available_bikes': 'mean',
        'available_bike_stands': 'mean',
        'temperature_2m': 'mean',
        'relative_humidity_2m': 'mean',
        'dew_point_2m': 'mean',
        'apparent_temperature': 'mean',
        'precipitation_probability': 'mean',
        'precipitation': 'mean',
        'surface_pressure': 'mean'
    }).reset_index()

    #List of all stations
    stations = aggregated_data['number'].unique()

    for station in stations:
        #Filter data for current station
        filtered_data = aggregated_data[aggregated_data['number'] == station]
        print(filtered_data.head())

        #File name for saving data
        station_path = os.path.join(path_output, f"{station}.csv")

        #Save filtered data to CSV file
        filtered_data.to_csv(station_path, mode='w', header=True, index=False)

        print(f"Data for station {station} successfully saved to file {station_path}")

merge_and_save_data(INPUT_MBAJK_FILE, INPUT_WEATHER_FILE, OUTPUT_FILE)



