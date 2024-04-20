import pandas as pd
import os

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_MBAJK_FILE = os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'data', 'raw', 'fetch_mbajk.csv'))
INPUT_WEATHER_FILE = os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'data', 'raw', 'weather', 'weather.csv'))
OUTPUT_FILE = os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'data', 'processed'))

def process_and_save_data(input_mbajk_file, input_weather_file, output_folder):

   # Load data for mbajk and weather
    mbajk = pd.read_csv(input_mbajk_file)
    weather = pd.read_csv(input_weather_file)

    #Rename last_update to date
    mbajk.rename(columns={'last_update':'date'}, inplace=True)

    # Rename station_name to name
    weather.rename(columns={'station_name':'number'}, inplace=True)

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


    # List of all stations
    stations = aggregated_data['number'].unique()
    
    for station in stations:
        # Filtriramo podatke za trenutno postajališče
        filtered_data = aggregated_data[aggregated_data['number'] == station]
        print(filtered_data.head())
        
        # Ime datoteke za shranjevanje podatkov
        station_path = os.path.join(output_folder, f"{station}.csv")
        
        # Shranimo filtrirane podatke v CSV datoteko
        filtered_data.to_csv(station_path, mode="a", header=not os.path.exists(station_path), index =False)

        print(f"Podatki za postajališče {station} so bili uspešno prepisani v datoteko {filtered_data}")

process_and_save_data(INPUT_MBAJK_FILE,INPUT_WEATHER_FILE, OUTPUT_FILE)
