import pandas as pd
import os

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_WEATHER_DIR = os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'data', 'raw','weather','test_weather.csv'))
OUTPUT_DIR = os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'data', 'preprocessed','preprocessed_weather.csv'))

def preprocess_weather(input_weather_file, output_folder):
    try:

        #load weather data
        weather = pd.read_csv(input_weather_file)

        # TODO read only last 29 rows
        #weather = weather.tail(29)
        
        #rename weather
        weather.rename(columns={'station_name':'number'},inplace=True)

        #if file does not exist, create it and write header
        if not os.path.exists(output_folder):
            weather.to_csv(output_folder, mode='w', header=True, index=False)
        else:
            weather.to_csv(output_folder, mode='a', header=False, index=False)
        
        print(f"Data succesfully fetched and saved to {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error: {e}")

preprocess_weather(INPUT_WEATHER_DIR, OUTPUT_DIR)
