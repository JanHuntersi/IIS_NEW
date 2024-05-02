import os
import json
import pandas as pd
from datetime import datetime
from fetch_weather import fetch_weather_data as fetch_weather
import ast

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_WEATHER_DIR =  os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'data', 'raw','weather','weather.csv'))
MBAJK_DIR =  os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'data', 'preprocessed', 'preprocessed_mbajk.csv'))

def fetch_weather_data(weather_path, mbajk_path):

    # read processed mbajk data
    mbajk = pd.read_csv(mbajk_path)

    if mbajk.empty:
        print("No data found in mbajk.csv")
        return

    # get last 29 rows

    mbajk = mbajk.tail(29)

    # sort by date
    mbajk = mbajk.sort_values(by='date', ascending=True)

    # get oldest date
    oldest_date = mbajk.iloc[0]['date']

    # get date
    oldest_date = oldest_date.split(' ')[0]
    
    # get latest date
    latest_date = mbajk.iloc[-1]['date']
    latest_date = latest_date.split(' ')[0]


    for index, row in mbajk.iterrows():

         # Convert position to dictionary
        position = ast.literal_eval(row['position'])

        lat = position['lat']
        lng = position['lng']

        timestamp = row['date']
        
        position = ast.literal_eval(row['position'])

        station_weather = fetch_weather(lat,lng, timestamp,oldest_date,latest_date)
        print(station_weather)
        station_weather['station_name'] = row['number']
        station_weather['date'] = timestamp

        try:
            #if file does not exist, create it and write header
            if not os.path.exists(weather_path):
                station_weather.to_csv(weather_path, mode='w', header=True, index=False)
            else:
                station_weather.to_csv(weather_path, mode='a', header=False, index=False)
        except:
            print("Error writing to weather.csv")
        
    print(f"Data succesfully fetched and saved to {weather_path}")
fetch_weather_data(SAVE_WEATHER_DIR, MBAJK_DIR)
