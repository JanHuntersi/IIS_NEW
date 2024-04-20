import requests
import csv
import os
import pandas as pd
from datetime import datetime
from fetch_weather import fetch_weather_data

URL = "https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b"

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_WEATHER_DIR =  os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'data', 'raw','weather', 'weather.csv'))
SAVE_MBAJK_DIR =  os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'data', 'raw', 'fetch_mbajk.csv'))

def fetch_data(api_url):
    try:
        res = requests.get(api_url)
        if res.status_code == 200:
            data = res.json()
            return data
        else: 
            print(f"Error {res.status_code} - {res.text}")
            return None
    except requests.RequestException as e:
        print(f"Error: {e}")
        return None

data = fetch_data(URL)
if data is not None:

    print(f"Data fetched successfully, {len(data)} records found")

    
    if not os.path.exists(SAVE_WEATHER_DIR):

    
        station_weather = fetch_weather_data()
        station_weather["station_name"] = ""
        header = station_weather.keys()
        
        

        with open(SAVE_WEATHER_DIR,'w', newline='', encoding='utf-8',errors='replace') as weatherFile:
            weather_writer = csv.writer(weatherFile, lineterminator='\n')
            weather_writer.writerow(header)


    #check if file already has data 

    print(data)

    df = pd.DataFrame(data)
    
    df = df[['last_update']].sort_values(by='last_update', ascending=True)

    df['last_update'] = pd.to_datetime(df['last_update'], unit='ms')
    df['last_update'] = df['last_update'].dt.round('H')

    print(f"first time is {df.iloc[0]}" )

    df['last_update'] = df['last_update'].dt.date

    #print first row
    first = df.iloc[0]
    last = df.iloc[-1]

    date_before_first_day = first - pd.Timedelta(days=1)
    date_after_last_day = first + pd.Timedelta(days=1)

    
    for station in data:
        timestamp = pd.to_datetime(station['last_update'], unit='ms')
        rounded_timestamp = timestamp.round('H')
        formatted_timestamp = rounded_timestamp.strftime('%Y-%m-%d %H:%M:%S+00:00')

        #  print("timestamp is", formatted_timestamp)
        station_weather = fetch_weather_data(station['position']['lat'], station['position']['lng'], formatted_timestamp,date_before_first_day,date_after_last_day)
        station_weather['number'] = station['number']

        station['last_update']=formatted_timestamp
        station_weather['date']=formatted_timestamp

        print(f"station {station['name']}with timestamp {formatted_timestamp} and weather {station_weather['date']}")

# try to write to weather.csv file
        try:
            station_weather.to_csv(SAVE_WEATHER_DIR, mode='a', header=False, index=False)
        except:
            print("Errro writting to weather.csv ")

# write to mbajk.csv file        
    with open(SAVE_MBAJK_DIR, 'a', newline='', encoding='utf-8',errors='replace') as csvfile:
        csv_writer = csv.writer(csvfile, lineterminator='\n')
# Če je datoteka prazna, dodaj header
        if os.path.getsize(SAVE_MBAJK_DIR) == 0:  
            header = data[0].keys()
            csv_writer.writerow(header)

        for row in data:
            csv_writer.writerow(row.values())
        
        print(f"Data succesfully fetched and saved to {SAVE_MBAJK_DIR}")
else: 
    print("Error occurred, data not available")
