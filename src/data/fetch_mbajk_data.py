import requests
import csv
import os
import pandas as pd

URL = "https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b"

# TODO update csv file name

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
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


 #write to mbajk.csv file        
with open(SAVE_MBAJK_DIR, 'a', newline='', encoding='utf-8',errors='replace') as csvfile:
    csv_writer = csv.writer(csvfile, lineterminator='\n')
# Če je datoteka prazna, dodaj header
    if os.path.getsize(SAVE_MBAJK_DIR) == 0:  
        header = data[0].keys()
        csv_writer.writerow(header)

    for row in data:
        csv_writer.writerow(row.values())
    
    print(f"Data succesfully fetched and saved to {SAVE_MBAJK_DIR}")
