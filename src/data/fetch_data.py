import requests
import csv
import os
from datetime import datetime

URL = "https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b"
SAVE_FILE = r'C:\Users\Jan\Desktop\FERI\IPT\2-SEMESTER\IIS\vaje\data\raw\fetch_mbajk.csv'

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
    with open(SAVE_FILE, 'a', newline='', encoding='utf-8', errors='replace') as csvfile:
        csv_writer = csv.writer(csvfile, lineterminator='\n')

        if os.path.getsize(SAVE_FILE) == 0:  # Če je datoteka prazna, dodaj header
            header = data[0].keys()
            csv_writer.writerow(header)

        for row in data:
            csv_writer.writerow(row.values())
        
        print(f"Data succesfully fetched and saved to {SAVE_FILE}")
else: 
    print("Error occurred, data not available")
