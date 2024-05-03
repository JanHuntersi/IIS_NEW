import pandas as pd
import os

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_MBAJK_FILE = os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'data', 'raw', 'fetch_mbajk.csv'))
OUTPUT_DIR = os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'data', 'preprocessed','preprocessed_mbajk.csv'))

def preprocess_mbajk(input_mbajk_file, output_folder):
    try:

        #load mbajk data
        mbajk = pd.read_csv(input_mbajk_file)

        #  read only last 29 rows
        mbajk = mbajk.tail(29)

        mbajk.rename(columns={'last_update':'date'},inplace=True)

        mbajk = mbajk[['number','available_bikes','available_bike_stands','position','date']]
        mbajk['date'] = pd.to_datetime(mbajk['date'], unit='ms').dt.round('H')
        mbajk['date'] = mbajk['date'].dt.strftime('%Y-%m-%d %H:%M:%S+00:00')

        #if file does not exist, create it and write header
        if not os.path.exists(output_folder):
            mbajk.to_csv(output_folder, mode='w', header=True, index=False)
        else:
            mbajk.to_csv(output_folder, mode='a', header=False, index=False)
        
        print(f"Data succesfully fetched and saved to {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error: {e}")

preprocess_mbajk(INPUT_MBAJK_FILE, OUTPUT_DIR)
