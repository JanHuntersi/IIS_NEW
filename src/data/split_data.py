import pandas as pd
import os

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'data', 'processed'))
OUTPUT_DIR = os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'data', 'test_train'))

def main():

    # Load data
    for filename in os.listdir(PROCESSED_DIR):
        if filename.endswith(".csv"):
            data = pd.read_csv(os.path.join(PROCESSED_DIR, filename))

            # Sort data by date
            station_data = data.sort_values(by='date', ascending=False)
            
            #Get test size
            test_size = int(0.1 * len(station_data))

            # Split data
            train_data =  station_data.iloc[test_size:]
            test_data = station_data.iloc[:test_size]

            # save data
            train_data.to_csv(os.path.join(OUTPUT_DIR, f"train_{filename}"), index=False)
            test_data.to_csv(os.path.join(OUTPUT_DIR, f"test_{filename}"), index=False)

    print("Data successfully split to train and test sets")

if __name__ == '__main__':
    main()
