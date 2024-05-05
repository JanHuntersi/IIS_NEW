import os
import pandas as pd

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DATA = os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'data', 'reference_data.csv'))
CURRENT_DATA = os.path.abspath(os.path.join(CURR_DIR, '..', '..', 'data', 'current_data.csv'))

def main():
    reference_data = pd.read_csv(REFERENCE_DATA)
    current_data = pd.read_csv(CURRENT_DATA)
    print("Reference data length is: ", len(reference_data))
    print("Current data length is: ", len(current_data))

if __name__ == '__main__':
    main()
