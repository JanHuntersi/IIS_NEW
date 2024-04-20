import os
import src.models.train_model as tm

# Construct the path to the 'models' directory relative to the script's directory
current_dir = os.path.dirname(__file__)
models_dir = os.path.join(current_dir, '..', '..', 'data', 'processed')


def main():

    # Print the path to the 'models' directory
    print("Path to models directory:", models_dir)

    for filename in os.listdir(models_dir):
        file_path = os.path.join(models_dir, filename)
        
        #STATION FILE PATHTO USE WHEN DATASET IS BIG ENOUGH
        relative_file_path = os.path.relpath(file_path, current_dir)
        
        station_name = filename[:-4]


        #USE OG_DATASET FOR NOW
        og_dataset_path = "../../data/raw/og_dataset.csv"
       
       #TODO USE PROCESSED DATASET WHEN DATASET IS BIG ENOUGH
       
        print(f"Learning model for station {station_name} from file {relative_file_path}")
        tm.train_model(og_dataset_path,station_name,window_size=8,test=True)
    print("Finished training all models.")


if __name__ == '__main__':
    main()
