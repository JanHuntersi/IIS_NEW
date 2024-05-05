import os
import src.models.train_model as tm

# Construct the path to the 'models' directory relative to the script's directory
current_dir = os.path.dirname(__file__)
models_dir = os.path.join(current_dir, '..', '..', 'data', 'processed')
train_test_dir = os.path.join(current_dir, '..', '..', 'data', 'test_train')


def main():

    # Print the path to the 'models' directory
    print("Path to models directory:", models_dir)

    for filename in os.listdir(models_dir):
    
        station_name = filename[:-4]
       
        train_path = os.path.join(train_test_dir, f"train_{filename}")
       
        print(f"Learning model for station {station_name} from file {train_path}")
        tm.train_model(train_path,station_name,window_size=8,test=False)
    print("Finished training all models.")


if __name__ == '__main__':
    main()
