from datetime import datetime
import os
import src.models.train_model as tm

# Construct the path to the 'models' directory relative to the script's directory
current_dir = os.path.dirname(__file__)
models_dir = os.path.join(current_dir, '..', '..', 'data', 'processed')
train_test_dir = os.path.join(current_dir, '..', '..', 'data', 'test_train')


def main():

    # start timer for training all models
    start_time = datetime.now()
    
    counter = 0

    for filename in os.listdir(models_dir):

    
        station_name = filename[:-4]
       
        train_path = os.path.join(train_test_dir, f"train_{filename}")
       
        print(f"Learning model for station {station_name} from file {train_path}")
        tm.train_model(train_path,station_name,window_size=8,test=False)

        counter += 1
        if counter == 2:
            break

    print("Finished training all models.")

    end_time = datetime.now()

    time_taken = end_time - start_time
    minutes_taken = time_taken.total_seconds() / 60
    seconds_taken = time_taken.total_seconds() % 60

    # Print the time it took to train all models in minutes and seconds
    print(f"Training all models took {minutes_taken} minutes and {seconds_taken} seconds.")


if __name__ == '__main__':
    main()
