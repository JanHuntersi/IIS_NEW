from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.write_concern import WriteConcern
import src.environment_settings as settings
from datetime import datetime

uri = settings.mongodb_connection

def get_client():
    try:
        client = MongoClient(uri, server_api=ServerApi(version='1'))
        return client
    except Exception as e:
        print(f"Error when connecting to mongodb: {e}")
        return None
    
def get_prediction_today_station(station_name):
    client = get_client()
    if client is None:
        print("Error when connecting to mongodb")
        return

    try:
        collection = client.get_database('predictions').get_collection(station_name)
        prediction = collection.find(sort=[("date", -1)])
        return list(prediction)
    except Exception as e:
        print(f"Error when getting prediction from MongoDB: {e}")
        return None
    finally:
        client.close()


def insert_predictions(station_id, prediction):
    client = get_client()
    if client is None:
        print("Error when connecting to mongodb")
        return

    try:
        # Check if prediction is a dictionary
        if not isinstance(prediction, dict):
            raise TypeError("Predictions must be a dictionary")

        # Add station_id and date to the prediction
        prediction['station_id'] = station_id
        prediction['date'] = datetime.now()

        collection = client.get_database('predictions').get_collection(f"station_{station_id}")
        collection.insert_one(prediction)
        print("Added prediction to db")
    except Exception as e:
        print(f"Error when inserting prediction into MongoDB: {e}")
    finally:
        client.close()

# Example usage:
# predictions = {"temperature": 23, "humidity": 60}  # This should be a dictionary
# insert_predictions(predictions, "station1")

        
