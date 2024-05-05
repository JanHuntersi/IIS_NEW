import os
from dotenv import load_dotenv


load_dotenv()

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")


# for testing purposes

print(f"mlflow_tracking_uri {mlflow_tracking_uri}")
print(f"mlflow_tracking_username {mlflow_tracking_username}")
print(f"mlflow_tracking_password {mlflow_tracking_password}")
