import mlflow
import tensorflow as tf
import tf2onnx
from mlflow.models import infer_signature
from src.helpers.tools import make_dir_if_not_exist
import os
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', '..', 'models')

def save_pipeline(mlflow_client, pipeline, station_name):
    print(f"Calling save pipeline for {station_name}")
    metadata = {
        "station_name": station_name,
    }

    pipeline = mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path=f"models/{station_name}/pipeline",
        registered_model_name=f"pipeline={station_name}",
        metadata=metadata,
    )

    pipeline_version = mlflow_client.create_model_version(
        name=f"pipeline={station_name}",
        source=pipeline.model_uri,
        run_id=pipeline.run_id
    )

    mlflow_client.transition_model_version_stage(
        name=f"pipeline={station_name}",
        stage="staging",
        version=pipeline_version.version
    )

    print(f"Saved pipeline for {station_name}")

def save_model_onnx(model,station_name,X_test,window_size=16,mlflow_client=mlflow.MlflowClient()):
 # SAVE MODEL
    model.output_names = ['output']

    input_signature = [tf.TensorSpec(shape=(None, window_size, 8), dtype=tf.double, name="input")]

    #convert model to onnx
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)

    # Log the model
    onnx_model = mlflow.onnx.log_model(
        onnx_model=onnx_model,
        artifact_path=f"models/{station_name}/model", 
        signature=infer_signature(X_test, model.predict(X_test)),
        registered_model_name=f"model={station_name}"
    )

    # Create model version

    model_version = mlflow_client.create_model_version(
        name=f"model={station_name}",
        source=onnx_model.model_uri,
        run_id=onnx_model.run_id
    )

    # Transition model version to staging
    mlflow_client.transition_model_version_stage(
        name=f"model={station_name}",
        version=model_version.version,
        stage="staging",
    )

    print(f"Saved model for {station_name}")


# save scalar
def save_scalar(mlf_flow_client,scaler, station_name, scalar_type):
    metadata = {
        "station_name": station_name,
        "scaler_type": scalar_type,
        "expected_features": scaler.n_features_in_,
        "feature_range": scaler.feature_range,
    }

    scaler = mlflow.sklearn.log_model(
        sk_model=scaler,
        artifact_path=f"models/{station_name}/{scalar_type}",
        registered_model_name=f"{scalar_type}={station_name}",
        metadata=metadata,
    )

    scaler_version = mlf_flow_client.create_model_version(
        name=f"{scalar_type}={station_name}",
        source=scaler.model_uri,
        run_id=scaler.run_id
    )

    mlf_flow_client.transition_model_version_stage(
        name=f"{scalar_type}={station_name}",
        version=scaler_version.version,
        stage="staging",
    )

    print(f"Saved {scalar_type} for {station_name}")


def download_latest_model(station_name,stage):


    model_name = f"model={station_name}"

    client = mlflow.MlflowClient()

    try:
        model = mlflow.onnx.load_model( client.get_latest_versions(name=model_name, stages=[stage])[0].source)

        # make dir if not exists
        model_path = os.path.join(models_dir,station_name)
        make_dir_if_not_exist(model_path)

        model_path = os.path.join(model_path, f"{station_name}_{stage}_model.onnx")


        mlflow.onnx.save_model(model, model_path)

        print(f"Model {model_name}  downloaded successfully")

        return model_path
    except Exception as e:
        print(f"Error: {e}")
        model = None

def download_scaler(station_name, scaler_type, stage):
    
        scaler_name = f"{scaler_type}={station_name}"
    
        client = mlflow.MlflowClient()
    
        try:
            scaler = mlflow.sklearn.load_model( client.get_latest_versions(name=scaler_name, stages=[stage])[0].source)
    
            # make dir if not exists
            scaler_path = os.path.join(models_dir,station_name)
            make_dir_if_not_exist(scaler_path)
    
            scaler_path = os.path.join(scaler_path, f"{station_name}_{stage}_{scaler_type}.pkl")
    
            joblib.dump(scaler, scaler_path)
    
            print(f"Scaler {scaler_name} downloaded successfully")
    
            return scaler
        except Exception as e:
            print(f"Error: {e}")
            scaler = None

def update_to_prod_model(station_name):
    
    client = mlflow.MlflowClient()
    model_name = f"model={station_name}"
    scaler = f"scaler={station_name}"
    other_scaler = f"other_scaler={station_name}"

    try:
        latest_prod_model = client.get_latest_versions(name=model_name, stages=["staging"])[0]
        latest_prod_scaler = client.get_latest_versions(name=scaler, stages=["staging"])[0]
        latest_prod_other_scaler = client.get_latest_versions(name=other_scaler, stages=["staging"])[0]

        client.transition_model_version_stage(name=model_name, version=latest_prod_model.version, stage="production")
        client.transition_model_version_stage(name=scaler, version=latest_prod_scaler.version, stage="production")
        client.transition_model_version_stage(name=other_scaler, version=latest_prod_other_scaler.version, stage="production")

        print(f"Model {model_name} updated to production")
    except Exception as e:
        print(f"There was an error when updating to production model Error: {e}")
        return None
    