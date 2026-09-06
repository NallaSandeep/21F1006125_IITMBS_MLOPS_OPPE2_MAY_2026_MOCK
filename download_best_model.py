import mlflow

mlflow.set_tracking_uri("http://35.202.51.100:8100")

model_uri = "models:/Oppe2MockDecisionTree/latest"

mlflow.artifacts.download_artifacts(
    artifact_uri=model_uri,
    dst_path="/app/model"
)