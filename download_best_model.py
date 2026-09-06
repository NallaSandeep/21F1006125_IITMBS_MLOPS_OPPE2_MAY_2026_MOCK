import mlflow

from commons import MLFLOW_TRACKING_URI, MODEL_PATH, MODEL_URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.artifacts.download_artifacts(
    artifact_uri=MODEL_URI,
    dst_path=MODEL_PATH,
)
