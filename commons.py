"""Shared configuration and schema constants for the Iris service."""

DATA_PATH = "./data/iris.csv"
MLFLOW_TRACKING_URI = "http://35.202.51.100:8100"

MODEL_NAME = "Oppe2MockDecisionTree"
MODEL_URI = f"models:/{MODEL_NAME}/latest"
MODEL_PATH = "/app/model"

FEATURE_COLUMNS = (
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
)
TARGET_COLUMN = "species"
EXPECTED_COLUMNS = (*FEATURE_COLUMNS, TARGET_COLUMN)

