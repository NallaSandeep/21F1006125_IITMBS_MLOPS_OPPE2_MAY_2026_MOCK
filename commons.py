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
TARGET_CLASSES = frozenset({"setosa", "versicolor", "virginica"})

DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.4
DECISION_TREE_MAX_DEPTH = 3
FEATURE_RANGES = {
    "sepal_length": (4.0, 8.0),
    "sepal_width": (2.0, 5.0),
    "petal_length": (1.0, 7.0),
    "petal_width": (0.0, 3.0),
}
