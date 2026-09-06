from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from build_model import load_data, split_data

import mlflow
import mlflow.sklearn

from commons import (
    DATA_PATH,
    MLFLOW_TRACKING_URI,
    MODEL_URI,
)

data = load_data(DATA_PATH)
_, X_test, _, y_test = split_data(data)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

inference_model = mlflow.sklearn.load_model(
    model_uri=MODEL_URI
)
predictions = inference_model.predict(X_test)


print(f"- Accuracy : {accuracy_score(y_test, predictions):.3f}")
print(f"- Precision: {precision_score(y_test, predictions, average='macro'):.3f}")
print(f"- Recall   : {recall_score(y_test, predictions, average='macro'):.3f}")
print(f"- F1 Score : {f1_score(y_test, predictions, average='macro'):.3f}")
