"""Audit Iris model performance by the binary ``location`` sensitive attribute."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from commons import (
    DECISION_TREE_MAX_DEPTH,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def audit_fairness(
    data_path: Path, random_state: int = DEFAULT_RANDOM_STATE
) -> MetricFrame:
    """Train an Iris classifier and return metrics grouped by ``location``."""
    iris = pd.read_csv(data_path)
    if "location" not in iris.columns:
        raise ValueError(
            "The input data needs a 'location' column. Run add_location_column.py first."
        )

    train, test = train_test_split(
        iris,
        test_size=DEFAULT_TEST_SIZE,
        stratify=iris[TARGET_COLUMN],
        random_state=random_state,
    )
    model = DecisionTreeClassifier(
        max_depth=DECISION_TREE_MAX_DEPTH,
        random_state=random_state,
    )
    model.fit(train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN])
    predictions = model.predict(test[list(FEATURE_COLUMNS)])

    metrics = {
        "accuracy": accuracy_score,
        "precision": lambda y_true, y_pred: precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall": lambda y_true, y_pred: recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }
    return MetricFrame(
        metrics=metrics,
        y_true=test[TARGET_COLUMN],
        y_pred=predictions,
        sensitive_features=test["location"],
    )


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/iris_with_location.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metric_frame = audit_fairness(args.input)
    print("Overall metrics:")
    print(metric_frame.overall.round(3))
    print("\nMetrics by location:")
    print(metric_frame.by_group.round(3))
    print("\nMaximum group difference:")
    print((metric_frame.by_group.max() - metric_frame.by_group.min()).round(3))
