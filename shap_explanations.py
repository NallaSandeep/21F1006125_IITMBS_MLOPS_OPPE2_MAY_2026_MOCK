"""Create full-dataset SHAP summary plots for the three Iris classes."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without requiring a graphical display.

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.tree import DecisionTreeClassifier

from commons import (
    DECISION_TREE_MAX_DEPTH,
    DEFAULT_RANDOM_STATE,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def generate_shap_summary_plots(data_path: Path, output_dir: Path) -> list[Path]:
    """Fit on the full data set and save a SHAP summary plot for every class."""
    iris = pd.read_csv(data_path)
    features = iris[list(FEATURE_COLUMNS)]
    model = DecisionTreeClassifier(
        max_depth=DECISION_TREE_MAX_DEPTH,
        random_state=DEFAULT_RANDOM_STATE,
    )
    model.fit(features, iris[TARGET_COLUMN])

    # Passing the complete feature frame makes every sample part of the explanation.
    masker = shap.maskers.Independent(features, max_samples=len(features))
    explainer = shap.Explainer(model, masker)
    shap_values = explainer(features)
    print(shap_values.shape)
    print(shap_values[..., 0].shape)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for class_index, class_name in enumerate(model.classes_):
        plt.figure()
        # The final dimension is the model-output (class) dimension.
        shap.summary_plot(shap_values[..., class_index], features, show=False)
        plt.title(f"SHAP summary: evidence for {class_name}")
        plt.tight_layout()
        output_path = output_dir / f"shap_summary_{class_name}.png"
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        output_paths.append(output_path)

    return output_paths


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/iris.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/shap"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for path in generate_shap_summary_plots(args.input, args.output_dir):
        print(f"Saved {path}")
