
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)


def find_best_threshold(
    y_true,
    anomaly_scores,
    beta=2.0
):
    """
    Find the optimal anomaly threshold using
    the F-beta score on validation data.
    """

    y_true = np.asarray(y_true)
    anomaly_scores = np.asarray(anomaly_scores)

    if len(y_true) != len(anomaly_scores):
        raise ValueError("Input lengths do not match.")

    if not np.isfinite(anomaly_scores).all():
        raise ValueError("Anomaly scores contain invalid values.")

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        anomaly_scores
    )

    if len(thresholds) == 0:
        raise ValueError("Unable to calculate thresholds.")

    beta_squared = beta ** 2

    f_scores = (
        (1 + beta_squared) * precision[:-1] * recall[:-1]
        / (
            beta_squared * precision[:-1]
            + recall[:-1]
            + 1e-10
        )
    )

    best_index = np.argmax(f_scores)
    best_threshold = float(thresholds[best_index])

    predictions = (
        anomaly_scores >= best_threshold
    ).astype(int)

    return {
        "threshold": best_threshold,
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f_beta": fbeta_score(
            y_true, predictions,
            beta=beta,
            zero_division=0
        ),
        "predictions": predictions
    }


def evaluate_predictions(
    y_true,
    predictions,
    anomaly_scores=None
):
    """
    Calculate classification metrics,
    confusion matrix and classification report.
    """

    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)

    if len(y_true) != len(predictions):
        raise ValueError("Input lengths do not match.")

    metrics = {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1-Score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "F2-Score": fbeta_score(
            y_true, predictions,
            beta=2,
            zero_division=0
        )
    }

    if anomaly_scores is not None:
        anomaly_scores = np.asarray(anomaly_scores)

        if len(anomaly_scores) != len(y_true):
            raise ValueError("Score lengths do not match.")

        metrics["ROC-AUC"] = roc_auc_score(
            y_true, anomaly_scores
        )

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    report = classification_report(
        y_true,
        predictions,
        target_names=["Normal", "Anomaly"],
        zero_division=0
    )

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report
    }


def build_model_comparison(model_results):
    """
    Compare evaluation metrics across models.
    """

    comparison = pd.DataFrame(model_results)

    return comparison


def calculate_runtime_metrics(
    training_time,
    scoring_time,
    threshold_time,
    records
):
    """
    Calculate training time, inference time
    and processing throughput.
    """

    if records <= 0:
        raise ValueError("Record count must be positive.")

    if min(training_time, scoring_time, threshold_time) < 0:
        raise ValueError("Runtime values cannot be negative.")

    total_inference_time = (
        scoring_time + threshold_time
    )

    throughput = (
        records / total_inference_time
        if total_inference_time > 0
        else float("inf")
    )

    return {
        "Training Time (s)": training_time,
        "Scoring Time (s)": scoring_time,
        "Threshold Time (s)": threshold_time,
        "Total Inference Time (s)": total_inference_time,
        "Throughput (records/s)": throughput
    }
