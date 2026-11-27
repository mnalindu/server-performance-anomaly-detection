
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def evaluate_k_values(
    X,
    k_values=(2, 3, 4, 5, 6),
    random_state=42
):
    """
    Evaluate different K values using
    inertia and silhouette scores.
    """

    results = []

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10
        )

        labels = model.fit_predict(X)

        score = silhouette_score(
            X,
            labels,
            sample_size=min(10000, len(X)),
            random_state=random_state
        )

        results.append({
            "K": k,
            "Inertia": model.inertia_,
            "Silhouette Score": score
        })

    results_df = pd.DataFrame(results)

    best_k = int(
        results_df.loc[
            results_df["Silhouette Score"].idxmax(),
            "K"
        ]
    )

    return results_df, best_k


def calculate_anomaly_scores(model, X):
    """
    Calculate distance to nearest cluster centroid.
    Higher distance indicates a potential anomaly.
    """

    distances = model.transform(X)

    anomaly_scores = np.min(
        distances,
        axis=1
    )

    return anomaly_scores


def train_kmeans_model(
    X_train,
    X_test,
    n_clusters=2,
    threshold_percentile=99,
    random_state=42
):
    """
    Train K-Means and detect potential anomalies
    using distance-based thresholding.
    """

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )

    # Train model
    model.fit(X_train)

    # Predict cluster assignments
    train_clusters = model.predict(X_train)
    test_clusters = model.predict(X_test)

    # Calculate anomaly scores
    train_scores = calculate_anomaly_scores(
        model, X_train
    )

    test_scores = calculate_anomaly_scores(
        model, X_test
    )

    # Calculate threshold using training data
    threshold = np.percentile(
        train_scores,
        threshold_percentile
    )

    # Generate anomaly predictions
    train_anomalies = (
        train_scores > threshold
    ).astype(int)

    test_anomalies = (
        test_scores > threshold
    ).astype(int)

    return {
        "model": model,
        "train_clusters": train_clusters,
        "test_clusters": test_clusters,
        "train_scores": train_scores,
        "test_scores": test_scores,
        "threshold": threshold,
        "train_anomalies": train_anomalies,
        "test_anomalies": test_anomalies
    }


def get_cluster_anomaly_summary(
    clusters,
    anomaly_predictions
):
    """
    Generate anomaly statistics for each cluster.
    """

    df = pd.DataFrame({
        "cluster": clusters,
        "is_anomaly": anomaly_predictions
    })

    summary = df.groupby("cluster").agg(
        Total_Records=("is_anomaly", "count"),
        Anomalies=("is_anomaly", "sum"),
        Anomaly_Percentage=(
            "is_anomaly",
            lambda x: round(x.mean() * 100, 2)
        )
    )

    return summary



def analyze_clusters(
    X,
    clusters,
    anomaly_predictions,
    feature_names=None,
    top_n=15
):
    """
    Analyze cluster characteristics and identify
    features that differentiate the clusters.
    """

    # Convert input into DataFrame
    if isinstance(X, pd.DataFrame):
        feature_df = X.reset_index(drop=True).copy()
        feature_cols = feature_df.columns.tolist()

    else:
        X = np.asarray(X)

        if feature_names is None:
            feature_names = [
                f"feature_{i}" for i in range(X.shape[1])
            ]

        feature_df = pd.DataFrame(
            X,
            columns=feature_names
        )

        feature_cols = feature_names

    clusters = np.asarray(clusters)
    anomaly_predictions = np.asarray(
        anomaly_predictions
    )

    # Validate lengths
    if not (
        len(feature_df) == len(clusters)
        == len(anomaly_predictions)
    ):
        raise ValueError("Input lengths do not match.")

    # Add cluster and anomaly information
    feature_df["cluster"] = clusters
    feature_df["is_anomaly"] = anomaly_predictions

    # Calculate cluster-wise feature means
    cluster_profile = feature_df.groupby(
        "cluster"
    )[feature_cols].mean()

    # Calculate feature differences
    feature_difference = (
        cluster_profile.max(axis=0)
        - cluster_profile.min(axis=0)
    )

    # Identify top differentiating features
    top_features = (
        feature_difference
        .sort_values(ascending=False)
        .head(top_n)
    )

    top_features_df = pd.DataFrame({
        "Feature": top_features.index,
        "Absolute Difference": top_features.values
    })

    # Calculate anomaly summary
    anomaly_summary = feature_df.groupby(
        "cluster"
    ).agg(
        Total_Records=("is_anomaly", "count"),
        Anomalies=("is_anomaly", "sum"),
        Anomaly_Percentage=(
            "is_anomaly",
            lambda x: round(x.mean() * 100, 2)
        )
    )

    return {
        "cluster_profile": cluster_profile,
        "feature_difference": feature_difference,
        "top_features": top_features_df,
        "anomaly_summary": anomaly_summary
    }
