
import pandas as pd


def create_temporal_features(
    df,
    feature_cols=None,
    timestamp_col="timestamp_(min)",
    windows=(5, 10)
):
    """
    Create temporal features using lag and rolling statistics.

    Features generated:
    - Lag 1
    - Rolling Mean (5, 10)
    - Rolling Standard Deviation (5, 10)
    """

    data = df.copy()

    # Sort data chronologically
    if timestamp_col in data.columns:
        data = data.sort_values(
            timestamp_col
        ).reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)

    # Identify original feature columns
    if feature_cols is None:
        feature_cols = [
            col for col in data.columns
            if col.startswith("feature_")
        ]

    if not feature_cols:
        raise ValueError("No feature columns found.")

    temporal_features = {}

    for col in feature_cols:

        values = pd.to_numeric(
            data[col],
            errors="coerce"
        )

        # Lag feature
        temporal_features[f"{col}_lag_1"] = (
            values.shift(1).fillna(values)
        )

        # Rolling statistics
        for window in windows:

            rolling = values.rolling(
                window=window,
                min_periods=1
            )

            temporal_features[
                f"{col}_rolling_mean_{window}"
            ] = rolling.mean()

            temporal_features[
                f"{col}_rolling_std_{window}"
            ] = rolling.std().fillna(0)

    # Combine all generated features at once
    temporal_df = pd.DataFrame(
        temporal_features,
        index=data.index
    )

    result = pd.concat(
        [data, temporal_df],
        axis=1
    )

    return result


def get_temporal_feature_columns(df):

    """Return all generated temporal feature columns."""

    return [
        col for col in df.columns
        if "_lag_" in col
        or "_rolling_mean_" in col
        or "_rolling_std_" in col
    ]



def select_features_by_correlation(
    train_df,
    test_df,
    feature_cols=None,
    threshold=0.95
):
    """
    Remove highly correlated redundant features.

    Correlation is calculated using training data only
    to avoid data leakage.
    """

    if feature_cols is None:
        feature_cols = [
            col for col in train_df.columns
            if col.startswith("feature_")
        ]

    missing_cols = [
        col for col in feature_cols
        if col not in test_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing columns in test data: {missing_cols}"
        )

    # Calculate absolute correlation using training data
    corr_matrix = train_df[feature_cols].corr().abs()

    selected_features = []
    removed_features = []

    # Remove redundant features
    for col in feature_cols:

        correlated_with = None

        for previous_col in selected_features:

            correlation = corr_matrix.loc[
                previous_col, col
            ]

            if correlation > threshold:
                correlated_with = previous_col
                break

        if correlated_with is not None:

            removed_features.append({
                "Removed Feature": col,
                "Correlated With": correlated_with,
                "Correlation": corr_matrix.loc[
                    correlated_with, col
                ]
            })

        else:
            selected_features.append(col)

    # Apply same selected columns to both datasets
    train_selected = train_df[
        selected_features
    ].copy()

    test_selected = test_df[
        selected_features
    ].copy()

    removed_df = pd.DataFrame(removed_features)

    print("FEATURE SELECTION COMPLETED")
    print("Original Features:", len(feature_cols))
    print("Selected Features:", len(selected_features))
    print("Removed Features:", len(removed_features))

    return (
        train_selected,
        test_selected,
        selected_features,
        removed_df
    )
