# Server Performance Anomaly Detection

## Project Overview

This project focuses on detecting abnormal server performance behaviour using Machine Learning techniques.

The main objective is to identify unusual server performance patterns and detect potential anomalies using unsupervised machine learning algorithms.

Two machine learning models are implemented and evaluated:

- K-Means Clustering
- Isolation Forest

## Project Objectives

- Understand server performance data.
- Perform Exploratory Data Analysis (EDA).
- Clean and preprocess the dataset.
- Perform feature engineering.
- Apply machine learning models for anomaly detection.
- Optimize anomaly detection thresholds.
- Evaluate and compare model performance.
- Analyze model execution time and throughput.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Machine Learning Models

### 1. K-Means Clustering

K-Means is an unsupervised clustering algorithm used to group similar server performance records.

Anomaly scores are calculated using the distance between observations and their nearest cluster centroids.

### 2. Isolation Forest

Isolation Forest is an anomaly detection algorithm that identifies unusual observations by isolating them through random partitioning.

Anomaly scores are used to identify potential abnormal server behaviour.

## Model Evaluation

Both models were evaluated using the same final testing dataset containing 61,488 records.

| Metric | K-Means | Isolation Forest |
|---|---:|---:|
| Accuracy | 0.3033 | 0.3053 |
| Precision | 0.2481 | 0.2451 |
| Recall | 0.9963 | 0.9682 |
| F1-Score | 0.3973 | 0.3912 |
| F2-Score | 0.6215 | 0.6089 |
| ROC-AUC | 0.7870 | 0.7651 |

### Evaluation Summary

K-Means achieved a higher Recall, F1-Score, F2-Score and ROC-AUC compared with Isolation Forest.

Isolation Forest achieved slightly higher Accuracy.

Based on the current evaluation results, K-Means performed slightly better overall, particularly when considering anomaly detection recall and F2-score.

However, both models produced a high number of false positives, which is an important limitation to consider.

## Project Structure

```text
server-performance-anomaly-detection/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_environment_check.ipynb
│   ├── 02_dataset_understanding.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_isolation_forest.ipynb
│   ├── 05_data_preprocessing.ipynb
│   ├── 06_feature_engineering.ipynb
│   └── 07_model_evaluation.ipynb
│
├── src/
│   ├── feature_engineering.py
│   ├── models.py
│   └── evaluation.py
│
├── results/
│   └── figures/
│
├── docs/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

1. Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Navigate to the project directory:

```bash
cd server-performance-anomaly-detection
```

3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate the virtual environment:

```bash
.venv\Scripts\activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Open Jupyter Notebook:

```bash
jupyter notebook
```

Run the notebooks in the appropriate project workflow order.

## Results

The project generates:

- Model performance comparison.
- Confusion matrices.
- ROC curves.
- Precision-Recall curves.
- Anomaly score distributions.
- Threshold optimization visualizations.
- Runtime performance comparisons.

Final evaluation CSV files are stored in the `results/` directory.

## Limitations

- Both models generate a relatively high number of false positives.
- Anomaly detection performance depends on threshold selection.
- Further feature optimization and threshold tuning may improve the results.

## Conclusion

This project demonstrates the application of unsupervised machine learning techniques for server performance anomaly detection.

Both K-Means and Isolation Forest were implemented, evaluated and compared.

Based on the current experimental results, K-Means achieved better overall anomaly detection performance according to F2-Score and ROC-AUC.

Further improvements can focus on reducing false positives and improving model reliability.

## Dataset

Dataset source and download instructions should be provided in `data/README.md`.

## Team Contributions

This project was developed collaboratively by the project team, with responsibilities distributed across data understanding, preprocessing, feature engineering, model development and evaluation.