# RUN THIS DAILY

import os
import logging
import joblib
import pandas as pd
import numpy as np

# ----- logging -----
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "model_monitoring.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)
logger.info("Monitoring system initialized")

# ----- load production package -----
package = joblib.load("readmission_production.joblib")

model = package["model"]          # calibrated pipeline
threshold = package["threshold"]
features = package["features"]

logger.info(f"Model loaded | threshold={threshold:.3f}")

# ----- score patients -----
def score_patients(df: pd.DataFrame):

    # ensure feature order and presence
    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

    df = df[features]

    # predict
    proba = model.predict_proba(df)[:, 1]
    high_risk = (proba >= threshold).astype(int)

    # log population stats
    logger.info(
        f"Scored batch | n={len(df)} | "
        f"mean_risk={proba.mean():.3f} | "
        f"high_risk_rate={high_risk.mean():.3f}"
    )

    return pd.DataFrame({
        "readmission_risk": proba,
        "high_risk": high_risk
    })

# ----- data drift detection -----
def calculate_psi(expected, actual, bins=10):
    expected_hist, bin_edges = np.histogram(expected, bins=bins)
    actual_hist, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_hist / len(expected)
    actual_pct = actual_hist / len(actual)

    psi = np.sum(
        (actual_pct - expected_pct) *
        np.log((actual_pct + 1e-6) / (expected_pct + 1e-6))
    )
    return psi


def monitor_data_drift(new_df: pd.DataFrame, psi_threshold=0.2):

    ref = pd.read_csv("artifacts/training_reference.csv")

    for col in features:
        if np.issubdtype(ref[col].dtype, np.number):
            psi = calculate_psi(ref[col], new_df[col])

            if psi > psi_threshold:
                logger.warning(f"Drift detected | {col} PSI={psi:.3f}")
            else:
                logger.info(f"Stable | {col} PSI={psi:.3f}")

# ----- performance monitoring -----
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def monitor_performance(df_with_outcomes):

    df = df_with_outcomes.copy()

    y_true = df["readmitted_30d"]
    X = df[features]

    proba = model.predict_proba(X)[:, 1]

    roc = roc_auc_score(y_true, proba)
    pr = average_precision_score(y_true, proba)
    brier = brier_score_loss(y_true, proba)

    logger.info(
        f"Performance | ROC={roc:.3f} | PR={pr:.3f} | Brier={brier:.3f}"
    )

    if roc < 0.65:
        logger.warning("Model performance degradation detected")

# ----- calibration drift monitoring -----
from sklearn.calibration import calibration_curve

def monitor_calibration(df):

    ref = pd.read_csv("artifacts/calibration_reference.csv")

    y = df["readmitted_30d"]
    X = df[features]

    proba = model.predict_proba(X)[:,1]

    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10)

    error = np.mean(np.abs(frac_pos - ref["frac_pos"]))

    logger.info(f"Calibration error={error:.3f}")

    if error > 0.05:
        logger.warning("Calibration drift detected")
        return True

    return False

# ----- threshold drift monitoring -----
def monitor_threshold(df):

    y = df["readmitted_30d"]
    X = df[features]

    proba = model.predict_proba(X)[:,1]
    preds = (proba >= threshold).astype(int)

    precision = ( (preds==1) & (y==1) ).sum() / max((preds==1).sum(),1)
    recall = ( (preds==1) & (y==1) ).sum() / y.sum()

    logger.info(f"Threshold | precision={precision:.3f} recall={recall:.3f}")

    if precision < 0.30:
        logger.warning("Threshold performance degraded")
        return True

    return False

def run_monitoring():

    data = pd.read_csv("recent_labeled_data.csv")

    drift_flag = monitor_data_drift(data)
    perf_flag = monitor_performance(data)
    cal_flag = monitor_calibration(data)
    thr_flag = monitor_threshold(data)

    return drift_flag, perf_flag, cal_flag, thr_flag


if __name__ == "__main__":
    run_monitoring()