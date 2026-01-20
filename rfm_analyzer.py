import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

REQUIRED_RFM_COLUMNS = ["Recency", "Frequency", "Monetary"]


def validate_rfm_columns(df):
    """Ensure required RFM columns exist."""
    missing = [c for c in REQUIRED_RFM_COLUMNS if c not in df.columns]
    if missing:
        logging.error(f"Missing RFM columns: {missing}")
        return False
    return True


def safe_qcut(series, q, labels):
    """
    Safe wrapper around qcut.
    Falls back to rank-based binning if qcut fails.
    """
    try:
        return pd.qcut(series, q=q, labels=labels, duplicates="drop")
    except Exception:
        logging.warning(f"qcut failed for {series.name}. Falling back to rank-based bins.")
        return pd.cut(series.rank(method="first"), bins=q, labels=labels)


def calculate_rfm_scores(rfm_data):
    """Calculate RFM scores and segmentation with safety checks."""

    if rfm_data is None or rfm_data.empty:
        logging.error("No RFM data available for scoring.")
        return None

    if not validate_rfm_columns(rfm_data):
        return None

    logging.info("Calculating RFM Scores and Segmentation...")

    # Assign scores using safe quantile binning
    rfm_data["R_Score"] = safe_qcut(rfm_data["Recency"], q=5, labels=[5, 4, 3, 2, 1])
    rfm_data["F_Score"] = safe_qcut(rfm_data["Frequency"], q=5, labels=[1, 2, 3, 4, 5])
    rfm_data["M_Score"] = safe_qcut(rfm_data["Monetary"], q=5, labels=[1, 2, 3, 4, 5])

    # Convert to integers
    rfm_data[["R_Score", "F_Score", "M_Score"]] = (
        rfm_data[["R_Score", "F_Score", "M_Score"]].astype(int)
    )

    # Composite score
    rfm_data["RFM_Score"] = (
        rfm_data["R_Score"] + rfm_data["F_Score"] + rfm_data["M_Score"]
    )

    # RFM group string
    rfm_data["RFM_Group"] = (
        rfm_data["R_Score"].astype(str)
        + rfm_data["F_Score"].astype(str)
        + rfm_data["M_Score"].astype(str)
    )

    logging.info(
        f"RFM scoring completed. Average RFM Score: {rfm_data['RFM_Score'].mean():.2f}"
    )

    return rfm_data


def prepare_for_clustering(rfm_data):
    """Prepare RFM data for clustering (log transform + scaling)."""

    if rfm_data is None or rfm_data.empty:
        logging.error("No RFM data available for clustering preparation.")
        return None

    if not validate_rfm_columns(rfm_data):
        return None

    logging.info("Preparing RFM data for clustering...")

    # Extract numeric features
    rfm_clustering = rfm_data[["Recency", "Frequency", "Monetary"]].copy()

    # Ensure no negative values
    rfm_clustering = rfm_clustering.clip(lower=0)

    # Log transform
    rfm_clustering = np.log1p(rfm_clustering)

    # Standardize
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_clustering)

    rfm_scaled_df = pd.DataFrame(
        rfm_scaled, columns=["Recency", "Frequency", "Monetary"]
    )

    logging.info("RFM data successfully scaled and transformed for clustering.")
    return rfm_scaled_df