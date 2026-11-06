import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from scipy.stats import ks_2samp, wasserstein_distance


def classification_metrics(labels, predictions, bins=15):
    labels = np.asarray(labels, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    positives = labels.sum()
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("ROC-AUC requires both classes")
    ranks = pc.rank(pa.array(predictions), tiebreaker="min").to_numpy()
    _, inverse, counts = np.unique(predictions, return_inverse=True, return_counts=True)
    ranks = ranks + (counts[inverse] - 1) / 2
    auc = (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (
        positives * negatives
    )

    order = np.argsort(predictions, kind="stable")
    reliability = []
    for indices in np.array_split(order, min(bins, len(labels))):
        reliability.append({
            "rows": len(indices),
            "mean_prediction": float(predictions[indices].mean()),
            "observed_ctr": float(labels[indices].mean()),
        })
    ece = sum(
        item["rows"] * abs(item["mean_prediction"] - item["observed_ctr"])
        for item in reliability
    ) / len(labels)
    probs = predictions.clip(1e-7, 1 - 1e-7)
    return {
        "rows": len(labels),
        "ctr": float(labels.mean()),
        "auc": float(auc),
        "logloss": float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log1p(-probs))),
        "brier": float(np.mean((labels - predictions) ** 2)),
        "ece": float(ece),
        "reliability": reliability,
    }


def logits_loss(logits, labels):
    return np.mean(np.logaddexp(0, logits) - labels * logits)


def fit_temperature(predictions):
    logits = predictions["logit"].to_numpy().astype(np.float64)
    labels = predictions["label"].to_numpy()
    optimum = minimize_scalar(
        lambda log_t: logits_loss(logits / np.exp(log_t), labels),
        bounds=(np.log(0.05), np.log(20)), method="bounded",
    )
    if not optimum.success:
        raise RuntimeError("Temperature fitting failed")
    return float(np.exp(optimum.x))


def slope_and_intercept(logits, labels):
    def objective(params):
        intercept, slope = params
        scaled = intercept + slope * logits
        residual = expit(scaled) - labels
        return logits_loss(scaled, labels), np.array([residual.mean(), (residual * logits).mean()])

    result = minimize(objective, [0.0, 1.0], method="BFGS", jac=True)
    if not result.success:
        raise RuntimeError(f"Calibration regression failed: {result.message}")
    return {"intercept": float(result.x[0]), "slope": float(result.x[1])}


def calibrated_results(predictions, temperature):
    logits = predictions["logit"].to_numpy().astype(np.float64)
    labels = predictions["label"].to_numpy()
    probabilities = expit(logits / temperature)
    return (
        {
            "temperature": temperature,
            "raw_calibration": slope_and_intercept(logits, labels),
            "scaled": classification_metrics(labels, probabilities),
        },
        predictions.append_column("prediction_scaled", pa.array(probabilities)),
    )


def prediction_drift(previous, current):
    difference = np.abs(previous - current)
    cutoff = np.quantile(previous, 0.1)
    previous_std = np.std(previous)
    return {
        "mae": float(difference.mean()),
        "wasserstein": float(wasserstein_distance(previous, current)),
        "ks": float(ks_2samp(previous, current, method="asymp").statistic),
        "std_ratio": float(np.std(current) / previous_std) if previous_std else None,
        "bottom_decile_mae": float(difference[previous <= cutoff].mean()),
    }
