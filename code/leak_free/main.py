from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from .config import (
    CV_FOLDS,
    INPUT_CSV,
    INPUT_SHA256,
    K_BEST,
    OUTPUT_DIR,
    SEED,
    TARGET,
    TEST_SIZE,
)
from .features import prepare_customer_frame
from .io import load_dataframe, validate_input, write_json, write_records_csv
from .modeling import (
    bootstrap_auc,
    build_pipelines,
    evaluate_holdout,
    run_cv_with_oof,
    select_best,
    select_threshold,
)


def _expand_frame(frame, minimum: int = 60):
    if frame.height >= minimum:
        return frame
    repeats = int(np.ceil(minimum / frame.height))
    return pl.concat([frame] * repeats)


def _write_figures(metrics, y_test, y_prob, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "roc_curve.png", dpi=150)
    plt.close()
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP={metrics['pr_auc']:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "pr_curve.png", dpi=150)
    plt.close()


def _threshold_curve(y_true, y_prob):
    from sklearn.metrics import f1_score

    rows = []
    for t in np.round(np.arange(0.10, 0.91, 0.05), 2):
        rows.append(
            (
                float(t),
                float(f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)),
            )
        )
    return rows


def run_pipeline_from_frame(frame, output_dir=None):
    output = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    frame = _expand_frame(frame)
    X = frame.drop(["Customer_ID", TARGET]).to_numpy()
    y = frame[TARGET].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    pipelines = build_pipelines(
        pos_weight=pos_weight, k_best=min(K_BEST, X.shape[1])
    )
    scores, oof = run_cv_with_oof(pipelines, X_train, y_train)
    best_name = select_best(scores)
    threshold = select_threshold(y_train, oof[best_name])
    metrics, y_prob = evaluate_holdout(
        pipelines[best_name], X_train, y_train, X_test, y_test, threshold
    )
    low, high = bootstrap_auc(y_test, y_prob)

    output.mkdir(parents=True, exist_ok=True)
    records = [
        {"model": name, "fold": int(fold), "roc_auc": float(value)}
        for name, values in scores.items()
        for fold, value in enumerate(values)
    ]
    write_records_csv(output / "cv_results.csv", records)
    thr_rows = [
        {"threshold": t, "f1": f1_value}
        for t, f1_value in _threshold_curve(y_train, oof[best_name])
    ]
    write_records_csv(output / "threshold_selection.csv", thr_rows)
    final = {
        "best_model": best_name,
        "threshold": threshold,
        "metrics": metrics,
        "bootstrap_ci": {"low": low, "high": high},
        "cv_mean": float(np.mean(scores[best_name])),
        "cv_std": float(np.std(scores[best_name])),
    }
    write_json(output / "final_metrics.json", final)
    joblib.dump(pipelines[best_name], output / "best_model.pkl")
    _write_figures(metrics, y_test, y_prob, output)
    manifest = {
        "schema_version": 1,
        "input_sha256": INPUT_SHA256,
        "seed": SEED,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "k_best": K_BEST,
        "rows": int(frame.height),
        "best_model": best_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def run_pipeline(limit=None, output_dir=None):
    actual = validate_input(INPUT_CSV, INPUT_SHA256)
    df = load_dataframe(INPUT_CSV)
    frame = prepare_customer_frame(df)
    if limit is not None:
        frame = frame.head(limit)
    target = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    manifest = run_pipeline_from_frame(frame, output_dir=target)
    manifest["input_sha256"] = actual
    write_json(target / "manifest.json", manifest)
    return manifest
