import os
import pandas as pd
import numpy as np
from src.config import AppConfig
from src.reproducibility import set_seed
from src.dataset import (
    build_dataset_manifest,
    split_dataset_unsupervised,
    bootstrap_synthetic_dataset_if_empty
)
from src.pipelines import PIPELINE_VARIANTS, extract_features_from_files
from src.models import UnsupervisedAnomalyDetector
from src.evaluation import calculate_quantitative_metrics
from src.robustness import evaluate_noise_robustness
from src.visualization import (
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrices,
    plot_ablation_comparison,
    plot_noise_robustness
)
from src.error_analysis import perform_error_analysis

def train_and_evaluate_all_pipelines(cfg: AppConfig, logger=None, progress_callback=None):
    set_seed(cfg.random_seed)

    # 1. Bootstrap dataset & manifest
    bootstrap_synthetic_dataset_if_empty(
        cfg.data_root,
        num_normal=40,
        num_abnormal=15,
        sample_rate=cfg.sample_rate,
        duration_seconds=cfg.duration_seconds
    )
    df_manifest = build_dataset_manifest(cfg.data_root)

    os.makedirs(cfg.results_dir, exist_ok=True)
    df_manifest.to_csv(os.path.join(cfg.results_dir, "dataset_manifest.csv"), index=False)

    # 2. Strict Unsupervised Split
    df_train_norm, df_eval = split_dataset_unsupervised(
        df_manifest, train_ratio=cfg.train_ratio, seed=cfg.random_seed
    )

    trained_detectors = {}
    eval_data_dict = {}
    summary_records = []
    threshold_records = []

    total_variants = len(PIPELINE_VARIANTS)

    for idx, pipeline_name in enumerate(PIPELINE_VARIANTS):
        if progress_callback:
            progress_callback(float(idx) / (total_variants + 1), f"Training Pipeline: '{pipeline_name}'...")

        if logger:
            logger.info(f"--- Training Pipeline Variant: {pipeline_name} ---")

        # Extract features for training (Normal ONLY)
        X_train, _, _ = extract_features_from_files(df_train_norm, pipeline_name, cfg)

        # Fit Unsupervised Anomaly Detector
        detector = UnsupervisedAnomalyDetector(
            model_type=cfg.model_type,
            input_dim=X_train.shape[1],
            hidden_dims=cfg.hidden_dims,
            learning_rate=cfg.learning_rate,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            threshold_percentile=cfg.threshold_percentile,
            random_seed=cfg.random_seed
        )
        detector.fit(X_train)

        # Save model artifacts
        artifact_path = os.path.join(cfg.artifacts_dir, pipeline_name)
        detector.save(artifact_path)
        trained_detectors[pipeline_name] = detector

        # Extract features for evaluation (Normal Eval + ALL Abnormal)
        X_eval, y_eval, eval_paths = extract_features_from_files(df_eval, pipeline_name, cfg)
        preds, scores = detector.predict(X_eval)

        metrics = calculate_quantitative_metrics(y_eval, preds, scores, pipeline_name)
        summary_records.append(metrics)

        threshold_records.append({
            "pipeline": pipeline_name,
            "threshold": detector.threshold,
            "threshold_percentile": cfg.threshold_percentile,
            "input_dim": X_train.shape[1]
        })

        eval_data_dict[pipeline_name] = {
            "y_true": y_eval,
            "preds": preds,
            "scores": scores,
            "paths": eval_paths,
            "metrics": metrics,
            "detector": detector
        }

        if logger:
            logger.info(f"[{pipeline_name}] Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")

    # 3. Create Summary Tables & Save CSVs
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(cfg.results_dir, "summary_metrics.csv"), index=False)

    threshold_df = pd.DataFrame(threshold_records)
    threshold_df.to_csv(os.path.join(cfg.results_dir, "thresholds.csv"), index=False)

    # 4. Generate Publication Plots
    plot_roc_curves(eval_data_dict, os.path.join(cfg.results_dir, "roc_curves.png"))
    plot_pr_curves(eval_data_dict, os.path.join(cfg.results_dir, "precision_recall_curves.png"))
    plot_confusion_matrices(eval_data_dict, cfg.results_dir)
    plot_ablation_comparison(summary_df, os.path.join(cfg.results_dir, "ablation_comparison.png"))

    # 5. Evaluate Noise Robustness
    if progress_callback:
        progress_callback(0.9, "Evaluating Noise Robustness...")

    robustness_df = evaluate_noise_robustness(df_manifest, cfg, logger=logger)
    robustness_df.to_csv(os.path.join(cfg.results_dir, "noise_robustness.csv"), index=False)
    plot_noise_robustness(robustness_df, os.path.join(cfg.results_dir, "noise_robustness.png"))

    # 6. Error Analysis
    perform_error_analysis(eval_data_dict, os.path.join(cfg.results_dir, "error_analysis"))

    if progress_callback:
        progress_callback(1.0, "Execution Completed Successfully!")

    return trained_detectors, summary_df, robustness_df
