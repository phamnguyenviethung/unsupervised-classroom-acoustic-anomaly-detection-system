#!/usr/bin/env python3
import os
import sys
import json
import argparse
import pandas as pd

from src.config import load_config
from src.logging_utils import setup_logger
from src.training import train_and_evaluate_all_pipelines

def main():
    parser = argparse.ArgumentParser(
        description="DSP501 Final Assignment CLI — Unsupervised Classroom Acoustic Anomaly Detection"
    )
    parser.add_argument("--data", type=str, default="data", help="Dataset root directory containing normal/ and abnormal/")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target audio sample rate in Hz (default: 16000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=40, help="Number of Autoencoder training epochs")
    parser.add_argument("--threshold-percentile", type=float, default=99.0, help="Percentile of normal train scores for threshold")
    parser.add_argument("--model-type", type=str, default="autoencoder", choices=["autoencoder", "isolation_forest"], help="Anomaly detection model type")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply CLI overrides
    config.data_root = args.data
    config.sample_rate = args.sample_rate
    config.random_seed = args.seed
    config.epochs = args.epochs
    config.threshold_percentile = args.threshold_percentile
    config.model_type = args.model_type

    logger = setup_logger("DSP501_CLI")
    logger.info("Starting DSP501 CLI Reproducible Research Execution...")

    # Save experiment config JSON
    os.makedirs(config.results_dir, exist_ok=True)
    with open(os.path.join(config.results_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    # Train and evaluate all 5 pipeline variants
    detectors, summary_df, robustness_df = train_and_evaluate_all_pipelines(config, logger=logger)

    # Display final Terminal Results Table
    print("\n" + "=" * 80)
    print("DSP501 EXPERIMENTAL RESULTS SUMMARY — FIVE PIPELINE VARIANTS")
    print("=" * 80)
    print(summary_df[["pipeline", "accuracy", "precision", "recall", "f1_score", "roc_auc", "false_alarm_rate", "detection_rate"]].to_string(index=False))
    print("=" * 80)
    print(f"\nAll artifacts saved to: '{config.artifacts_dir}/'")
    print(f"All CSV logs and publication figures saved to: '{config.results_dir}/'")
    print("CLI Execution Completed Successfully!\n")

if __name__ == "__main__":
    main()
