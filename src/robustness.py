import pandas as pd
from src.config import AppConfig
from src.dataset import split_dataset_unsupervised
from src.pipelines import PIPELINE_VARIANTS, extract_features_from_files
from src.models import UnsupervisedAnomalyDetector
from src.evaluation import calculate_quantitative_metrics

def evaluate_noise_robustness(
    df_manifest: pd.DataFrame,
    cfg: AppConfig,
    logger=None
) -> pd.DataFrame:
    df_train_norm, df_eval = split_dataset_unsupervised(
        df_manifest, train_ratio=cfg.train_ratio, seed=cfg.random_seed
    )

    records = []

    for pipeline_name in PIPELINE_VARIANTS:
        # Extract features for training (clean normal data)
        X_train_clean, _, _ = extract_features_from_files(df_train_norm, pipeline_name, cfg, additive_noise_std=0.0)

        # Train model
        detector = UnsupervisedAnomalyDetector(
            model_type=cfg.model_type,
            input_dim=X_train_clean.shape[1],
            hidden_dims=cfg.hidden_dims,
            learning_rate=cfg.learning_rate,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            threshold_percentile=cfg.threshold_percentile,
            random_seed=cfg.random_seed
        )
        detector.fit(X_train_clean)

        for noise_std in cfg.noise_levels:
            # Extract evaluation features with noise
            X_eval_noisy, y_eval, _ = extract_features_from_files(df_eval, pipeline_name, cfg, additive_noise_std=noise_std)
            preds, scores = detector.predict(X_eval_noisy)

            metrics = calculate_quantitative_metrics(y_eval, preds, scores, pipeline_name)
            metrics["noise_std"] = noise_std
            records.append(metrics)

            if logger:
                logger.info(f"[Robustness] Pipeline: {pipeline_name} | Noise: {noise_std} | F1: {metrics['f1_score']:.4f}")

    res_df = pd.DataFrame(records)
    return res_df
