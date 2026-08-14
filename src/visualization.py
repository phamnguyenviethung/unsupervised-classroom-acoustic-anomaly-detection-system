import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

def set_publication_style():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300
    })

def plot_roc_curves(eval_data_dict: dict, output_path: str):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, data in eval_data_dict.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["scores"])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {data['metrics']['roc_auc']:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier", alpha=0.6)
    ax.set_xlabel("False Positive Rate (False Alarm Rate)")
    ax.set_ylabel("True Positive Rate (Detection Rate)")
    ax.set_title("ROC Curves Across Five Pipeline Variants")
    ax.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_pr_curves(eval_data_dict: dict, output_path: str):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, data in eval_data_dict.items():
        prec, rec, _ = precision_recall_curve(data["y_true"], data["scores"])
        ax.plot(rec, prec, label=f"{name} (PR-AUC = {data['metrics']['pr_auc']:.3f})", linewidth=2)

    ax.set_xlabel("Recall (Detection Rate)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves Across Five Pipeline Variants")
    ax.legend(loc="lower left")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_confusion_matrices(eval_data_dict: dict, output_dir: str):
    set_publication_style()
    os.makedirs(output_dir, exist_ok=True)

    for name, data in eval_data_dict.items():
        cm = confusion_matrix(data["y_true"], data["preds"], labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Normal", "Abnormal"],
            yticklabels=["Normal", "Abnormal"],
            ax=ax
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix: {name}")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"confusion_matrix_{name}.png"), dpi=300)
        plt.close(fig)

def plot_ablation_comparison(summary_df: pd.DataFrame, output_path: str):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    metrics_to_plot = ["accuracy", "f1_score", "roc_auc"]
    df_plot = summary_df.melt(id_vars=["pipeline"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")

    sns.barplot(data=df_plot, x="pipeline", y="Score", hue="Metric", ax=ax, palette="deep")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Pipeline Variant")
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Metrics Comparison Across Pipeline Variants")
    plt.xticks(rotation=15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_noise_robustness(robustness_df: pd.DataFrame, output_path: str):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.lineplot(
        data=robustness_df,
        x="noise_std",
        y="f1_score",
        hue="pipeline",
        marker="o",
        linewidth=2.5,
        ax=ax
    )
    ax.set_xlabel("Additive Gaussian Noise Std Dev")
    ax.set_ylabel("F1 Score")
    ax.set_title("Noise Robustness: Performance Under Additive Audio Noise")
    ax.legend(title="Pipeline", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
