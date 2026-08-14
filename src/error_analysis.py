import os
import pandas as pd
import numpy as np

def perform_error_analysis(
    eval_data_dict: dict,
    output_dir: str
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    error_summary = {}

    for name, data in eval_data_dict.items():
        y_true = data["y_true"]
        preds = data["preds"]
        scores = data["scores"]
        paths = data["paths"]

        records = []
        for i in range(len(y_true)):
            records.append({
                "pipeline": name,
                "file_path": paths[i],
                "filename": os.path.basename(paths[i]),
                "true_label": int(y_true[i]),
                "predicted_label": int(preds[i]),
                "anomaly_score": float(scores[i]),
                "threshold": float(data["detector"].threshold),
                "is_correct": int(y_true[i] == preds[i]),
                "error_type": "FP" if (y_true[i] == 0 and preds[i] == 1) else ("FN" if (y_true[i] == 1 and preds[i] == 0) else "Correct")
            })

        df_err = pd.DataFrame(records)
        df_err.to_csv(os.path.join(output_dir, f"error_records_{name}.csv"), index=False)

        fp_count = len(df_err[df_err["error_type"] == "FP"])
        fn_count = len(df_err[df_err["error_type"] == "FN"])

        df_err[df_err["error_type"] == "FP"].to_csv(os.path.join(output_dir, f"false_positives_{name}.csv"), index=False)
        df_err[df_err["error_type"] == "FN"].to_csv(os.path.join(output_dir, f"false_negatives_{name}.csv"), index=False)

        error_summary[name] = {
            "total_eval_samples": len(y_true),
            "false_positives": fp_count,
            "false_negatives": fn_count,
            "accuracy": float(np.mean(y_true == preds))
        }

    df_err_summary = pd.DataFrame.from_dict(error_summary, orient="index")
    df_err_summary.to_csv(os.path.join(output_dir, "error_summary.csv"))

    return error_summary
