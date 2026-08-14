import numpy as np
from src.evaluation import calculate_quantitative_metrics

def test_metrics_calculation():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    scores = np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.85, 0.95, 0.3])

    m = calculate_quantitative_metrics(y_true, y_pred, scores, "test_pipe")

    assert m["accuracy"] == 0.75
    assert m["tp"] == 3
    assert m["tn"] == 3
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["false_alarm_rate"] == 0.25
    assert m["detection_rate"] == 0.75

if __name__ == "__main__":
    test_metrics_calculation()
    print("Metrics unit tests passed!")
