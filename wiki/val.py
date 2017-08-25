import numpy as np

def smape(y_true, y_pred):
    denominator = (y_true + y_pred) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)
