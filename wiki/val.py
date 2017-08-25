import numpy as np

def smape(y_true, y_pred):
    denominator = (y_true + y_pred) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)

def smape_df(df, y_true_label, y_pred_label):
    df = df.dropna(subset=[y_true_label])
    y_true = df[y_true_label]
    y_pred = df[y_pred_label]
    denominator = (y_true + y_pred) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)
