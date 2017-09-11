import numpy as np

def smape(y_true, y_pred, axis=None):
    # NOTE: should check and make sure that NaNs aren't included
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff) if axis is None else np.nanmean(diff, axis=axis) 

def smape_df(df, y_true_label, y_pred_label):
    df = df.dropna(subset=[y_true_label])
    y_true = df[y_true_label]
    y_pred = df[y_pred_label]
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)

def get_smape_df(train, yhat):
    denominator = (train + yhat) / 200
    diff = np.abs(train - yhat) / denominator
    diff[denominator == 0] = 0.0
    #return np.nanmean(diff, axis=0)
    #diff.mean(axis=1, skipna=True)
    return diff
