import os
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

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
    assert train.shape == yhat.shape
    denominator = (train + yhat) / 200
    diff = np.abs(train - yhat) / denominator
    diff[denominator == 0] = 0.0
    #return np.nanmean(diff, axis=0)
    #diff.mean(axis=1, skipna=True)
    diff = diff.round(decimals=8)
    assert diff.max().max() <= 200
    assert diff.min().min() >= 0
    return diff

def get_yhat_rolling_smape(train, yhat):
    assert train.shape == yhat.shape # think this is over kill
    yhat_smape = get_smape_df(train, yhat)
    yhat_rolling_smape = yhat_smape.rolling(60,
                    axis=1, min_periods=0).mean().iloc[:,60:].round(5)
    yhat_rolling_smape = yhat_rolling_smape.shift(-59, axis=1)
    assert yhat_rolling_smape.max().max() <= 200
    assert yhat_rolling_smape.min().min() >= 0
    print('prophet_rolling_smape indexing ::: index -> smape for that following (non_inclusive) 60 days period')
    print('(df.v7t_val.round(decimals=6).fillna(-10) == yhat_rolling_smape.iloc[:,-60].round(decimals=6).fillna(-10)).sum() -> 145063')
    return yhat_rolling_smape

def load_prophet_rolling_smape(VERSION, prop='yhat', force_remake=False, test=None, test_version=False):
    PROPHET_PATH = '../data/prophet/'
    CACHE_PATH = 'cache/'
    assert VERSION[-1] == '/'
    rolling_path = PROPHET_PATH+CACHE_PATH+VERSION[:-1]+'rolling_'+prop+'.f'
    if test_version:
        rolling_path = PROPHET_PATH+CACHE_PATH+VERSION[:-1]+'rolling_'+prop+'_test.f'
    if os.path.isfile(rolling_path) and not force_remake:
        print('prophet_rolling_smape indexing ::: index -> smape for that following (non_inclusive) 60 days period')
        print('(df.v7t_val.round(decimals=6).fillna(-10) == yhat_rolling_smape.iloc[:,-60].round(decimals=6).fillna(-10)).sum() -> 145063')
        return pd.read_feather(rolling_path)
    else:
        yhat_path = PROPHET_PATH+CACHE_PATH+VERSION[:-1]+prop+'.f'
        if os.path.isfile(yhat_path) and not force_remake:
            yhat = pd.read_feather(yhat_path)
        else:
            df = pd.read_feather('../data/train.f')
            forecast_files = [x.split('/')[-1] for x in glob.glob(PROPHET_PATH+VERSION+'*df.f')]
            init_forc = pd.read_feather(PROPHET_PATH+VERSION+forecast_files[0])
            ds_min = init_forc.ds.min().date()
            ds_max = init_forc.ds.max().date()
            df = df.loc[:,str(ds_min):str(ds_max)]
            df.loc[:] = np.nan
            try:
                    assert df.shape[1]-1 == (ds_max-ds_min).days
            except AssertionError:
                    assert df.shape[1] == 793
            for file_path in tqdm(forecast_files[:test]):
                forecast = pd.read_feather(PROPHET_PATH+VERSION+file_path)
                df.loc[int(file_path[:-4])] = forecast[prop].values
            df.sort_index(inplace=True)
            df = df.apply(pd.to_numeric)
            df.to_feather(yhat_path) 
            yhat = df
        train = pd.read_feather('../data/train.f')
        if test_version:
            train.iloc[:,-60:] = np.nan
            print('TEST VERSION so can"t see data train.iloc[:,-60:]')
        rolling_smape_df = get_yhat_rolling_smape(train, yhat)
        rolling_smape_df.to_feather(rolling_path)
        return rolling_smape_df
    

def load_median_rolling_smape():
    print('median_rolling_smape indexing ::: index -> smape for that following (non_inclusive) 60 days period')
    print('(df.smape_60_to_0.fillna(-1) == median_rolling_smape.iloc[:,-60].fillna(-1)).sum() -> 145063')
    return pd.read_feather('../data/median_rolling_smape.f')

def load_median_rolling(force_remake=False):
    df_path = '../data/median_rolling.f'
    if os.path.isfile(df_path) and not force_remake:
        return pd.read_feather(df_path)
    else:
        train = pd.read_feather('../data/train.f')
        med_rolling = train.rolling(49, axis=1, min_periods=0).median()
        med_rolling = med_rolling.round().fillna(0).astype(int)
        med_rolling.to_feather(df_path)
        return med_rolling


def load_test_median_rolling_smape(force_remake=False):
    df_path = '../data/median_test_rolling_smape.f'
    print('median_rolling_smape indexing ::: index -> smape for that following (non_inclusive) 60 days period')
    print('(df.smape_60_to_0.fillna(-1) == median_rolling_smape.iloc[:,-60].fillna(-1)).sum() -> 145063')
    if os.path.isfile(df_path) and not force_remake:
        return pd.read_feather(df_path)
    train = pd.read_feather('../data/train.f')
    train.iloc[:,-60:] = np.nan
    med_rolling = train.rolling(49, axis=1, min_periods=0).median()
    med_rolling = med_rolling.round().fillna(0).astype(int)
    median_rolling_smape = train.copy()
    median_rolling_smape.iloc[:,0] = np.nan
    for start_index in tqdm(range(1,train.shape[1])):
        median_rolling_smape.iloc[:,start_index] = _median_smape_periodStart(train,
                                                                            med_rolling.iloc[:,start_index-1],
                                                                            start_index)
    median_rolling_smape = median_rolling_smape.round(decimals=8)
    assert median_rolling_smape.max().max() <= 200
    assert median_rolling_smape.min().min() >= 0
    median_rolling_smape.to_feather(df_path)
    print('TEST VERSION so can"t see data train.iloc[:,-60:]')
    print('median_rolling_smape indexing ::: index -> smape for that following (non_inclusive) 60 days period')
    print('(df.smape_60_to_0.fillna(-1) == median_rolling_smape.iloc[:,-60].fillna(-1)).sum() -> 145063')
    return median_rolling_smape

def _median_smape_periodStart(train, preds, periodStart):
    denominator = train.iloc[:,periodStart:periodStart+60].add(preds, axis=0) / 200
    diff = np.abs(train.iloc[:,periodStart:periodStart+60].subtract(preds, axis=0)) / denominator
    diff[denominator == 0] = 0.0
    #return np.nanmean(diff, axis=0)
    return diff.mean(axis=1, skipna=True)
