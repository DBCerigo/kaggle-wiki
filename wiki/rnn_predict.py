import os.path
import multiprocessing
from functools import partial
import numpy as np
import pandas as pd
import feather
import pickle
from . import val
from .utils import clock

def get_pages_df_template(data_dir):
    """Fetch from the data directory (or create and save if it doesn't exist) 
    the pages_df template, which has a row for every page with its name and a 
    unique id also used in the pred_df. To be populated with smape scores for 
    that page for a model.
    
    Args:
        data_dir -- string the directory path where it is or should be saved
    Returns:
        pages_df -- template described above
    """
    pages_fp = data_dir+'pages_df.f'
    if os.path.isfile(pages_fp):
        pages_df = pd.read_feather(pages_fp)
    else:
        print('No pages df template found. Creating and saving...')
        train_df = pd.read_csv(data_dir+'train_2.csv')
        pages_df = train_df.Page.to_frame().reset_index()
        pages_df.columns = ["id", "page"]
        pages_df.to_feather(pages_fp)
    return pages_df

def get_dates(data_dir):
    """Fetch from the data directory (or create and save if it doesn't exist) 
    a nparray with the dates of the timeseries, for use in making the prediction
    dataframes.
    Args:
        data_dir -- string the directory path where it is or should be saved
    Returns:
        dates -- np.array described above
    """
    dates_fp = data_dir+'dates.npy'
    if os.path.isfile(dates_fp):
        dates = pd.read_pickle(dates_fp)
    else:
        print('No dates df template found. Creating and saving...')
        train_df = pd.read_csv(data_dir+'train_2.csv')
        dates = train_df.drop('Page', axis=1).transpose().index.values
        np.save(dates_fp, dates)
    return dates

def combine_prediction_data(outputs, targets, sequences):
    """Combine prediction data into ground truth and full predicted sequence. 
    Use args outputted from model.predict()"""
    predictions_end = outputs.cpu().data.numpy().squeeze()
    truth_start = sequences.cpu().numpy().squeeze()
    try:
        truth_end = targets.cpu().numpy().squeeze()
    except:
        truth_end = targets.data.cpu().numpy().squeeze()

    truth = np.concatenate((truth_start, truth_end), axis=1) ; truth.shape
    predictions = np.concatenate((truth_start, predictions_end), axis=1)
    
    return truth, predictions

def create_pred_dfs(dates, outputs, targets, sequences, scaler):
    """Gives a list of dataframes, each with columns:
    - y and y_orig representing the original time series
    _ yhat represeting the predicted values from the model
    which can be used with the val.smape calculation functions and the 
    newphet.prophet_plot visualisation functions.
    Use this with the output of model.predict from an unshuffled dataset from
    the train_2.csv, so that the indices of the list correspond to the page_df
    ids to store smape values.
    
    Args:
        dates -- output of get_dates()
	    outputs -- first output of model.predict()
	    targets -- second output of model.predict()
	    sequences -- third output of model.predict()
	    scaler -- the sklearn scaler that originally scaled the data (returned 
	    from rnn_train.scale_values()
    Returns:
        populated list of pred_dfs described above
    """
    truth, predictions = combine_prediction_data(outputs, targets, sequences)
    truth = scaler.inverse_transform(truth.T).T
    predictions = scaler.inverse_transform(predictions.T).T

    p = multiprocessing.Pool(multiprocessing.cpu_count())
    dfs = p.map(partial(create_pred_df, dates=dates), zip(predictions,truth))
    return dfs

def create_pred_df(row_tuple, dates):
    """Create a prediction dataframe from a tuple of predictions. The dataframe
    has the required columns to be passed to the val.smape calculation functions
    and the newprophet.prophet_plot visualisation functions."""
    pred_r, truth_r = row_tuple
    df = pd.DataFrame()
    df['ds'] = pd.to_datetime(dates)
    df['y'] = df['y_org'] = truth_r
    df['yhat'] = pred_r
    df['train'] = [1]*490 + [0]*60
    return df

def create_pages_df(pages_df_template, pred_dfs, col_name):
    """Create the pages_df, which has a row for every page with its name and a 
    unique id also used in the pred_df, populated with smape scores for 
    that page for the model.
	
    Args:
        pages_df_template -- output of get_pages_df_template
        pred_dfs -- output of create_pred_df. Must be indexed 
        col_name -- the name of the column to store the smape values in the 
        resulting df
    Returns:
        populated pages_df described above
    """
    p = multiprocessing.Pool(multiprocessing.cpu_count())

    smape = p.map(calculate_smape_inner_df, pred_dfs)

    pages_df_template[col_name] = smape
    return pages_df_template

def calculate_smape_inner_df(df):
    """Calculate smape from an inner df of the pred_df"""
    df = df[df.train==0]
    return val.smape_df(df, 'y', 'yhat')
