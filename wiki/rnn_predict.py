import os.path
import multiprocessing
import numpy as np
import pandas as pd
import feather
from . import val

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
		train_df = pd.read_csv(data_dir+'train_1.csv')
		pages_df = train_df.Page.to_frame().reset_index()
		pages_df.columns = ["id", "page"]
		pages_df.to_feather(pages_fp)
	return pages_df

def get_pred_df_template(data_dir):
    """Fetch from the data directory (or create and save if it doesn't exist) 
    the pred_df template, which is a multi-indexed array: 
    top level columns:
    - datestring
    - the 'train' column used in newphet.prophet_plot
    - a column for each page id
    second level columns under each page id, for use in newphet.prophet_plot:
    - y and y_orig representing the original time series
    _ yhat represeting the predicted values from the model

    To use with newphet.prophet_plot, call get_inner_pred_df with the id of the
    page you require.

    Args:
        data_dir -- string the directory path where it is or should be saved
    Returns:
        pred_df -- template described above
    """
    template_fp = data_dir+'pred_df_template.f'
    if os.path.isfile(template_fp):
        template = pd.read_feather(template_fp)
    else:
        train_df = pd.read_csv(data_dir+'train_1.csv')
        dates = train_df.drop('Page', axis=1).transpose().index.values
        #Multiple index columns - page IDs, then time series + predictions
        template = pd.DataFrame(
            columns=pd.MultiIndex.from_product(
                list(range(len(train_df))),
                ['y', 'y_org', 'yhat', 'train']
            )
        )
        template['ds', 'ds'] = dates
        template['train', 'train'] = [1]*490 + [0]*60
        template.to_feather(template_fp)
    return template

def combine_prediction_data(outputs, targets, sequences):
    """Combine prediction data into ground truth and full predicted sequence. 
    Use args outputted from model.predict()"""
    predictions_end = outputs.cpu().data.numpy().squeeze()
    truth_end = targets.cpu().numpy().squeeze()
    truth_start = sequences.cpu().numpy().squeeze()

    truth = np.concatenate((truth_start, truth_end), axis=1) ; truth.shape
    predictions = np.concatenate((truth_start, predictions_end), axis=1)
    
    return truth, predictions

def create_prediction_df(pred_df_template, outputs, targets, sequences, scaler):
    """Create the prediction df which is a multi-indexed array: 
    top level columns:
    - datestring
    - the 'train' column used in newphet.prophet_plot
    - a column for each page id
    second level columns under each page id, for use in newphet.prophet_plot:
    - y and y_orig representing the original time series
    _ yhat represeting the predicted values from the model
    
    To use with newphet.prophet_plot, call get_inner_pred_df with the id of the
    page you require.

    Args:
        pred_df_template -- output of get_pred_df_template
	    outputs -- first output of model.predict()
	    targets -- second output of model.predict()
	    sequences -- third output of model.predict()
	    scaler -- the sklearn scaler that originally scaled the data (returned 
	    from rnn_train.scale_values()
    Returns:
		populated pred_df described above
    """
    truth, predictions = combine_prediction_data(outputs, targets, sequences)
    truth = scaler.inverse_transform(truth.T).T
    predictions = scaler.inverse_transform(predictions.T).T

    pred_df = pred_df_template.copy(True)
    for i, (p, t) in enumerate(zip(predictions, truth)):
        pred_df[i, 'y'] = pred_df[i, 'y_org'] = t
        pred_df[i, 'yhat'] = p

    return pred_df

def get_inner_pred_df(pred_df, page_id):
    """Get an inner prediction dataframe, with columns for the date, the
    original time series, the predictions and the dates trained on from the 
    pred_df.

    Args:
        pred_df -- output of create_pred_df
        page_id -- the page id (as per the pages_df template) of the page who's
        series you require 
    Returns:
        df described above
    """
    inner = df[[page_id,'ds','train']]
    inner.columns = inner.columns.get_level_values(1)
    return inner

def create_pages_df(pages_df_template, pred_df, col_name):
	"""Create the pages_df, which has a row for every page with its name and a 
	unique id also used in the pred_df, populated with smape scores for 
	that page for the model.
	
	Args:
        pages_df_template -- output of get_pages_df_template
        pred_df -- output of create_pred_df
        col_name -- the name of the column to store the smape values in the 
        resulting df
	Returns:
		populated pages_df described above
	"""
	p = multiprocessing.Pool(multiprocessing.cpu_count())

	smape = p.map(calculate_smape_inner_df, inner_array_generator(pred_df))

	pages_df_template[col_name] = smape
	return pages_df_template

def inner_array_generator(pred_df):
    """Generator returning all inner dfs of the pred_df"""
    #Cut off the last two level 0 column headers (which are 'ds' and 'train') to
    #leave just the page indices
    for index in df.columns.levels[0].values[-2]:
        yield get_inner_pred_df(pred_df, index)

def calculate_smape_inner_df(df):
    """Calculate smape from an inner df of the pred_df"""
    df = df[df.train==0]
    return val.smape_df(df, 'y', 'yhat')
