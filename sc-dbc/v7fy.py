# coding: utf-8


# NOTE: should be ran with `python3 vX.py 2>> some_log_file 1> /dev/null`, this capture the good stuff in the log
# ...and dumps the shitty STAN stuff 


import logging as lg
lg.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=lg.INFO)
import os
import pandas as pd
import numpy as np 
import pickle as pk
import glob
from fbprophet import Prophet
import sys
sys.path.append('../')
import wiki
from wiki import utils 
import multiprocessing as mp
from tqdm import tqdm


# ## Version 7 FINAL year before
# Should set version directory name in next cell. Should describe version specifics (outliers, holidays, validation period)

#* TRAINING
#    * Train indexing on None
#    * Val indexing on None to None (end/None)
#    * Now with try:except:except: for the `RuntimeError': k initialized to invalid value (-nan)` which replaces first `y` with 0.001...
#       * ...and for the `TypeError` which replaces first 10 `y` with 0 then first y with 0.001
#    
#* PREDICTIONS
#    * Truncating predictions at 0 
#    * Rounding to nearest int

# ### Remarks
# * ?

PROPHET_PATH = '../data/prophet/'
RESULTS_PATH = 'results/'

lg.info('Loading base pagedf and ds')
pagedf = pd.read_feather(PROPHET_PATH+'pagedf.f')
ds = pd.read_feather(PROPHET_PATH+'ds.f')
lg.info('Finished loading base pagedf and ds')

# should break if the dir already exists - avoids accidental overwriting
VERSION = 'v7fy/'
assert VERSION[-1] == '/'
upper_lim = None # data beyond here considered future
val_lims = utils.prevYear_shift((74-62,74))
os.makedirs(PROPHET_PATH+VERSION)

# # WARNING:
# Turned off the chained assignment warning - when slicing dfs they can return copies sometimes instead,
# which will mean your assignment wont be done on the actual base df.
# Not sure why it's still compaining at me when I'm using .loc for assignations everywhere... shitty
pd.options.mode.chained_assignment = None

def process_page(page):
    base_log_info = '[Process:{0}, on page:{1}] '.format(mp.current_process().name, page)
    lg.info(base_log_info)
    df_path = PROPHET_PATH+VERSION+page+'df.f'
    model_path = PROPHET_PATH+VERSION+page+'m.pk'
    if os.path.isfile(df_path) and os.path.isfile(model_path):
        lg.info(base_log_info +'df and m EXIST loop')
        forecast = pd.read_feather(df_path)
    else:
        lg.info(base_log_info +'COMPUTE loop')
        df = ds.join(pagedf[page])
        df.columns = ['ds','y']
        df['y_org'] = df.y
        # doing validation on year previous time period, so set to NaN
        df.y.iloc[val_lims[0]:val_lims[1]] = np.nan
        df['train'] = 1 # feather won't serialize bool so 1s and 0s...
        df.train.iloc[val_lims[0]:val_lims[1]] = 0 # set labels to test
        # check
        assert df.iloc[val_lims[0]:val_lims[1]].y.count() == 0
        assert df.iloc[val_lims[0]:val_lims[1]].train.sum() == 0
        traindf = df.iloc[:upper_lim]
        try:
            m = Prophet(yearly_seasonality=True)
            m.fit(traindf)
        except RuntimeError:
            lg.info(base_log_info+'RuntimeError triggered on fit (all 0), replacing first y with 0.001 and retry')
            traindf.loc[0,'y'] = 0.001
            m = Prophet(yearly_seasonality=True)
            m.fit(traindf)
        except TypeError:
            lg.info(base_log_info+'TypeError triggered on fit (all NaN), replacing first 10 y with 0 and first y with 0.001 and retry')
            traindf.loc[:10,'y'] = 0
            traindf.loc[0,'y'] = 0.001
            m = Prophet(yearly_seasonality=True)
            m.fit(traindf)
        forecast = m.predict(ds.iloc[:upper_lim])
        forecast['yhat_org'] = forecast['yhat']
        forecast.loc[forecast['yhat'] < 0,['yhat']] = 0.0
        forecast.loc[:,'yhat'] = forecast.yhat.round(0).astype(int)
        df = df.iloc[:upper_lim]
        forecast = forecast.join(df.y)
        forecast = forecast.join(df.y_org)
        forecast = forecast.join(df.train)
        forecast.to_feather(df_path)
        with open(model_path, 'wb') as file:
            pk.dump(m,file)
        lg.info(base_log_info+'COMPUTE and STORE FINISHED')
    train_smape = wiki.val.smape(forecast[forecast['train'] == 1].y_org, forecast[forecast['train'] == 1].yhat)
    val_smape = wiki.val.smape(forecast[forecast['train'] == 0].y_org,forecast[forecast['train'] == 0].yhat)
    lg.info(base_log_info +'smape calc finished')
    return (page, train_smape, val_smape)

def wrapper(pages):
    base_log_info = '[Process:{0}] '.format(mp.current_process().name)
    val_results = []
    lg.info(base_log_info +'starting its pages loop')
    for page in tqdm(pages):
        val_results.append(process_page(page))
    lg.info(base_log_info +'finished its pages loop')
    return val_results

# testing

total_proc = mp.cpu_count()
# NOTE: shuffle the cols to that any pages that still need models built get distributied evenly
# NOTE: shuffling the index directly switches all the pages from their corresponding series... BAD
cols = pagedf.columns.values.copy()
np.random.shuffle(cols)
col_split = np.array_split(cols, total_proc)
mp_pool = mp.Pool(total_proc)
with utils.clock():
    val_results = mp_pool.map(wrapper, col_split)
    lg.info('Finished pool map')
lg.info('Val results recieved - processes ended')
val_results = [item for sublist in val_results for item in sublist]
val_results = pd.DataFrame(val_results, columns=['page_index',VERSION[:-1]+'_train',VERSION[:-1]+'_val'])
val_results.to_feather(PROPHET_PATH+RESULTS_PATH+VERSION[:-1]+'df.f')
lg.info('Script complete')
