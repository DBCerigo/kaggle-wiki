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

# ## Version 3
# Should set version directory name in next cell. Should describe version specifics (outliers, holidays, validation period)
#
#* TRAINING
#    * Val indexing on -60
#    * Cut outliers out on upper 95% quartile `forecast.loc[forecast['yhat'] < 0,['yhat']] = 0.0`
#    * Linear growth
#    * NO fillna(0) on everything - will use the NaNs as missing data
#    * Now with try:except: for the `RuntimeError': k initialized to invalid value (-nan)` which replaces first `y` with 0.001
#       * Now with try:except: for the `TypeError` which replaces first 10 `y` with 0 then first y with 0.001
#    
#* PREDICTIONS
#    * Truncating predictions at 0 
#    * Rounding to nearest int


# ### Remarks
# * ?

# Blend 1
# median and v3 only

PROPHET_PATH = '../data/prophet/'
BLENDS_PATH = 'blends/'
RESULTS_PATH = 'results/'

# should break if the dir already exists - avoids accidental overwriting
VERSION = 'v3f/'
assert VERSION[-1] == '/'
val_lims = (None,None)
os.makedirs(PROPHET_PATH+VERSION)

BLEND_NUMBER = 'b1'

pagedf = pd.read_feather(PROPHET_PATH+'pagedf.f')
ds = pd.read_feather(PROPHET_PATH+'ds.f')
# January, 1st, 2017 up until March 1st, 2017.
lg.info('Make pred_ds frame')
pred_ds = pd.DataFrame(pd.date_range('1/1/2017', '3/1/2017'), columns=['ds'])

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
        if page == '93470':
            df.loc[0,'y'] = 67
        # should also consider doing validation on the time period we are forecasting
        traindf = df.iloc[val_lims[0]:val_lims[1]]
        traindf.loc[traindf.y > traindf.y.quantile(.95), ['y']] = np.nan
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
        forecast = m.predict(pred_ds)
        forecast['yhat_org'] = forecast['yhat']
        forecast.loc[forecast['yhat'] < 0,['yhat']] = 0.0
        forecast.loc[:,'yhat'] = forecast.yhat.round(0).astype(int)
        forecast.to_feather(df_path)
        with open(model_path, 'wb') as file:
            pk.dump(m,file)
        lg.info(base_log_info+'COMPUTE and STORE FINISHED')

def wrapper(pages):
    base_log_info = '[Process:{0}] '.format(mp.current_process().name)
    lg.info(base_log_info +'starting its pages loop')
    for page in tqdm(pages):
        process_page(page)
    lg.info(base_log_info +'finished its pages loop')

# load the pages for this blend for this version
blend_pages = pd.read_feather(PROPHET_PATH+BLENDS_PATH+BLEND_NUMBER+VERSION[:-2]+'df.f')
pages = blend_pages.page_index.values.copy()
np.random.shuffle(pages)

total_proc = mp.cpu_count()
page_split = np.array_split(pages, total_proc)
mp_pool = mp.Pool(total_proc)
with utils.clock():
    mp_pool.map(wrapper, page_split)
    lg.info('Finished pool map')
lg.info('Script complete')
