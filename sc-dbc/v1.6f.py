# coding: utf-8


# NOTE: should be ran with `python3 vX.py 2>> some_log_file 1> /dev/null`, this capture the good stuff in the log
# ...and dumps the shitty STAN stuff 

# In[1]:


# In[2]:


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


# In[3]:


PROPHET_PATH = '../data/prophet/'
RESULTS_PATH = 'results/'


# In[4]:


pagedf = pd.read_feather(PROPHET_PATH+'pagedf.f')
ds = pd.read_feather(PROPHET_PATH+'ds.f')


# ## Version 1.8
# Should set version directory name in next cell. Should describe version specifics (outliers, holidays, validation period)

# * No val - for submission!
# * No outlier fixing
# * Linear growth
# * Truncating predictions at 0
# * Fill first 0s with 0.0001
# * Fill first NaNs with 0.0001
# * Fill ALL other NaNs to 0

# ### Remarks
# * ?



# should break if the dir already exists - avoids accidental overwriting
VERSION = 'v1.6f/'
assert VERSION[-1] == '/'
val_lims = (None,None)
#os.makedirs(PROPHET_PATH+VERSION)


# In[6]:

pagedf.loc[:0,pagedf.loc[0]==0] = 0.001
pagedf.loc[:0] = pagedf.loc[:0].fillna(0.001)
pagedf = pagedf.fillna(0); pagedf.head()


# # WARNING:
# Turned off the chained assignment warning - when slicing dfs they can return copies sometimes instead,
# which will mean your assignment wont be done on the actual base df.
# Not sure why it's still compaining at me when I'm using .loc for assignations everywhere... shitty

# In[8]:


pd.options.mode.chained_assignment = None


# In[9]:


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
        # should also consider doing validation on the time period we are forecasting
        traindf = df.iloc[val_lims[0]:val_lims[1]]
        traindf['train'] = 1 # feather won't serialize bool so 1s and 0s...
        # do outlier removal here NOTE: WARN: this could remove the 0.001 and then cause prophet to fail fml
        #traindf.loc[traindf.y > traindf.y.quantile(.95), ['y']] = None
        m = Prophet(yearly_seasonality=True)
        m.fit(traindf)
        forecast = m.predict(ds)
        forecast['yhat_org'] = forecast['yhat']
        forecast.loc[forecast['yhat'] < 0,['yhat']] = 0.0
        forecast = forecast.join(df.y)
        forecast = forecast.join(traindf.loc[:,['train']]).fillna({'train':0}) # 0 bools
        forecast.to_feather(df_path)
        with open(model_path, 'wb') as file:
            pk.dump(m,file)
        lg.info(base_log_info+'COMPUTE and STORE FINISHED')
    full_smape = wiki.val.smape(forecast.y, forecast.yhat)
    val_smape = wiki.val.smape(forecast[forecast['train'] == 0].y,forecast[forecast['train'] == 0].yhat)
    lg.info(base_log_info +'smape calc finished')
    return (page, full_smape, val_smape)

def wrapper(pages):
    val_results = []
    for page in tqdm(pages):
        val_results.append(process_page(page))
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
val_results = [item for sublist in val_results for item in sublist]
val_results = pd.DataFrame(val_results, columns=['page_index',VERSION[:-1]+'_full',VERSION[:-1]+'_val'])
val_results.to_feather(PROPHET_PATH+RESULTS_PATH+VERSION[:-1]+'df.f')
lg.info('Script complete')
