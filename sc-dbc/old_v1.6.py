
# coding: utf-8

# In[1]:


# In[2]:


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


# ## Version 1.6
# Should set version directory name in next cell. Should describe version specifics (outliers, holidays, validation period)

# * Val indexing on -60
# * No outlier fixing
# * Linear growth
# * Truncating predictions at 0
# * Fill first NaNs with 0.0001
# * Fill ALL other NaNs to 0

# ### Remarks
# * ?

# In[5]:


# should break if the dir already exists - avoids accidental overwriting
VERSION = 'v1.6/'
val_lims = (0,-60)
#os.makedirs(PROPHET_PATH+VERSION)


# In[6]:


pagedf.loc[:0] = pagedf.loc[:0].fillna(0.001)


# In[7]:


pagedf = pagedf.fillna(0); pagedf.head()


# # WARNING:
# Turned off the chained assignment warning - when slicing dfs they can return copies sometimes instead,
# which will mean your assignment wont be done on the actual base df.
# Not sure why it's still compaining at me when I'm using .loc for assignations everywhere... shitty

# In[8]:


pd.options.mode.chained_assignment = None


# In[9]:


def process_page(page):
    df = ds.join(pagedf[page])
    df.columns = ['ds','y']
    # note this is doing validation on last 60 days
    # should also consider doing validation on the time period we are forcasting
    traindf = df.iloc[val_lims[0]:val_lims[1]]
    traindf['train'] = 1 # feather won't serialize bool so 1s and 0s...
    # do outlier removal here
    #traindf.loc[traindf.y > traindf.y.quantile(.95), ['y']] = None
    m = Prophet(yearly_seasonality=True)
    m.fit(traindf)
    forecast = m.predict(ds)
    forecast['yhat_org'] = forecast['yhat']
    forecast.loc[forecast['yhat'] < 0,['yhat']] = 0.0
    forecast = forecast.join(df.y)
    forecast = forecast.join(traindf.loc[:,['train']]).fillna({'train':0}) # 0 bools
    forecast.to_feather(PROPHET_PATH+VERSION+page+'df.f')
    with open(PROPHET_PATH+VERSION+page+'m.pk', 'wb') as file:
        pk.dump(m,file)
    full_smape = wiki.val.smape(forecast.y, forecast.yhat)
    val_smape = wiki.val.smape(forecast[forecast['train'] == 0].y,forecast[forecast['train'] == 0].yhat)
    return (page, full_smape, val_smape)


# In[10]:


def wrapper(pages):
    val_results = []
    for page in tqdm(pages):
        val_results.append(process_page(page))
    return val_results




# In[12]:


total_proc = mp.cpu_count()


# In[13]:


col_split = np.array_split(pagedf.columns, total_proc)
mp_pool = mp.Pool(total_proc)


# In[14]:


with utils.clock():
    val_results = mp_pool.map(wrapper, col_split)


# In[15]:


val_results = [item for sublist in val_results for item in sublist]
val_results = pd.DataFrame(val_results, columns=['page_index',VERSION[:-1]+'_full',VERSION[:-1]+'_val'])


# In[16]:


val_results.to_feather(PROPHET_PATH+RESULTS_PATH+VERSION[:-1]+'df.f')

