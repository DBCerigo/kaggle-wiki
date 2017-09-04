# coding: utf-8
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
total_proc = None
from tqdm import tqdm
from IPython.display import clear_output


PROPHET_PATH = '../data/prophet/'
RESULTS_PATH = 'results/'
VERSION = 'v1.6f/'
assert VERSION[-1] == '/'
assert VERSION[-2] == 'f'

lg.info('Start load page_index df')
page_index = pd.read_feather(PROPHET_PATH+'page_index.f')

# January, 1st, 2017 up until March 1st, 2017.
lg.info('Make ds frame')
ds = pd.DataFrame(pd.date_range('1/1/2017', '3/1/2017'), columns=['ds'])

#testing
#page_index = page_index.loc[:100]



def process_page(index_page):
    base_log_info = '[Process:{0}, on page:{1}] '.format(mp.current_process().name, index_page[0])
    lg.info(base_log_info)
    # use index_page[0] to load model
    lg.info(base_log_info+'Start load model')
    with open(PROPHET_PATH+VERSION+str(index_page[0])+'m.pk', 'rb') as pkf:
        m = pk.load(pkf)
        pkf.close()
    lg.info(base_log_info+'Finished load model')
    # use model + ds to get prediction
    # NOTE: this is the limiting step ~ 1s -> should make parallel 
    lg.info(base_log_info+'Start model predict')
    tdf = m.predict(ds).loc[:,['ds','yhat']]
    lg.info(base_log_info+'Finish model predict')
    lg.info(base_log_info+'Start truncate to zero')
    tdf.loc[tdf['yhat'] < 0,['yhat']] = 0.0
    lg.info(base_log_info+'Finish truncate to zero')
    # use index_page[1] and apply to make key col
    lg.info(base_log_info+'Start apply make page_date col')
    tdf['Page'] = tdf.ds.apply(lambda x: index_page[1]+'_'+str(x.date()))
    lg.info(base_log_info+'Finish apply make page_date col')
    lg.info(base_log_info+'Start del ds column')
    del tdf['ds']
    lg.info(base_log_info+'Finish del ds column')
    return tdf

def wrapper(pageindexs):
    base_log_info = '[Process:{0}] '.format(mp.current_process().name)
    lg.info(base_log_info +'starting its pages loop')
    for index_page in tqdm(pageindexs):
        tdf = process_page(index_page)
        try:
            lg.info(base_log_info+'Start append')
            df = df.append(tdf, ignore_index=True)
            lg.info(base_log_info+'Finish append')
        except NameError:
            df = tdf
    lg.info(base_log_info +'finished its pages loop')
    return df

# parallel processing loop
total_proc = mp.cpu_count()
# NOTE: bad name, actually [[index1,page1],[index2,page2],...]
pageindexs = page_index.values.copy()
np.random.shuffle(pageindexs)
pageindexs_split = np.array_split(pageindexs, total_proc)
lg.info(pageindexs_split[0])
mp_pool = mp.Pool(total_proc)
with utils.clock():
    dfs = mp_pool.map(wrapper, pageindexs_split)
    lg.info('Finished pool map')
lg.info('Val results recieved - processes ended')
df = pd.concat(dfs, ignore_index=True)
lg.info(df.head())

# writing
subf = open(PROPHET_PATH+'submissions/'+VERSION[:-1]+'.csv', 'w')
subf.write('Id,Visits\n')
lg.info('Start load key df')
keydf = pd.read_csv('../data/key_1.csv')
lg.info('Finish load key df')
lg.info('Start merg on Page')
df = df.merge(keydf, on='Page', how='left').loc[:,['Id','yhat']]
lg.info('Finish merg on Page')
lg.info('Start to_csv')
df.to_csv(subf,header=False, index=False)
subf.close()
lg.info('Finished to_csv')

