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


PROPHET_PATH = '../data/prophet/'
BLENDS_PATH = 'blends/'
RESULTS_PATH = 'results/'
BLEND_NUMBER = 'b1/'
assert BLEND_NUMBER[-1] == '/'
VERSION = 'v3f/'
assert VERSION[-1] == '/'
assert VERSION[-2] == 'f'


def process_page(page_num):
    base_log_info = '[Process:{0}, on page:{1}] '.format(mp.current_process().name, page_num)
    lg.info(base_log_info)
    # use index_page[0] to load model
    lg.info(base_log_info+'Start load forecast')
    tdf = pd.read_feather(PROPHET_PATH+VERSION+page_num+'df.f').loc[:,['ds','yhat']]
    lg.info(base_log_info+'Finished load forecast')
    # use page_index.loc[page_num] and apply to make key col
    lg.info(base_log_info+'Start apply make page_date col')
    tdf['Page'] = tdf.ds.apply(lambda x: page_index.loc[int(page_num)].Page+'_'+str(x.date()))
    lg.info(base_log_info+'Finish apply make page_date col')
    lg.info(base_log_info+'Start del ds column')
    del tdf['ds']
    lg.info(base_log_info+'Finish del ds column')
    return tdf

def wrapper(blend_pages):
    base_log_info = '[Process:{0}] '.format(mp.current_process().name)
    lg.info(base_log_info +'starting its pages loop')
    for page_num in tqdm(blend_pages):
        tdf = process_page(page_num)
        try:
            lg.info(base_log_info+'Start append')
            df = df.append(tdf, ignore_index=True)
            lg.info(base_log_info+'Finish append')
        except NameError:
            df = tdf
    lg.info(base_log_info +'finished its pages loop')
    return df


lg.info('Start load page_index df')
page_index = pd.read_feather(PROPHET_PATH+'page_index.f')
page_index = page_index.set_index('page_index')

lg.info('Start load blend version df')
blend_pages = pd.read_feather(PROPHET_PATH+BLENDS_PATH+BLEND_NUMBER
                              +VERSION[:-2]+'df.f')
blend_pages = blend_pages.values.flatten()
# testing
#blend_pages = blend_pages[:100]
np.random.shuffle(blend_pages)
# parallel processing loop
total_proc = 1# mp.cpu_count()
blend_pages_split = np.array_split(blend_pages, total_proc)
mp_pool = mp.Pool(total_proc)
with utils.clock():
    dfs = mp_pool.map(wrapper, blend_pages_split)
    lg.info('Finished pool map')
lg.info('Val results recieved - processes ended')
df = pd.concat(dfs, ignore_index=True)
lg.info(df.head())

# writing
subf = open(PROPHET_PATH+'submissions/'+BLEND_NUMBER[:-1]+VERSION[:-1]+'.csv', 'w')
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

