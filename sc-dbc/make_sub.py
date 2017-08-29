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

lg.info('Load base dfs')
page_index = pd.read_feather(PROPHET_PATH+'page_index.f')
keydf = pd.read_csv('../data/key_1.csv')

# January, 1st, 2017 up until March 1st, 2017.
lg.info('Make ds frame')
ds = pd.DataFrame(pd.date_range('1/1/2017', '3/1/2017'), columns=['ds'])

#testing
#page_index = page_index.loc[:10]

subf = open(PROPHET_PATH+'submissions/'+VERSION[:-1]+'.csv', 'w')
subf.write('Id,Visits\n')
lg.info('Start load predict and write loop')
for row in tqdm(page_index.iterrows()):
    # use row[1][0] to load model
    lg.info('Start load model')
    with open(PROPHET_PATH+VERSION+str(row[1][0])+'m.pk', 'rb') as pkf:
        m = pk.load(pkf)
        pkf.close()
    lg.info('Finished load model')
    # use model + ds to get prediction
    lg.info('Start model predict')
    df = m.predict(ds).loc[:,['ds','yhat']]
    lg.info('Finish model predict')
    lg.info('Start truncate to zero')
    df.loc[df['yhat'] < 0,['yhat']] = 0.0
    lg.info('Finish truncate to zero')
    # use row[1][1] and apply to make key col
    lg.info('Start apply make page_date col')
    df['Page'] = df.ds.apply(lambda x: row[1][1]+'_'+str(x.date()))
    lg.info('Finish apply make page_date col')
    lg.info('Start merg on Page')
    # NOTE: WARN: this takes ~5 seconds and is the chock
    # Bad because we are every time searching for out page in the massive key list
    # Solutions: filter on the Page name first? Make key df for each page first and then just read that little df and merge
    # Go through the page numbers in order and only read the corresponding part of the csv
    df = df.merge(keydf, on='Page', how='left').loc[:,['Id','yhat']]
    lg.info('Finish merg on Page')
    lg.info('Start to_csv')
    df.to_csv(subf,header=False, index=False)
    lg.info('Finished to_csv')
    # write it to csv
subf.close()

