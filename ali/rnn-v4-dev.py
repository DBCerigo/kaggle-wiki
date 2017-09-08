
# coding: utf-8

# # v4
# we basically need a different batch generator. Will first look at what Hooker's got and see if it's worth stealing - an extra thing to think about is to think properly how to deal with the time-dependent and -independent covariates and how best to provide them (particularly for the prediction stage when the dates are 'in the future')

# In[44]:


import sys
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.autograd import Variable

from tqdm import tqdm

sys.path.append('../')
from wiki.utils import clock
from wiki import rnn, rnn_predict, newphet, val, submissions, rnn_windowed


# In[19]:


base_dir = '../data/'
pred_len = 62
batch_size = 1024


# In[9]:


train_df = pd.read_csv(base_dir+'train_2.csv').fillna(0)


# In[45]:


page_groups = rnn_windowed.get_page_groups(train_df)


# In[46]:


np.array(page_groups).shape


# In[47]:


values = train_df.drop('Page', axis=1).values ; values.shape


# In[48]:


dates = train_df.columns[1:].values
s_date = dates[0]
e_date = dates[-1]


# In[49]:


dates = pd.date_range(s_date, e_date)


# In[50]:


ages = np.arange(len(dates))
dows = dates.dayofweek.values
woys = dates.weekofyear.values


# In[51]:


#Expand the dims to make broadcasting work - since numpy
#refuses to add dimensions to the right when broadcasting
series_idxs = np.arange(values.shape[0])
#series_idxs = series_idxs.reshape((series_idxs.shape+(1,1)))


# In[52]:


series_idxs.shape


# In[15]:


timedep = np.stack([ages, dows, woys], axis=-1)


# In[16]:


seriesdep = np.array(page_groups)


# In[17]:


values, scaler = rnn.scale_values(values)


# Ok, the `DataLoaders` aren't gonna work anymore since it makes everything inside a Variable which require gradients. Our embedding indices compute gradient wrt to the embeddings, not the indices, so it breaks. That means we've gotta split it up - so we might as well just do it ourselves.

# In[18]:


class test_datagen(object):
    def __init__(self, *args):
        self.args = args
        self.generator = self.gen(*args)
        
    def __iter__(self):
        return self.gen(*self.args)
    
    def gen(self, timeseries, timedep, seriesdep, predlen, batch_size):
        """"timeseries: (total, series_length, 1)
        timedep: (series_length, num_feats)
        seriesdep: (total)
        return:
               train_series: (batch_size, window_size, 1), 
            train_timedep: (batch_size, window_size, num_feats),
            train_seriesdep: (batch_size, 1)
            target_series: (batch_size, window_size, 1)
            target_timedep: (batch_size, window_size, num_feats),
            target_seriesdep: (batch_size, 1)
        """
        train_series = timeseries[:,:-predlen,:]
        target_series = timeseries[:,-predlen:,:]
        train_timedep = timedep[:-predlen,:]
        target_timedep = timedep[-predlen:,:]
        seriesdep = np.expand_dims(seriesdep, axis=-1)
        i=0
        while i<train_series.shape[0]:
            if i+batch_size > train_series.shape[0]:
                batch_size = train_series.shape[0] - i

            conv = lambda x: torch.from_numpy(x)
            yield (
                conv(train_series[i:i+batch_size,:,:]).float(),
                conv(np.broadcast_to(train_timedep, (batch_size,)+train_timedep.shape)).float(),
                conv(seriesdep[i:i+batch_size,:]).long(),
                conv(target_series[i:i+batch_size,:,:]).float(),
                conv(np.broadcast_to(target_timedep, (batch_size,)+target_timedep.shape)).float(),
                conv(seriesdep[i:i+batch_size,:]).long()
            )
            i += batch_size


# In[20]:


#THROWS AWAY LAST EXAMPLES!!! it will break if number of examples is divisible by batch size
class train_datagen(object):
    def __init__(self, *args):
        self.args = args
        self.generator = self.gen(*args)
        
    def __iter__(self):
        return self.gen(*self.args)
    
    def gen(self, timeseries, timedep, seriesdep, window_size, window_space, num_per_series, batch_size):
        """"timeseries: (total, series_length, 1)
        timedep: (series_length, num_feats)
        seriesdep: (total)
        return: (broadcasted if necessary)
            train_series: (batch_size, window_size, 1), 
            train_timedep: (batch_size, window_size, num_feats),
            train_seriesdep: (batch_size, 1)
            target_series: (batch_size, window_size, 1)
            target_timedep: (batch_size, window_size, num_feats),
            target_seriesdep: (batch_size, 1)
        """
        train_series, target_series, train_seriesdep = [],[],[]
        train_timedep, target_timedep = [],[]
        for series, seriesdep in zip(timeseries, seriesdep):
            #for k in range(num_per_series):
            for k in range(num_per_series):
                train_series.append(series[-2*window_size-k*window_space:-window_size-k*window_space, :])
                if k != 0:
                    target_series.append(series[-window_size-k*window_space:-k*window_space, :])
                    target_timedep.append(timedep[-window_size-k*window_space:-k*window_space, :])
                else:
                    target_series.append(series[-window_size-k*window_space:, :])
                    target_timedep.append(timedep[-window_size-k*window_space:, :])
                train_timedep.append(timedep[-2*window_size-k*window_space:-window_size-k*window_space, :])
                train_seriesdep.append(np.expand_dims(seriesdep, axis=-1))
                if len(train_series) == batch_size:
                    conv = lambda x: torch.from_numpy(np.stack(x))
                    yield (
                        conv(train_series).float(),
                        conv(train_timedep).float(),
                        conv(train_seriesdep).long(),
                        conv(target_series).float(),
                        conv(target_timedep).float(),
                        conv(train_seriesdep).long()
                    )
                    train_series, target_series, train_seriesdep = [],[],[]
                    train_timedep, target_timedep = [],[]


# In[21]:


traingen = train_datagen(values, timedep, seriesdep, 62, 10, 10, batch_size)
valgen = test_datagen(values, timedep, seriesdep, 62, batch_size)


# In[30]:


model = rnn_windowed.RNN()


# In[33]:


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
save_best_path = base_dir+'rnn_stage2_v4_lr1_weights.mdl'
with clock():
    model.fit(traingen, valgen, optimizer=optimizer, num_epochs=25, save_best_path=save_best_path)


# In[ ]:


save_best_path = base_dir+'rnn_stage2_v4_lr1_weights.mdl'
model = rnn_meta.RNN().cuda()
model.load_state_dict(torch.load(save_best_path))


# In[ ]:


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
save_best_path = base_dir+'rnn_stage2_v4_lr2_weights.mdl'
with clock():
    model.fit(traingen, valgen, optimizer=optimizer, num_epochs=20, save_best_path=save_best_path)


# In[ ]:


save_best_path = base_dir+'rnn_stage2_v4_lr2_weights.mdl'
model = rnn_meta.RNN().cuda()
model.load_state_dict(torch.load(save_best_path))


# In[ ]:


outputs, targets, sequences = model.predict(valgen)


# In[ ]:


_, predictions = rnn_predict.combine_prediction_data(outputs, targets, sequences)


# In[ ]:


base_dir = '../data/'
train_df = pd.read_csv(base_dir+'train_2.csv')
X = train_df.drop('Page', axis=1).values


# In[ ]:


predictions = scaler.inverse_transform(predictions.T).T
true = X


# In[ ]:


smapes = val.smape(true[:,-60:], predictions[:,-60:], axis=1)
smapes_clipped = val.smape(true[:,-60:], predictions[:,-60:].round().clip(0), axis=1)


# In[ ]:


np.nanmean(smapes), np.nanmean(smapes_clipped)


# In[ ]:


np.save(base_dir+'rnn_v3_predictions.npy', outputs)

