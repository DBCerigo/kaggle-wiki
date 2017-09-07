
# coding: utf-8

# In[1]:


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
from wiki import rnn, rnn_predict, newphet, val, submissions, rnn_meta as rnn_meta


# In[2]:


base_dir = '../data/'
pred_len = 60
batch_size = 1024


# In[3]:


train_df = pd.read_csv(base_dir+'train_1.csv').fillna(0)


# In[4]:


values = train_df.drop('Page', axis=1).values ; values.shape


# In[5]:


dates = train_df.columns[1:].values
s_date = dates[0]
e_date = dates[-1]


# In[6]:


dates = pd.date_range(s_date, e_date)


# In[7]:


ages = np.arange(len(dates))
dows = dates.dayofweek.values
woys = dates.weekofyear.values


# In[8]:


#Expand the dims to make broadcasting work - since numpy
#refuses to add dimensions to the right when broadcasting
series_idxs = np.expand_dims(np.arange(values.shape[0]), axis=-1)


# In[9]:


values, scaler = rnn.scale_values(values)


# In[10]:


values = values.squeeze()


# In[11]:


br = lambda x: np.broadcast_to(x, values.shape)


# In[12]:


features = np.stack([values, br(ages), br(dows), br(woys), br(series_idxs)], axis=-1)


# In[13]:


features.shape


# In[14]:


trainloader = data_utils.DataLoader(
    data_utils.TensorDataset(
        torch.from_numpy(features[:,:-2*pred_len,:]).float(),
        torch.from_numpy(features[:,-2*pred_len:-pred_len,:]).float()
    ),
    batch_size=batch_size, shuffle=False
)
valloader = data_utils.DataLoader(
    data_utils.TensorDataset(
        torch.from_numpy(features[:,:-pred_len,:]).float(),
        torch.from_numpy(features[:,-pred_len:,:]).float()
    ),
    batch_size=batch_size, shuffle=False
)


# In[15]:


model = rnn_meta.RNN().cuda()


# In[16]:


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
save_best_path = base_dir+'rnn_v3_lr1_weights.mdl'
with clock():
    model.fit(trainloader, valloader, optimizer=optimizer, num_epochs=20, save_best_path=save_best_path)


# In[17]:


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
save_best_path = base_dir+'rnn_v3_lr2_weights.mdl'
with clock():
    model.fit(trainloader, valloader, optimizer=optimizer, num_epochs=10, save_best_path=save_best_path)


# In[ ]:


save_best_path = base_dir+'rnn_v3_lr2_weights.mdl'
model = rnn_meta.RNN().cuda()
model.load_state_dict(torch.load(save_best_path)).cuda()


# In[ ]:


outputs, targets, sequences = model.predict(valloader)


# In[ ]:


np.save(base_dir+'rnn_v3_predictions.npy', outputs)

