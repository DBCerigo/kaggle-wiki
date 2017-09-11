
# coding: utf-8

# In[1]:


import sys
import gc

from multiprocessing import cpu_count, Pool
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data_utils

from tqdm import tqdm

sys.path.append('../')
from wiki.utils import clock
from wiki import rnn, rnn_predict, newphet, val


# In[2]:


torch.manual_seed(1)
torch.cuda.set_device(0)


# In[3]:


batch_size = 1024
pred_len = 62


# In[4]:


base_dir = '../data/'
train_df = pd.read_csv(base_dir+'train_2.csv').fillna(0)


# In[5]:


X = train_df.drop('Page', axis=1).values
X, scaler = rnn.scale_values(X)


# In[6]:




# In[7]:


trainloader = data_utils.DataLoader(
    data_utils.TensorDataset(
        torch.from_numpy(X[:,-pred_len-430:-pred_len,:]).float(),
        torch.from_numpy(X[:,-pred_len:,:]).float()
    ),
    batch_size=batch_size, shuffle=False
)


# In[8]:


testloader = data_utils.DataLoader(
    data_utils.TensorDataset(
        torch.from_numpy(X[:,pred_len:,:]).float(),
        torch.from_numpy(X[:,-pred_len:,:]).float()
    ),
    batch_size=batch_size, shuffle=False
)


# In[9]:


loss_func = nn.L1Loss()
model = rnn.RNN(loss_func=loss_func).cuda()


# In[60]:


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
with clock():
    model.fit(trainloader, valloader=None, optimizer=optimizer, num_epochs=17)


# In[ ]:


save_best_path = base_dir+'rnn_stage2_FINAL_v2_lr1.mdl'
torch.save(model.state_dict(), save_best_path)


# In[ ]:


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
with clock():
    model.fit(trainloader, valloader=None, optimizer=optimizer, num_epochs=8)


# In[10]:


save_best_path = base_dir+'rnn_stage2_FINAL_v2_lr2.mdl'
torch.save(model.state_dict(), save_best_path)


# In[11]:


outputs, targets, sequences = model.predict(testloader)


# In[12]:


_, predictions = rnn_predict.combine_prediction_data(outputs, targets, sequences)


# In[14]:


predictions = scaler.inverse_transform(predictions.T).T


# In[16]:


predictions = predictions.round().clip(0)


# In[ ]:


fp = '../data/submissions/rnn_v2_final.csv'
submissions.write_submission(predictions, fp)

print('FINISHED PREDICTION')
