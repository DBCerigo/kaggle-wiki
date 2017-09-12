
# coding: utf-8

# In[1]:




# In[2]:


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [16,9]
plt.style.use('ggplot')
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
from wiki.newphet import load_prophet_prop
import multiprocessing as mp
total_proc = None
from tqdm import tqdm
from IPython.display import clear_output
import gc


# In[3]:


def load_plot(page, plot_y_org=True):
    forecast = pd.read_feather(PROPHET_PATH+VERSION+page+'df.f')
    wiki.newphet.prophet_plot(forecast, plot_y_org=plot_y_org)
    plt.show()
    forecast = pd.read_feather(PROPHET_PATH+YVERSION+page+'df.f')
    wiki.newphet.prophet_plot(forecast, plot_y_org=plot_y_org)
    plt.show()


# In[4]:


PROPHET_PATH = '../data/prophet/'
CACHE_PATH = 'cache/'
RESULTS_PATH = 'results/'
VERSION ='v7t/'
YVERSION ='v7ty/'


# In[5]:


mediandf = pd.read_feather('../data/median_for_120_60.f') 


# In[6]:


test_df = pd.read_feather(PROPHET_PATH+RESULTS_PATH+VERSION[:-1]+'df.f'); 
test_df.page_index = test_df.page_index.astype(int)
print(test_df[VERSION[:-1]+'_train'].mean())
test_df = test_df.sort_values(by='page_index').reset_index().drop('index', axis=1)
#test_df = test_df.drop(VERSION[:-1]+'_train', axis=1)
print(test_df.shape)
print(test_df.info())
print(test_df[VERSION[:-1]+'_val'].mean())
test_df.head()


# In[7]:


mediandf.page_index = mediandf.page_index.astype(int)
mediandf = mediandf.sort_values(by='page_index').reset_index().drop('index', axis=1)
print(mediandf.iloc[:,2:].mean())
print(mediandf.info())
mediandf.head()


# In[8]:


Ytest_df = pd.read_feather(PROPHET_PATH+RESULTS_PATH+YVERSION[:-1]+'df.f'); 
Ytest_df.page_index = Ytest_df.page_index.astype(int)
print(Ytest_df[YVERSION[:-1]+'_train'].mean())
Ytest_df = Ytest_df.sort_values(by='page_index').reset_index().drop('index', axis=1)
#Ytest_df = Ytest_df.drop(VERSION[:-1]+'_train', axis=1)
print(Ytest_df.shape)
print(Ytest_df.info())
print(Ytest_df[YVERSION[:-1]+'_val'].mean())
Ytest_df.head()


# In[9]:


df = Ytest_df.merge(test_df, on='page_index'); print(df.shape)
df = df.merge(mediandf, on='page_index'); print(df.shape)
df['Ytest_gain'] = df.prevYear_smape_60_to_0 - df[YVERSION[:-1]+'_val']
df['test_gain'] = df.smape_60_to_0 - df[VERSION[:-1]+'_val']; df.head()


# # Magic Feature

# In[10]:


median_rolling_smape =  wiki.val.load_test_median_rolling_smape()


# In[11]:


print(median_rolling_smape.iloc[[0],[0,-1]])
print(median_rolling_smape.shape)
median_rolling_smape.head()


# In[12]:


yhat_rolling_smape =  wiki.val.load_prophet_rolling_smape(VERSION, test_version=True)


# In[13]:


print(yhat_rolling_smape.iloc[[0],[0,-1]])
print(yhat_rolling_smape.shape)
yhat_rolling_smape.head()


# In[14]:




# In[15]:


# remove the yhat smape for the t
yhat_rolling_smape2 = yhat_rolling_smape.copy()
#yhat_rolling_smape2.iloc[:,-120:] = np.nan


# In[16]:


cut_off = None
df['rolling_gain_mean'] = (median_rolling_smape.iloc[:,:cut_off] - yhat_rolling_smape2.iloc[:,:cut_off]).mean(axis=1)
df['rolling_gain_std'] = (median_rolling_smape.iloc[:,:cut_off] - yhat_rolling_smape2.iloc[:,:cut_off]).std(axis=1)


# In[17]:


print(df['rolling_gain_mean'].max(), df['rolling_gain_mean'].min())
print(df['rolling_gain_std'].max(), df['rolling_gain_std'].min())


# # Shift preds

# In[18]:


y_df = wiki.newphet.load_prophet_prop(VERSION, 'y')


# In[19]:


y_df = y_df.shift(365, axis=1).fillna(0)


# In[20]:


train = pd.read_feather('../data/train_old.f')


# In[21]:


y_df = wiki.val.get_smape_df(train,y_df)


# In[22]:


y_df.head()


# In[23]:


df['shift_y_smape'] = y_df.iloc[:,-60:].mean(axis=1)


# In[24]:


df['shift_y_gain'] = df.smape_60_to_0 - df['shift_y_smape']


# # Filter CV

# In[25]:


BASE_SCORE = df.smape_60_to_0.mean()


# In[26]:


BASE_SCORE


# In[27]:


#v1{'a': 1.5, 'b': 2.0, 'c': 8.0, 'd': 0}
filter_df = ((df['rolling_gain_std']*1.5 < df['rolling_gain_mean']*2) &
            (df['Ytest_gain'] > 8) &
             (df['nans_start_to_120'] <= 0)
             )
print(len(df[filter_df]))


# In[28]:


pd.concat([df[filter_df].v7t_val,
           df[~filter_df].smape_60_to_0]).mean()


# In[29]:


pd.concat([df[filter_df].shift_y_smape,
           df[~filter_df].smape_60_to_0]).mean()


# # Parameter Selection

# In[30]:


from sklearn.model_selection import GridSearchCV


# In[31]:


def get_filter_score(a,b,c,d):
    filter_df = ((df['rolling_gain_std']*a < df['rolling_gain_mean']*b) &
            (df['Ytest_gain'] > c) &
             (df['nans_start_to_120'] <= d))
    return pd.concat([df[filter_df].shift_y_smape,
           df[~filter_df].smape_60_to_0]).mean()


# In[32]:


from sklearn.base import BaseEstimator, ClassifierMixin

class MeanClassifier(BaseEstimator):  
    """An example of classifier"""

    def __init__(self, a=1, b=1, c=0, d=0, filter_df=df):
        """
        Called when initializing the classifier
        """
        self.a=a
        self.b=b
        self.c=c
        self.d=d
        self.filter_df=filter_df


    def fit(self, df=None, y=None):
        #print(df.head())
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """

        self.filter_df = (((self.filter_df['rolling_gain_std']*self.a) < (self.filter_df['rolling_gain_mean']*self.b)) &
            (self.filter_df['Ytest_gain'] > self.c) &
             (self.filter_df['nans_start_to_120'] <= self.d))

        return self

    def predict(self, df):
        self.smapes = pd.concat([df[self.filter_df].shift_y_smape,
           df[~self.filter_df].smape_60_to_0])

        return(self.smapes)

    def score(self, df, y=None):
        # counts number of values bigger than mean
        #clear_output()
        gc.collect()
        return(-self.predict(df).mean()) 


# In[33]:


# Set the parameters by cross-validation
tuned_parameters = {'a': [i for i in np.arange(1,2,0.2)], 
                     'b': [i for i in np.arange(1.5,2.5,0.2)],
                     'c': np.arange(5,11,1),
                    'd': np.arange(0,700,300)}


# In[36]:


# In[37]:


gs = GridSearchCV(MeanClassifier(), tuned_parameters, cv=10, n_jobs=-1, error_score='raise',
                 verbose=60)


# In[38]:


gs.fit(df)


# In[ ]:




# In[ ]:


gs.fit(df)


# In[47]:


gs.best_params_ 
#v1{'a': 1.5, 'b': 2.0, 'c': 8.0, 'd': 0} -> 40.54743243933121


# In[ ]:


print('gs.best_params_ ',gs.best_params_ )


# In[38]:


df.shape


# In[39]:


print('gs.predict(df).mean()',gs.predict(df).mean())


# # Saving Results 

# In[45]:
