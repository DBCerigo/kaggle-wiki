from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

def load_prophet_prop(VERSION, prop, force_remake=False, test=None):
    PROPHET_PATH = '../data/prophet/'
    CACHE_PATH = 'cache/'
    assert VERSION[-1] == '/'
    df_path = PROPHET_PATH+CACHE_PATH+VERSION[:-1]+prop+'.f'
    if os.path.isfile(df_path) and not force_remake:
        return pd.read_feather(df_path)
    else:
        df = pd.read_feather('../data/train.f')
        forecast_files = [x.split('/')[-1] for x in glob.glob(PROPHET_PATH+VERSION+'*df.f')]
        init_forc = pd.read_feather(PROPHET_PATH+VERSION+forecast_files[0])
        ds_min = init_forc.ds.min().date()
        ds_max = init_forc.ds.max().date()
        df = df.loc[:,str(ds_min):str(ds_max)]
        df.loc[:] = np.nan
        try:
            assert df.shape[1]-1 == (ds_max-ds_min).days
        except AssertionError:
            assert df.shape[1] == 803
        for file_path in tqdm(forecast_files[:test]):
            forecast = pd.read_feather(PROPHET_PATH+VERSION+file_path)
            df.loc[int(file_path[:-4])] = forecast.iloc[:-64][prop].values
        df.sort_index(inplace=True)
        df = df.apply(pd.to_numeric)
        df.to_feather(df_path)
    return df

def prophet_plot(df, ax=None, uncertainty=True, plot_cap=True, plot_y_org=True, plot_yhat_org=False, plot_y=True,
         xlabel='ds',
         ylabel='y_org'):
    """Plot the Prophet forecast.
    Parameters
    ----------
    fcst: pd.DataFrame output of self.predict.
    ax: Optional matplotlib axes on which to plot.
    uncertainty: Optional boolean to plot uncertainty intervals.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    xlabel: Optional label name on X-axis
    ylabel: Optional label name on Y-axis
    Returns
    -------
    A matplotlib figure.
    """
    if ax is None:
        fig = plt.figure(facecolor='w', figsize=(14, 8))
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    df_train = df[df['train'] == 1]
    df_val = df[df['train'] == 0]
    if plot_y_org:
        ax.plot(df_train['ds'].values, df_train['y_org'], 'k')
    ax.plot(df_val['ds'].values, df_val['y_org'], 'r')
    ax.plot(df['ds'].values, df['yhat'], ls='-', c='#0072B2')
    if plot_y:
        ax.plot(df_train['ds'].values, df_train['y'], c='#3CB371', alpha=0.95)
    if plot_yhat_org:
        ax.plot(df_train['ds'].values, df_train['yhat_org'], c='#00FFFF', alpha=0.95)
    if 'cap' in df and plot_cap:
        ax.plot(df['ds'].values, df['cap'], ls='--', c='k')
    if uncertainty:
        ax.fill_between(df['ds'].values, df['yhat_lower'],
                        df['yhat_upper'], color='#0072B2',
                        alpha=0.15)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
