from matplotlib import pyplot as plt

def prophet_plot(df, ax=None, uncertainty=True, plot_cap=True, plot_yhat_org=False, plot_y=True,
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
    ax.plot(df_train['ds'].values, df_train['y_org'], 'k')
    ax.plot(df_val['ds'].values, df_val['y_org'], 'r')
    ax.plot(df['ds'].values, df['yhat'], ls='-', c='#0072B2')
    if plot_y:
        ax.plot(df_train['ds'].values, df_train['y'], c='#3CB371', alpha=0.2)
    if plot_yhat_org:
        ax.plot(df_train['ds'].values, df_train['yhat_org'], c='#00FFFF', alpha=0.2)
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
