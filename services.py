import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import bisect

import calendar
import datetime


class VolService(object):

    @classmethod
    def calc_realized_vol(cls, trading_days, underlying_series):
        returns = np.log(underlying_series / underlying_series.shift(1))
        returns.fillna(0, inplace=True)
        return returns.rolling(window=trading_days).std() * np.sqrt(trading_days)


class PlotlyFormattingService(object):

    @classmethod
    def update_layout_quick(cls, fig, title=None, xaxis_title=None, yaxis_title=None, yaxis_tickformat='.2f'):
        return fig.update_layout(
            title={
                'text': title,
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
            },

            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            legend_title_text=' ',

            xaxis={
                'showline': True,
                'showgrid': True,
                'linecolor': 'black',
                'tickfont': {'family': 'Calibri'}
            },

            yaxis={
                'showline': True,
                'showgrid': True,
                'linecolor': 'black',
                'tickfont': {'family': 'Times New Roman'},
                'tickformat': yaxis_tickformat,
            },

            legend={
                'yanchor': 'top',
                'y': 0.6,
                'xanchor': 'left',
                'x': 1.05,
            },

            plot_bgcolor='white'
        )


class DataQuickGlanceService(object):

    @classmethod
    def draw_distplot(cls, df, cols, figsize=(10, 15)):
        ls_len = len(cols)
        fig, axes = plt.subplots(ls_len, 1, sharex=True, figsize=figsize)
        for col, i in zip(cols, range(0, ls_len)):
            sns.distplot(ax=axes[i], x=df[col])
            axes[i].set_title(col)
        return fig

    @classmethod
    def summarize_missing_data(cls, df):
        return pd.DataFrame(
            [(i, df[df[i].isna()].shape[0], df[df[i].isna()].shape[0] / df.shape[0]) for i in df.columns],
            columns=['column', 'nan_counts', 'nan_rate'])


class DateService(object):

    @classmethod
    def get_expiry_dates_for_year(cls, year):
        cal = calendar.Calendar(4)
        expiries = list()
        for month in [3, 6, 9, 12]:
            dates = list(cal.itermonthdates(year, month))
            expiries.append(dates[14 + (dates[0].month != month) * 7])
        return expiries

    @classmethod
    def get_next_expiry(cls, date=datetime.datetime.today().date()):
        expiries = cls.get_expiry_dates_for_year(date.year) + cls.get_expiry_dates_for_year(date.year + 1)
        return expiries[bisect.bisect_left(expiries, date)]

    @classmethod
    def get_prev_expiry(cls, date=datetime.datetime.today().date()):
        expiries = cls.get_expiry_dates_for_year(date.year - 1) + cls.get_expiry_dates_for_year(date.year)
        return expiries[bisect.bisect_left(expiries, date) - 1]
