import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import bisect
import calendar
import datetime

from matplotlib import colors

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


class DfFormatService(object):

    @classmethod
    def background_gradient(cls, s, m, M, cmap='Greens', low=0, high=0):
        rng = M - m
        norm = colors.Normalize(
            m - (rng * low),
            M + (rng * high),
        )
        normed = norm(s.values)
        c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
        return ['background-color: %s' % color for color in c]

'''
df = pd.DataFrame([[3, 2, 10, 4], [20, 1, 3, 2], [5, 4, 6, 1]])

df.style.apply(DfFormatService.background_gradient,
               cmap='Greens',
               m=df.min().min(),
               M=df.max().max(),
               low=0,
               high=0.2)
'''

#draw subplot
'''
fig = make_subplots(
    rows=2, 
    cols=1,
    vertical_spacing= 0.1,
    subplot_titles=("Realised Vol", "Random Chart")
)

fig.add_trace(
    go.Scatter(
        x = close_df['Date'],
        y = close_df['AAPL_realized_vol'],
        name="AAPL_realized_vol",
    ),
        row=1, 
        col=1,

)

fig.add_trace(
    go.Bar(
        x = close_df['Date'],
        y = close_df['GOOGL_realized_vol'],
        name = "GOOGL_realized_vol",
),
        row=1, 
        col=1,
)


fig.add_trace(
    go.Line(x=[20, 30, 40], 
            y=[50, 60, 70],
            name = "Random Test",
           ),
    
    row=2, col=1,
)

xaxes_layout = {
    'showline' : True,
    'showgrid' : True,
    'linecolor' :'black',
    'tickfont' : {'family' : 'Calibri'}
}

yaxes_pct_layout = {
    'showline' : True,
    'showgrid' : True,
    'linecolor' : 'black',
    'tickfont' : {'family' : 'Times New Roman'},
    'tickformat' : '.2%',
}

yaxes_number_layout = {
    'showline' : True,
    'showgrid' : True,
    'linecolor' : 'black',
    'tickfont' : {'family' : 'Times New Roman'},
    'tickformat' : '.2f',
}

fig.update_layout(
    
    height=1000, 
    width=800,
        
    title ={
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
    },

    legend_title_text=' ',
    
    xaxis1 = xaxes_layout, 
    xaxis2 = xaxes_layout,
            

    yaxis1 = yaxes_pct_layout,
    yaxis2 = yaxes_number_layout,

    legend = {
        'y' : 1.1,
        'x' : 0.1,
        'orientation' : "h",
    },
    
    plot_bgcolor = 'white'
)

fig.show()

'''
