# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import networkx as nx
import time
import timeit
# import datetime
from datetime import datetime,timedelta
from dateutil.relativedelta import relativedelta
import http.client
import requests
import json
import yfinance as yf
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from heapq import *
import planarity
from sklearn.linear_model import LinearRegression
import scipy as sc
import math
import random

#!pip install finta
from finta import TA
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

#!pip install PyPortfolioOpt
from pypfopt import expected_returns, efficient_frontier, risk_models

def weight_equal(date,stocks_lst, components_price_df):
    # return a equal weight vector of this stock list
    num_stocks = len(stocks_lst)
    weight_series = pd.Series(np.ones(num_stocks)/num_stocks, index = stocks_lst)
    return weight_series
    

# bug here
def weight_low_vol(date,stocks_lst, components_price_df):
    stocks_price_df = components_price_df.copy().loc[(date-timedelta(days=365)):date, stocks_lst]
    stocks_ret_df = stocks_price_df.pct_change().dropna()
    vol_series = stocks_ret_df.std()
    weight_series = (1/vol_series)
    weight_series[weight_series>999]=0
    weight_series = weight_series/ weight_series.sum()
    return weight_series

def weight_markowitz(date, stocks_lst, components_price_df):
    df = components_price_df.loc[(date-timedelta(days=365)):date, stocks_lst]
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.CovarianceShrinkage(df).ledoit_wolf()
    ef = efficient_frontier.EfficientFrontier(mu, S)
    weight_dict = ef.min_volatility()
    
    weight_series = pd.Series(dtype='float64')
    for (k, v) in weight_dict.items():
        weight_series[k] = v
    return weight_series

def weight_price(date, stocks_lst, components_price_df):
    df = components_price_df.copy()
    df = df[~df.index.to_period('m').duplicated()]
    valid_dates = df.index
    date = pd.to_datetime(date)
    for d in valid_dates:
        if date <= d:
            date = d
            break
    stocks_price = components_price_df.loc[date, stocks_lst]
    total_price = stocks_price.sum()
    weight_series = stocks_price/total_price
    
    return weight_series

def weight_return(date, stocks_lst, components_price_df):
    df = components_price_df.copy()
    df = df[~df.index.to_period('m').duplicated()]
    valid_dates = df.index
    date = pd.to_datetime(date)
    for d in valid_dates:
        if date <= d:
            date = d
            break
    date_index = list(components_price_df.index).index(date)
    pre_date = list(components_price_df.index)[date_index-20]
    stocks_return = (components_price_df.loc[date, stocks_lst]-components_price_df.loc[pre_date, stocks_lst])/components_price_df.loc[pre_date, stocks_lst]

    total_return = stocks_return.sum()
    
    weight_series = stocks_return/total_return
    
    return weight_series



# def weight_cap
"""function to be written, weights based on capitalization
this way makes more sense for replicating purpose
"""


def weight(date, stocks_lst, components_price_df, weight_method='cap'):
    if weight_method == 'equal':
        weight_series = weight_equal(date,stocks_lst, components_price_df)
    elif weight_method == 'low_vol':
        weight_series = weight_low_vol(date,stocks_lst, components_price_df)
    elif weight_method == 'markowitz':
        weight_series = weight_markowitz(date, stocks_lst, components_price_df)
    elif weight_method == 'return':
        weight_series = weight_return(date, stocks_lst, components_price_df)
    elif weight_method == 'price':
        weight_series = weight_price(date, stocks_lst, components_price_df)
    else:
        print('Weight input should be one of equal, low_vol, markowitz and cap.')
    return weight_series

