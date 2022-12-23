# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import networkx as nx
import time
import timeit
import datetime
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
import yfinance.shared as shared

#!pip install finta
from finta import TA
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

#!pip install PyPortfolioOpt
from pypfopt import expected_returns, efficient_frontier, risk_models
from datetime import datetime, timedelta

def test():
    return np.random.choice(2)

def feature_generate(ticker,OHLC,start,end):
    """Generate features from downloaded stocks' data
    """
    
    ohlc = OHLC.loc[start:end]
    sta = pd.DataFrame()
    sta["return"] = (ohlc["Close"] - ohlc["Close"].shift())/ohlc["Close"].shift()
    sta["open"] = ohlc["Open"]/ohlc["Close"]
    sta["high"] = ohlc["High"]/ohlc["Close"]
    sta["low"] = ohlc['Low']/ohlc["Close"]
    sta["volume"] = (ohlc["Volume"] - ohlc["Volume"].shift())/ohlc["Volume"].shift()
    
    df = pd.DataFrame()
    df['SMA']= TA.SMA(ohlc)
    df['SMM'] = TA.SMM(ohlc)
    df['ER'] = TA.ER(ohlc)
    df['MACD'] = TA.VW_MACD(ohlc).iloc[:,0]
    df['signal'] = TA.VW_MACD(ohlc).iloc[:,1]
    df['MOM'] = TA.MOM(ohlc)
    df['ROC'] = TA.ROC(ohlc)
    df['RSI'] = TA.RSI(ohlc)
    df['TR']= TA.TR(ohlc)
    df['SAR'] = TA.SAR(ohlc)

    df['BBWIDTH'] = TA.BBWIDTH(ohlc)
    df['KC_UPPER'] = TA.KC(ohlc).iloc[:,0]
    df['KC_LOWER'] = TA.KC(ohlc).iloc[:,1]
    df['DO'] = TA.DO(ohlc).iloc[:,1]
    df['DMI'] = TA.DMI(ohlc).iloc[:,1]
    
    final = sta.join(df)
    final.columns = pd.MultiIndex.from_product([[ticker], final.columns])
    
    return final


def merge_df(stocks,ohlc,start,end):
    """Input: stocks name, corresponding ohlc dataframe, start(dt), end(dt)
    Output: merged multiindex dataframe with stocks and corresponding feature values
    """
    
    read_start = start-timedelta(days=60)
    features = feature_generate(stocks[0],ohlc[stocks[0]],read_start,end)
    for ticker in stocks[1:]:
        features = features.join(feature_generate(ticker,ohlc[ticker],read_start,end))
    df = features.loc[start:end,:]
    
    return df


def pca_df(components_df,q,merged_df):
    """Input: stock price dataframe (will inplace price with pca values)
    Output: dataframe of stock pca dataframe
    """
    
    pca = PCA(n_components=q)
    for date in merged_df.index:
        df = merged_df.loc[[date]]
        df = df.reset_index().drop('Date', axis=1, level=0).T.reset_index()
        df.columns = ['ticker', 'type', 'value']
        df = df.pivot_table(index='ticker', columns='type', values='value')
        df = df.apply(lambda x: (x-np.mean(x))/np.std(x),axis=0)
        df.fillna(0, inplace=True)
        pca.fit_transform(df)
        weight = pca.components_[:1,:]
        pc = df.dot(weight.T)
        pc.apply(lambda x: (x-np.mean(x))/np.std(x),axis=0)
        for ticker in pc.index:
            components_df.loc[date][ticker] = pc.loc[ticker, 0]
            
    return components_df


def PCA_distance(PCA_res):
    """To construct a 'matrix' representing 'weights' between points
    time-T, vector-q, number of stocks-N
    N * (q*T) matrix
    norm: distance norm 
    Output: N*N matrix"""
    
    T,N = PCA_res.shape
    res = np.zeros((N,N))
    for i in range(1,N):
        for j in range(i):
            Diff = PCA_res.iloc[:,i] - PCA_res.iloc[:,j]
            distance = sc.linalg.norm(Diff,1)
            res[i][j] = 1/distance   # larger distance means weaker correlation
    res = res + res.T  # symmetric matrix with diagonals vanish
    
    return res


def stock_pool(current_time,existing_mon,components_mark_df):
    """
    Obtain a pool of always-existing stocks in a past period of time
    input: current time (type:datetime), minimal months(type: int) that stocks have been existing
    output: a list of stocks
    """
    
    end = current_time
    start = end - relativedelta(months=existing_mon)
    start_string = start.strftime('%Y-%m-%d')
    end_string = end.strftime('%Y-%m-%d')
    new_df = components_mark_df[components_mark_df.index<=end_string]
    new_df = new_df[new_df.index>=start_string]
    always_stocks = list(new_df.columns[new_df.notnull().all()])
    
    return always_stocks

def temp_ohlc(current_time,pca_mon,components_mark_df,add_mon=2):
    """additional_time: 60 days so 2 months, consistent with merged_df: read_start=start-60days
    if we do 12-month pca, we need 12+2 months to generate features & ohlc
    so we check stocks that have been existing for a longer time
    """
    
    existing_mon = pca_mon + add_mon
    tickers = stock_pool(current_time,existing_mon,components_mark_df)
    
    end = current_time
    start = end - relativedelta(months=existing_mon)
    ohlc = yf.download(tickers,start,end,group_by='ticker')
    failed_tickers=list(shared._ERRORS.keys())
    
    # It will also return stock list and tickers that failed to download
    return ohlc, tickers, failed_tickers


def PCA_past_testPCA(current_time,pca_mon,components_mark_df,components_price_df,add_mon=2):
    import yfinance.shared as shared
    """Do PCA based on past 3 or 6 or 12 months' data
    ohlc: dataframe downloaded from yf before
    Output: dataframe with test scores of PCA
    """
    
    end = current_time
    start = end - relativedelta(months=pca_mon)
    ohlc, test_tickers, failed_tickers = temp_ohlc(current_time,pca_mon,components_mark_df,add_mon)
    # test_tickers = stock_pool(current_time,pca_mon+add_mon) # to be consistent with ohlc input

    test_price = components_price_df.copy()[test_tickers]
    merged_df = merge_df(test_tickers,ohlc,start,end) 
    # e.g. start = 2021-06-01, ohlc obtained from last step starts from 2021-04-01
    # here,ohlc will be automatically read 2 months earlier than start -- 2021-04-01
    merged_df = merged_df.dropna(axis = 1)
    enddate = merged_df.index[-1]
    test_price = test_price[start:enddate]
    # due to some download failure, merged has less columns than test_price
    # valid_tickers = [t for t in test_tickers if t not in failed_tickers]
    for ticker in test_price.columns:
        if ticker in failed_tickers:
            test_price=test_price.drop([ticker],axis=1)

    components_df = test_price
    pca = pca_df(components_df,1,merged_df)
    
    return pca


def unfiltered_graph_from_PCA(current_time,pca_mon,components_mark_df,components_price_df,add_mon=2):
    """Output: a complete unfiltered graph and its relation matrix in a dataframe form
    """
    test_pca = PCA_past_testPCA(current_time,pca_mon,components_mark_df,components_price_df,add_mon)
    relation_matrix = PCA_distance(test_pca)
    rel_mat = pd.DataFrame(data=relation_matrix,columns=test_pca.columns,index =test_pca.columns)
    G0 = nx.from_pandas_adjacency(rel_mat)    # weights are 1/distance
    
    return G0, rel_mat


def PMFG(G):
    """Planar maximal filtering graph, |E| = 3|V| - 2
    """
    
    # sort edges weight in descending order
    h = []
    for u,v,d in G.edges(data=True):
        heappush(h,(d['weight'],u,v))
    heapsort = [heappop(h) for i in range(len(h))]
    heapsort.reverse()
    
    # PMFG algorithm
    res = nx.Graph()
    for (w,u,v) in heapsort:
        res.add_edge(u,v,weight=w)
        if not planarity.is_planar(res):
            res.remove_edge(u,v)
            
        if res.number_of_edges() == 3*(G.number_of_nodes()-2):
            break   

    return res



def MST(G):
    """Minimal spanning tree, |E| = |V| - 1
    """
    
    # sort edges weight in descending order
    h = []
    for u,v,d in G.edges(data=True):
        heappush(h, (d['weight'],u,v))
    heapsort = [heappop(h) for i in range(len(h))]
    heapsort.reverse()
    
    # kruskal MST algorithm
    def find_subtree(parent, i):
        # i represents index of nodes
        if parent[i] == i:
            return i
        return find_subtree(parent, parent[i])
            
    res = nx.Graph()
    parent = [j for j in range(len(G.nodes))]
    subtree_sizes = [0]*len(G.nodes)
    i = 0
    e = 0 # number of edges in MST
    
    while e<(len(G.nodes)-1):
        w, u, v = heapsort[i]
        i += 1
        u_index = list(G.nodes).index(u)
        v_index = list(G.nodes).index(v)
        x = find_subtree(parent, u_index)
        y = find_subtree(parent, v_index)
        # if u and v belongs to different subtree
        if x!=y:
            e += 1
            res.add_edge(u, v, weight = w)
            # connect two trees
            parent[y] = x
    return res 



def WTA(G,threshold,method='proportion'):
    """Threshold method, winner-take-all
    Can take the resulting subgraph fromm SS as input
    """
    
    # sort edges weight in descending order
    h = []
    for u,v,d in G.edges(data=True):
        heappush(h,(d['weight'],u,v))
    heapsort = [heappop(h) for i in range(len(h))]
    heapsort.reverse()
    
    res = nx.Graph()
    
    if method == 'proportion':
        # total number of possible edges
        num_nodes = len(list(G.nodes()))
        num_all_possible_edges = num_nodes * (num_nodes - 1) / 2
        target_num = math.ceil(threshold * num_all_possible_edges)
        cnt = 0
        for (w,u,v) in heapsort:
            if cnt <= target_num:
                res.add_edge(u,v,weight=w)
                cnt += 1
        return res
    
    elif method == 'value':
        for (w,u,v) in heapsort:
            if w >= threshold:
                res.add_edge(u,v,weight=w)
        return res

    else:
        print('Method should be either proportion or value')

        
        
def SS(G,matrix,max_vertices,rescale_prob=0.5):
    """ Spread sampling
    Input: G-the complete graph, matrix-relationship matrix, 
    rescale_prob: to rescale the probability of a node to be chosen
    max_vertices: number of vertices of the subgraph is bounded by this
    Output: a graph with nodes S, as a subgraph of G
    """
    
    # obtain a dictionary (probability of choosing a node or not) based on relevance matrix
    N = matrix.shape[0]  # number of nodes in the original graph G
    total = np.dot(np.abs(matrix),np.ones(N))   # matrix @ e
    scaled = rescale_prob * total / np.max(total) # [0,rescale_prob]
    d = {}
    C = list(G.nodes())
    for i in range(len(C)):
        d[C[i]] = scaled[i]
        
    # SS: to obtain S, nodes of the subgraph
    S = []   # nodes of the subgraph
    R = []   # nodes removed from candidate set due to much connections to nodes in S
    
    threshold = np.mean(total)  # probably this is not optimal
    
    while (len(S) < max_vertices) and (len(C) > 0):
        for node in C: # to add some point into S while removing them from C; C is the candidate set for S
            indicator = sc.stats.bernoulli.rvs(d[node])
            if indicator:
                C.remove(node)
                S.append(node)
        for node in C: # to remove nodes in C where it has much 'conection' to S
            count = 0
            for s in S:
                count += G[s][node]['weight']
            if count >= threshold:
                C.remove(node)
                R.append(node)
                

                    
    # subgraph construction (it's also complete graph)
    res = nx.Graph()
    for node in S:
        for nbr in nx.neighbors(G,node):
            if nbr in S:
                res.add_edge(node,nbr,weight=G.edges[node, nbr]['weight'])
                
    return res


def SS_WTA(G,matrix,threshold,max_vertices,method='proportion',rescale_prob=0.2):
    """SS to select stocks, WTA to filter
    """
    
    subgraph = SS(G,matrix,max_vertices,rescale_prob)
    res = WTA(subgraph,threshold,method)
    
    return res


def graph_filter(G0,rel_mat,filter_method,threshold=0.02,max_vertices=400,rescale_prob=0.2):
    """input a complete unfiltered graph from a relationship matrix
    output a filtered graph by PMFG/MST/WTA, on which we do centrality sort for candidate stocks
    """
    if filter_method == 'PMFG':
        G = PMFG(G0)
    elif filter_method == 'MST':
        G = MST(G0)
    elif filter_method == 'WTA_value':
        G = SS_WTA(G0,rel_mat,threshold,max_vertices,'value',rescale_prob)
    elif filter_method == 'WTA_proportion':
        G = SS_WTA(G0,rel_mat,threshold,max_vertices,'proportion',rescale_prob)
    else:
        print('Filtering method should be one of PMFG/MST/WTA_value/WTA_proportion')
    return G


def centrality_sort(G_filtered,method):
    """Input: filtered graph
    method: distance -- reciprocal based
    method: degree
    """
    if method == 'distance':
        centrality = nx.closeness_centrality(G_filtered)
    elif method == 'degree':
        centrality = dict(G_filtered.degree)
    else:
        print('Should use either distance or degree as input method')
        
    centrality_sorted = {k: v for k, v in sorted(centrality.items(), key=lambda item: item[1])}
    centrality_rank = dict(zip(list(centrality_sorted.keys()),[i for i in range(len(centrality_sorted))]))
    
    return centrality_rank


def centrality_select(G_filtered,central_method,k,selection_way='interval'):
    """ Input: a filtered graph, centrality method: 'distance' or 'degree' in giving a rank
    k: number of stocks selected
    selection way: interval -- from top and evenly spread throughout the pool
    top -- stocks with top centrality rank; bottom; random (randomly select k)
    Output: an array of k selected stocks
    """
    centrality_rank = centrality_sort(G_filtered,central_method)
    # e.g. 204 nodes, k=10, choose every 20
    nodes = list(G_filtered.nodes())
    num_nodes = len(nodes)
    node_list = []
    
    if selection_way == 'interval':
        interval = math.floor(len(nodes)/k)
        for node in nodes:
            if centrality_rank[node] % interval == 0:
                node_list.append(node)
    elif selection_way == 'top':
        for node in nodes:
            if centrality_rank[node] < k:
                node_list.append(node)
    elif selection_way == 'bottom':
        for node in nodes:
            if centrality_rank[node] >= num_nodes - k:
                node_list.append(node)
    elif selection_way == 'random':
        node_list = random.sample(nodes,k)
    elif selection_way == 'top_bot':
        k_top = k/2
        k_bot = k/2
        for node in nodes:
            if centrality_rank[node] < k_top:
                node_list.append(node)
            elif centrality_rank[node] >= num_nodes - k_bot:
                node_list.append(node)
    else:
        print('Please input selection_way in one of: interval, top, bottom, random.')   
    return np.array(node_list)