import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import pickle # to access dataframe faster than csv
import glob, re
import os
import csv
from pathlib import Path
import scipy as sp

def blocked_cross_validation_idx(df,train_months=3,test_months=1,overlap_months=0):
    """
    return: list of tuples containing (train_start_idx, train_end_idx, test_start_idx, test_end_idx) for year 2012-2013
    """
    blocks=[]
    start_date=df['DateTime'].iloc[0]
    end_date=df['DateTime'].iloc[-1]
    current_train_start=start_date
    while current_train_start+pd.DateOffset(months=train_months+test_months)<=end_date+pd.Timedelta(hours=1):
        train_end=current_train_start+pd.DateOffset(months=train_months)-pd.Timedelta(hours=1)
        test_start=train_end+pd.Timedelta(hours=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(hours=1)
        train_start_idx=df.index[df['DateTime'] == current_train_start][0]
        train_end_idx=df.index[df['DateTime'] == train_end][0]
        test_end_idx=df.index[df['DateTime'] == test_end][0]
        test_start_idx=df.index[df['DateTime'] == test_start][0]
        blocks.append((train_start_idx, train_end_idx, test_start_idx, test_end_idx))
        current_train_start = current_train_start + pd.DateOffset(months=train_months + test_months - overlap_months)
    return blocks


def get_windows_idx(start,end,in_window,out_window):
    #timestamps=[]
    in_idx=[]
    out_idx=[]
    current=start
    while current+in_window+out_window<=end+1:
        in_idx.append(np.arange(start=current,stop=current+in_window))
        out_idx.append(np.arange(start=current+in_window,stop=current+in_window+out_window))
        #out_idx.append(pd.date_range(start=current+pd.Timedelta(hours=in_window),periods=out_window,freq='h'))
        current+=1
    return in_idx,out_idx


def get_train_batch_blocks(start,end,batch_size=64):
    blocks=[]
    curr_start=start
    while curr_start<end:
        curr_end=min(curr_start+batch_size,end)
        blocks.append(np.arange(curr_start,curr_end))
        curr_start+=batch_size
    return blocks