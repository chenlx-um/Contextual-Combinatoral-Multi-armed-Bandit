# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 13:23:28 2018

@author: Lixing Chen
"""

import pandas as pd
import numpy as np

def get_counter_quality(review_context_cur,user_context_cur,context_space):
    dimension = context_space.count()
    # num_hycub  = np.prod(dimension.values,dtype=np.int64)
    counters = np.full((dimension.values),0.0)
    quality = np.full((dimension.values),0.0)
    for idx, x in review_context_cur.iterrows():
        user_id = x['user_id']
    # get the counter index
        p_elite_idx = int(user_context_cur.loc[user_context_cur['user_id'] == user_id,'p_elite_idx'].values)
        p_fans_idx = int(user_context_cur.loc[user_context_cur['user_id'] == user_id,'p_fans_idx'].values)
        counters[p_elite_idx, p_fans_idx] += 1
        quality[p_elite_idx, p_fans_idx] += x['quality']
    return counters, quality
'''
def get_demand(record,context_space):
    dimension = context_space.count()
    # num_hycub  = np.prod(dimension.values,dtype=np.int64)
    demand = np.full((dimension.values),0)
    record_num = record.shape[0]
    for idx in range(0,record_num):
        record_temp = record.iloc[idx]    
        record_temp = record_temp.fillna(1000)
        c_names = context_space.columns
    # get the counter index
        count_idx=[];
        for temp_c_name in c_names:
            for i in range(0,context_space[temp_c_name].count()):
                if int(record_temp[temp_c_name]) in np.array(context_space[temp_c_name][i]):
                    count_idx = count_idx + [i]
                    break
        if record_temp['Q15_12'] == 1:       
            demand[tuple(count_idx)] += record_temp['Q15_12']
    return demand
'''