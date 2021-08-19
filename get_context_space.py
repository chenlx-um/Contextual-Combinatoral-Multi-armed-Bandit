#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 01:09:34 2018

@author: lixingchen
"""

import  pandas as pd
import numpy as np

def get_context_space():
    #app_total_type = 23
    #record_num=np.zeros(app_total_type+1)
    #record =  pd.read_excel('app_user_dataset_modified.xlsx', sheet_name = 'Sheet1')
    
    context_space =  pd.DataFrame() 
    
    '''
    column_list = record.columns.values.tolist()
    for column_name in column_list:
        temp =  pd.DataFrame({column_name : record[column_name].unique()})
        temp = temp.fillna(1000)
        context_space =  pd.concat([context_space, temp], axis=1) 

    screen_context=context_space['Q1_4_TEXT']  ## Screen size
    screen_context=screen_context.replace(to_replace = 'NaN', value = '0x0')  
    screen_split=screen_context.str.split('x')
    screen_context = ( pd.to_numeric(screen_split.str[0]) < 600).astype(int) + ( pd.to_numeric(screen_split.str[1]) < 600).astype(int)
    context_space['Q1_4_TEXT'] =  pd.DataFrame({'Q1_4_TEXT': screen_context.unique()})
    '''    
    
    #elite_context
    l = range(0,15)
    n = 3;
    elite_chunks=[l[i:i + n] for i in range(0, len(l), n)]
    context_elite =  pd.DataFrame() 
    context_elite['elite_range'] = pd.Series(elite_chunks).values 
    context_space['elite'] = context_elite['elite_range']
    
    l = range(0,100)
    n = 25;
    fans_chunks=[l[i:i + n] for i in range(0, len(l), n)]
    fans_chunks.append(range(100,1100))
    context_fans =  pd.DataFrame() 
    context_fans['fans_range'] = pd.Series(fans_chunks).values 
    context_space['fans'] = context_fans['fans_range']
 
    # context_space = context_space.dropna(axis=0, how='all', thresh=None, subset=None, inplace=False)
    
    return context_space