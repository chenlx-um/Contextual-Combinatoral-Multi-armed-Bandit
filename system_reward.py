# -*- coding: utf-8 -*-
"""
Created on Tue May  8 17:22:59 2018

@author: Lixing Chen
"""


import  pandas as pd
import numpy as np

def get_reward(selected_review):
    p = 3
    business = pd.DataFrame() 
    business['business_id'] = np.unique(selected_review['business_id'])
    business['reward'] = [0.0]*business['business_id'].shape[0]
    temp1 = 0
    for idx, x in enumerate(business['business_id']):
        temp = selected_review.loc[selected_review['business_id'] == x]
        for x1 in temp['quality']:
            temp1 = temp1 + x1**p
        business.ix[idx, 'reward'] = temp1**(1/p)
    
    system_reward = sum(business['reward'])
    
    return system_reward
        
    