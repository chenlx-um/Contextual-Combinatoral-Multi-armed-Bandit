# -*- coding: utf-8 -*-
"""
Created on Mon May  7 14:48:55 2018

@author: Lixing Chen
"""

import  pandas as pd
import numpy as np

def greedy_solu(user_context, review_context, business_context, B):
    p = 3
    for idx, x in review_context.iterrows():
        review_context.ix[idx,'mar_quality'] = user_context.loc[user_context['user_id'] == x['user_id'], 'quality'].values
    
    business_context['reward'] = [0.0]*business_context.shape[0]
    selected_review =  pd.DataFrame() 
    selected_bus = ''
    for k in range(0,B):
        marg_recomp = review_context.loc[ review_context['business_id'] == selected_bus]
        for idx, x in marg_recomp.iterrows():
            temp1 = business_context.loc[(business_context['business_id'] == x['business_id']), 'reward']
            temp2 = user_context.loc[(user_context['user_id'] == x['user_id']),'quality']
            review_context.ix[idx,'mar_quality'] = (temp1.values**p+temp2.values**p)**(1/p) - temp1.values
           # review_context.ix[idx,'mar_quality'] = ((business_context.loc[(business_context['business_id'] == x['business_id']), 'reward'])**p + (user_context.loc[(user_context['user_id'] == x['user_id']),'quality'])**p)**(1/p) - business_context.loc[(business_context['business_id'] == x['business_id']), 'reward']
          
        max_mar_value = review_context['mar_quality'].max()
        max_mar_index = review_context['mar_quality'].idxmax()
        selected_bus =  review_context.ix[max_mar_index, 'business_id']
        business_context.loc[business_context['business_id'] == review_context.ix[max_mar_index,'business_id'],'reward'] += max_mar_value
        selected_review = selected_review.append(review_context.ix[max_mar_index])
        
        
        review_context = review_context.drop(max_mar_index)

    return selected_review
    