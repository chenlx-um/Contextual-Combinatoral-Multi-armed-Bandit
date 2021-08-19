# -*- coding: utf-8 -*-
"""
Created on Fri May  4 12:11:55 2018

@author: Lixing Chen
"""
#import data
import pickle
import get_context_space
import counter_quality as cq
import pandas as pd
import numpy as np
from numpy.linalg import inv
import greedy_algorithm as ga
from datetime import timedelta
import system_reward

import scipy.io

#import imp
#imp.reload(system_reward)
'''
review_raw= pd.read_pickle('data/yelp_academic_dataset_review.pickle')
user_raw = pd.read_pickle('data/yelp_academic_dataset_user.pickle')
business_raw = pd.read_pickle('data/yelp_academic_dataset_business.pickle')

user_context_all = user_raw[['user_id','elite','fans']]
user_context_all['total_quality'] = [0]*user_context_all.shape[0]
user_context_all['review_times'] = [0]*user_context_all.shape[0]
user_context_all['elite'] = [len(i) for i in user_context_all['elite']]
business_all = business_raw[['business_id']]

text_len = [min([len(i),500]) for i in review_raw['text']]
votes = review_raw[['votes_cool','votes_funny','votes_useful']].sum(axis = 1)
votes = [min([i,9]) for i in votes]
text_len_norm = (np.array(text_len) - min(text_len)) / (max(text_len)-min(text_len))
quality = (np.array(votes) + 1)/2*text_len_norm
review_raw['quality'] = quality

business_context = business_all.sample(1000,replace = False)
review_context_all = review_raw [['date','business_id','user_id','quality']]
review_context = review_context_all.loc[review_context_all['business_id'].isin(business_context ['business_id'])]
user_context = user_context_all.loc[user_context_all['user_id'].isin(review_context['user_id'])]

#for index, row in review_context.iterrows():
#    user_context.loc[user_context['user_id'] == row['user_id'],'review_times'] += 1  
#   user_context.loc[user_context['user_id'] == row['user_id'],'total_quality'] += row['quality'] 
for index, row in user_context.iterrows():
    temp = review_context.loc[review_context['user_id'] == row['user_id']]
    user_context.ix[index,'total_quality'] = temp['quality'].sum()
    user_context.ix[index,'review_times'] = temp.shape[0]
    
pickle.dump([business_context,user_context,review_context], open( "save.p", "wb" ) )
'''




load_data = pickle.load( open( "save.p", "rb" ) )
business_context = load_data[0]
user_context = load_data[1]
review_context = load_data[2]

user_context['quality'] = user_context['total_quality']/user_context['review_times']
#fan_norm = (np.array(user_context['fans']) - min(user_context['fans'])) / (max(user_context['fans'])-min(user_context['fans']))*50+1
user_context['fan_norm'] = [min([i,30]) + 1 for i in user_context['fans']]
user_context['quality'] = user_context[['quality','fan_norm']].prod(axis = 1)
user_context['ora_quality'] = user_context['quality']
user_context['est_quality'] = [0.0]*user_context['quality'].shape[0]

A =np.array([[1,0],[0,1]])
b = np.array([[0],[0]])
theta = np.matmul(A,b)
user_context['matrix_A'] = [A]*user_context['quality'].shape[0]
user_context['matrix_b'] = [b]*user_context['quality'].shape[0]
user_context['matrix_theta'] = [theta]*user_context['quality'].shape[0]
user_context['LinUCB_p'] = [0.0]*user_context['quality'].shape[0]
user_context['UCB_total_quality'] = [0.0]*user_context['quality'].shape[0]
user_context['UCB_total_count'] = [1]*user_context['quality'].shape[0]

LinUCB_alpha = 0.7
for idx, x in review_context.iterrows():
    user_id = x['user_id']
    fan_norm = user_context.loc[user_context['user_id'] == user_id, 'fan_norm']
    review_context.ix[idx,'quality'] = review_context.ix[idx,'quality']*fan_norm.values
    
review_context['mar_quality'] = review_context['quality']

# np.corrcoef(user_context['ave_quality'],user_context['fans'])
# date_min = review_context['date'].iloc[0]
# start_date = date_min + timedelta(days=-1);
# end_date = start_date + timedelta(days=365)
# review_cur = review_context.loc[ (review_context['date']>start_date) & (review_context['date'] <= end_date)]
user_context['p_elite_idx'] = [0]*user_context['quality'].shape[0]
user_context['p_fans_idx'] = [0]*user_context['quality'].shape[0]
context_space = get_context_space.get_context_space()

for idx, x in user_context.iterrows():
    elite_value = x['elite']
    fans_value = x['fans']
    for iidx, ix in context_space.iterrows():
        if elite_value in ix['elite']:
            elite_idx = iidx
        if fans_value in ix['fans']:
            fans_idx = iidx   
    user_context.ix[idx, 'p_elite_idx'] = elite_idx
    user_context.ix[idx, 'p_fans_idx'] = fans_idx

arrived_arm_num = 0
'''
(counters,quality) = cq.get_counter_quality(review_context,user_context,context_space)
oracle_quality_est = np.nan_to_num(quality/counters)
'''

B = 10
T = 200
t = 0
D= 2
alpha = 1
par=2*alpha/(3*alpha+D)/3

dimension = context_space.count()
counters = np.full((dimension.values),0.0)
quality = np.full((dimension.values),0.0)   
p_est_quality = np.full((dimension.values),0.0)

sys_reward = [0.0]*T
cum_sys_reward = [0.0]*T

sys_reward_rand = [0.0]*T
cum_sys_reward_rand = [0.0]*T

sys_reward_orac = [0.0]*T
cum_sys_reward_orac = [0.0]*T

sys_reward_nonsub = [0.0]*T
cum_sys_reward_nonsub = [0.0]*T
    
sys_reward_LinUCB = [0.0]*T
cum_sys_reward_LinUCB = [0.0]*T

sys_reward_UCB = [0.0]*T
cum_sys_reward_UCB = [0.0]*T

while (t < T):
    K = (t+1)**(par)
    review_context_cur = review_context.iloc[arrived_arm_num : arrived_arm_num + 100]
    review_context_cur_copy = review_context_cur.copy()
    user_id_unique = np.unique(review_context_cur['user_id'])
    business_id_unique = np.unique(review_context_cur['business_id'])
    user_context_cur = user_context.loc[ user_context['user_id'].isin(user_id_unique) ]
    business_context_cur = business_context.loc[ business_context['business_id'].isin(business_id_unique)]

    selected_review =  pd.DataFrame()
    selected_review_explore =  pd.DataFrame() 
    selected_review_exploit =  pd.DataFrame() 
    k = 0  
    '''    
    for idx, x in user_context_cur.iterrows():
        elite_value = x['elite']
        fans_value = x['fans']
        for iidx, ix in context_space.iterrows():
            if elite_value in ix['elite']:
                elite_idx = iidx
            if fans_value in ix['fans']:
                fans_idx = iidx   
        user_context_cur.ix[idx, 'p_elite_idx'] = elite_idx
        user_context_cur.ix[idx, 'p_fans_idx'] = fans_idx
    '''    
    ## explore
    for idx, x in review_context_cur.iterrows():
        if k < B:
            user_id = x['user_id'] 
            elite_idx = int(user_context_cur.loc[user_context_cur['user_id'] == user_id,'p_elite_idx'].values)
            fans_idx = int(user_context_cur.loc[user_context_cur['user_id'] == user_id,'p_fans_idx'].values)     
            if (counters[elite_idx,fans_idx] < K):
                 selected_review_explore = selected_review_explore.append(x)
                 review_context_cur = review_context_cur.drop(idx)
                 k = k + 1
    
    for idx, x in user_context_cur.iterrows():
        elite_idx = int(x['p_elite_idx'])
        fans_idx = int(x['p_fans_idx'])
        user_context_cur.ix[idx,'est_quality'] = p_est_quality[elite_idx,fans_idx]
                 
    ## exploit
    if k < B:
        user_context_cur['quality'] = user_context_cur['est_quality']
        selected_review_exploit = ga.greedy_solu(user_context_cur,review_context_cur, business_context_cur, B-k)
        
    selected_review = selected_review_explore.append(selected_review_exploit)
    (counter_cur,quality_cur) = cq.get_counter_quality(selected_review,user_context_cur,context_space)
    
    ## RANDDOM SELECTION
    selected_review_rand = review_context_cur_copy.sample(B, replace = False)
    
    ## Oracle 
    user_context_cur['quality'] = user_context_cur['ora_quality']
    selected_review_orac = ga.greedy_solu(user_context_cur,review_context_cur_copy, business_context_cur, B)    
    
    ## Non-submodular
    for idx, x in review_context_cur.iterrows():
        review_context_cur.ix[idx,'mar_quality'] = user_context_cur.loc[user_context_cur['user_id'] == x['user_id'],'est_quality'].values
        
    selected_review_nonsub = review_context_cur.sort_values(by=['mar_quality'])[-(B-k):]
    selected_review_nonsub = selected_review_nonsub.append(selected_review_explore)
    
    
    ##LinUCB
    for idx, x in user_context_cur.iterrows():
        x_temp = np.array([[user_context_cur.ix[idx,'elite']],[user_context_cur.ix[idx,'fans']]])
        A_temp = user_context_cur.ix[idx,'matrix_A']
        b_temp = user_context_cur.ix[idx,'matrix_b']
        theta_temp = np.matmul(inv(A_temp),b_temp)
        user_context_cur.ix[idx,'LinUCB_p'] = float(theta_temp.transpose().dot(x_temp) + LinUCB_alpha*(x_temp.transpose().dot(inv(A_temp)).dot(x_temp))**(1/2))
    
    review_context_cur_copy['LinUCB_p'] = [0.0]*review_context_cur_copy.shape[0]
    for idx, x in review_context_cur_copy.iterrows():
        review_context_cur_copy.ix[idx,'LinUCB_p'] = user_context_cur.loc[user_context_cur['user_id'] == x['user_id'],'LinUCB_p'].values
    selected_review_LinUCB = review_context_cur_copy.sort_values(by=['LinUCB_p'])[-B:]
    
    user_id_selected = np.unique(selected_review_LinUCB['user_id'])
    user_selected_LinUCB = user_context_cur.loc[user_context['user_id'].isin(user_id_selected)]
    
    
    for idx, x in user_selected_LinUCB.iterrows():
        x_temp = np.array([[x['elite']],[x['fans']]])
        A_temp = user_context.ix[idx, 'matrix_A'] +  x_temp.dot(x_temp.transpose())
        user_context.set_value(idx,'matrix_A',A_temp); 
        r_temp = np.mean(selected_review_LinUCB.loc[selected_review_LinUCB['user_id'] == x['user_id'], 'quality'].values)
        b_temp = user_context.ix[idx, 'matrix_b'] + r_temp*x_temp
        user_context.set_value(idx,'matrix_b',b_temp);

    ## UCB
    review_context_cur_copy['UCB_val'] = [0.0]*review_context_cur_copy.shape[0]
    for idx, x in review_context_cur_copy.iterrows():
        mask = user_context_cur['user_id'] == x['user_id']
        review_context_cur_copy.ix[idx,'UCB_val'] = np.nan_to_num(user_context_cur.loc[mask,'UCB_total_quality'].values/user_context_cur.loc[mask,'UCB_total_count'].values) + (np.nan_to_num(np.log(t+1)/user_context_cur.loc[mask,'UCB_total_count'].values))**(1/2)
    selected_review_UCB = review_context_cur_copy.sort_values(by=['UCB_val'])[-B:] 
    
    for idx, x in selected_review_UCB.iterrows():
         user_idx = user_context_cur.index[user_context_cur['user_id'] == x['user_id']].values
         user_context.ix[user_idx,'UCB_total_quality'] = user_context.ix[user_idx,'UCB_total_quality'] + x['quality']
         user_context.ix[user_idx,'UCB_total_count'] = user_context.ix[user_idx,'UCB_total_count'] + 1
    
    sys_reward[t] = system_reward.get_reward(selected_review)
    cum_sys_reward[t] = sum(sys_reward[0:t+1])
    
    sys_reward_rand[t] = system_reward.get_reward(selected_review_rand)
    cum_sys_reward_rand[t] = sum(sys_reward_rand[0:t+1])
    
    sys_reward_orac[t] = system_reward.get_reward(selected_review_orac)
    cum_sys_reward_orac[t] = sum(sys_reward_orac[0:t+1])
    
    sys_reward_nonsub[t] = system_reward.get_reward(selected_review_nonsub)
    cum_sys_reward_nonsub[t] = sum(sys_reward_nonsub[0:t+1])
    
    sys_reward_LinUCB[t] = system_reward.get_reward(selected_review_LinUCB)
    cum_sys_reward_LinUCB[t] = sum(sys_reward_LinUCB[0:t+1])
    
    sys_reward_UCB[t] = system_reward.get_reward(selected_review_UCB)
    cum_sys_reward_UCB[t] = sum(sys_reward_UCB[0:t+1])
    
    
    counters = counters + counter_cur
    quality = quality + quality_cur
    p_est_quality = quality/counters   
    p_est_quality[np.isnan(p_est_quality)] = 0 
    
    print('Time slot', t, 'finished!')
    
    t = t + 1
    arrived_arm_num = arrived_arm_num + 100  
    
'''       
        for idx, x in selected_review_exploit.iterrows():
            user_id = x['user_id'] 
            elite_idx = [context_space['elite'].index(i) for i in context_space['elite'] if user_context_cur.loc[user_context_cur['user_id'] == user_id,'elite'].isin(i)]
            fan_idx = [context_space['fans'].index(i) for i in context_space['fans'] if user_context_cur.loc[user_context_cur['user_id'] == user_id,'fans'].isin(i)]
            counters_exploit[elite_idx, fan_idx] += 1
'''        
        
#import dill                            #pip install dill --user
#filename = 'globalsave.pkl'
#dill.dump_session(filename)

'''
filename = 'cum_system_reward_B50'
scipy.io.savemat(filename,{'cum_sys_reward':cum_sys_reward,\
                                               'cum_sys_reward_LinUCB':cum_sys_reward_LinUCB,\
                                               'cum_sys_reward_UCB':cum_sys_reward_UCB,\
                                               'cum_sys_reward_orac':cum_sys_reward_orac,\
                                               'cum_sys_reward_nonsub':cum_sys_reward_nonsub,\
                                               'cum_sys_reward_rand':cum_sys_reward_rand})    
    
    
scipy.io.savemat('user_context',{'fans':user_context['fans'].values,\
                                 'elite':user_context['elite'].values,\
                                 'quality':user_context['quality'].values})      
scipy.io.savemat('oracle_quality_est',{'oracle_quality_est':oracle_quality_est})       
'''