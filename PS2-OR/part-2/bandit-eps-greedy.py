#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:47:52 2017

@author: jmarnat
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# epsilon-greedy
# =============================================================================
def pull_arm(rewards, num_arm):
    p = np.random.uniform(0,1);
    if (p < rewards[num_arm]): return 1
    return 0


def get_arm(rewards_avg, eps):
    arms = len(rewards_avg)
    p = np.random.uniform(0,1);
    
    # play a random arm with p = eps :
    if (p < eps): return np.random.choice(arms)
    
    # else play the arm with the highest empirical mean with p = 1-eps :
    max_avg = np.max(rewards_avg)
    idx = np.where(rewards_avg == max_avg)[0]
    return np.random.choice(idx)



def eps_greedy(rewards, eps, steps, trials, error_graph=False, verbose=False):
    arms = len(rewards)
    if(verbose):
        print('starting...')
        print('rewards = ', str(rewards))

    rewards_total = np.zeros(arms)
    error_sum_per_step = np.zeros([trials,steps])
    for trial in range(trials):
        rewards_sum = np.zeros(arms)
        rewards_count = np.zeros(arms) + 0.00001
        rewards_avg = np.zeros(arms)
        rewards_cum = 0
        for step in range(steps):
            rewards_avg = np.divide(rewards_sum, rewards_count)
            arm = get_arm(rewards_avg, eps)
            reward = pull_arm(rewards, arm)
            rewards_sum[arm] += reward
            rewards_cum += reward
            rewards_count[arm] += 1
            
            if(error_graph):
                error = np.linalg.norm(rewards_avg - rewards)
                error_sum_per_step[trial][step] = error

        rewards_total += rewards_avg
        if(verbose):
            print('-'*80)
            print('TRIAL #', str(trial+1))
            print('rewards_sum   = ', str(np.round(rewards_sum,2)))
            print('rewards_count = ', str(np.round(rewards_count)))
            print('rewards_avg   = ', str(np.round(rewards_avg,2)))
            print('rewards_cum   = ', str(rewards_cum))

    final_rewards = np.round(np.divide(rewards_total, trials),2)
    best_arm = np.argmax(final_rewards)
    
    if(verbose):
        print('-'*80)
        print('done!')
        print('machine rewards = ', str(rewards))
        print('rewards found   = ',str(final_rewards))
        print('best arm is     = ', str(best_arm))
    
    if(error_graph):
        plt.figure()
        for trial in range(trials):
            plt.plot(error_sum_per_step[trial],':')
        plt.title('Bandit: eps-Greedy - trials='+str(trials)+', steps='+str(steps)+'\n'+
            'rewards = '+str(rewards)+' \n '+
            'results = '+str(np.round(final_rewards,2)), fontsize=7)
        plt.savefig('images/bandit_eps_greedy'+str(n)+'-steps='+str(steps)+'-tr='+str(trials)+'.png',dpi=200)
    
    return(best_arm)

n=1
rewards = [0.2,0.3,0.8]
eps_greedy(rewards, eps = 0.1, steps =  100, trials =  20, error_graph=True)
eps_greedy(rewards, eps = 0.1, steps =  100, trials = 100, error_graph=True)
eps_greedy(rewards, eps = 0.1, steps = 2000, trials =  20, error_graph=True)
eps_greedy(rewards, eps = 0.1, steps = 2000, trials = 100, error_graph=True)


n=2
rewards = [0.1,0.1,0.1,0.2,0.4,0.6,0.7,0.8,0.8,0.9]
eps_greedy(rewards, eps=0.1, steps= 100, trials= 20, error_graph=True, verbose=False)
eps_greedy(rewards, eps=0.1, steps= 100, trials=100, error_graph=True, verbose=False)
eps_greedy(rewards, eps=0.1, steps=2000, trials= 20, error_graph=True, verbose=False)
eps_greedy(rewards, eps=0.1, steps=2000, trials=100, error_graph=True, verbose=False)


