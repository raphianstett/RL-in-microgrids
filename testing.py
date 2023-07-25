
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import FormatStrFormatter
import matplotlib.transforms
# import original RL model
from learning import QLearning
from learning import Baseline
from environment import MDP
from environment import Reward

# import MDP with difference model
from Variations.environment_with_diff import MDP as dMDP
from Variations.learning_with_diff import QLearning as dQLearning

# import MDP without prediction model
from Variations.environment_without_pred import MDP as rMDP
from Variations.learning_without_pred import QLearning as rQLearning

from data import Data

"""
Run this file to evaluate the Single Agent Q-Learning implementation.
The below aspects can be tested with the listed functions to replicate the plots from the thesis.
Most of the functions, besides from the hyperparameter analysis can be called directly 
because they access stored trained Q-tables for different models and training episodes.
The trained Q-tables can be found in the 'Q_SA_models' and 'Q_test_steps' folders.
They are only available for [100,500,1000,2500,5000,10000] as number of training episodes.
If no trained Q-tables are available, this is noted in the list.
- train_models(): was used to train different models for different number of episodes, stores generated
   Q-tables in subfolders
Hyperparameters:
    - test_lr(): tests effect of learning rate on original MDP, no Q-tables provided
    - test_gamma(): tests effect of discount rate on original MDP,no Q-tables provided
    - test_epsilons(): tests exploration-exploitation trade-off and reward convergence, no Q-tables provided
Single Agent Microgrid design-parameters:
    - test_binning(): tests effect of discretization of continuous data in original MDP
    - test_steps_baseline(): tests effecto of charging-discharging steps on performance of rule-based baseline
    - test_steps(): tests effect of charging-discharging steps on performance and policy 
General Performance:
    - test_seasons(): trains the models and returns the rate of improvement when different winter, summer or complete data is used for training and testing
    - test_performances(): compares the cost outcomes of the different models for the given training episodes
    - test_policies(): returns bar chart of the composition of the different policies for the given training episodes
    - test_batteries(): returns the courses of ESS states in [start:end] for 3 and 5 bins, with difference and without prediction
Exemplary function calls are given from line 463 on.
"""

##### HYPERPARAMETERS #####

# learning rate is tested for alpha = [0.1,0.3,0.5,0.7,0.9] and for different numbers of training episodes
# the numbers of training episodes has to be passed to the function in a list
# in thesis: [100,500,1000,2500,5000]
def test_lr(episodes):
    training_data, test_data = Data.get_training_test(7, False, False)

    lrs = [0.1,0.3,0.5,0.7,0.9]
    mdp = MDP(1000,500,500,200,6000,5,5)
    results = np.zeros((len(lrs), len(episodes)))
    for j,x in enumerate(episodes):
        for i,l in enumerate(lrs):
            print("learning rate")
            Q, rewards_per_episode = QLearning.iterate(training_data,x,l,0.9,4, mdp)
            cost, policy, battery_states = mdp.find_policy(Q, test_data)
            results[i,j] = np.sum(cost)
    
    fig, ax = plt.subplots()
    colors = ["lightcoral", "sandybrown", "yellowgreen", "lightslategrey", "royalblue"]
    labels = ["lr = 0.1", "lr = 0.3","lr = 0.5","lr = 0.7", "lr = 0.9"]
    markers = ['^','s','x','o','d']

    for r in range(len(lrs)):
        plt.plot(results[r,], color = str(colors[r]), label = str(labels[r]), linestyle = "solid", marker = markers[r], markersize = 5)
    baseline_rewards, baseline_states, baseline_policy, baseline_battery= Baseline.find_baseline_policy(test_data, mdp)
    plt.plot([np.sum(baseline_rewards)]*len(episodes), label ="rule-based Baseline", color = "purple", linestyle = "dashdot")
    bs_without = mdp.get_total_costs(test_data[:,0] - test_data[:,1])
    plt.plot([bs_without]*len(episodes), label ="Baseline without ESS", color = "grey", linestyle = "dashed")
    
    plt.xticks(np.arange(0,len(episodes),1), episodes)
    plt.xlabel('Number of training episodes')
    plt.ylabel('Costs')
    plt.legend(fontsize ="small")
    plt.savefig("plots/SARL/learning_rate.png", dpi = 300)

# similar as for the learning rate
# training episodes in thesis: [100,500,750,1000]
def test_gamma(episodes):
    training_data, test_data = Data.get_training_test(7, False, False)
    gammas = [0.1,0.3,0.5,0.7,0.9]
    mdp = MDP(1000,500,500,200,6000,5,5)
    results = np.zeros((len(gammas), len(episodes)))
    for j,x in enumerate(episodes):
        for i, g in enumerate(gammas):
            Q, rewards_per_episode = QLearning.iterate(training_data,x,0.5,g,4, mdp)
            cost, policy, battery_states = mdp.find_policy(Q, test_data)
            results[i,j] = np.sum(cost)
    
    
    colors = ["lightcoral", "sandybrown", "yellowgreen", "lightslategrey", "royalblue"]
    labels = ["gamma = 0.1", "gamma = 0.3","gamma = 0.5","gamma = 0.7", "gamma = 0.9"]
    markers = ['^','s','x','o','d']

    for r in range(len(gammas)):
        plt.plot(results[r,], color = str(colors[r]), label = str(labels[r]), linestyle = "solid", marker = markers[r], markersize = 5)
    baseline_rewards, baseline_states, baseline_policy, baseline_bat= Baseline.find_baseline_policy(test_data, mdp)
    plt.plot([np.sum(baseline_rewards)]*len(episodes), label ="rule-based Baseline", color = "purple", linestyle = "dashdot")
    bs_without = mdp.get_total_costs(test_data[:,0] - test_data[:,1])
    plt.plot([bs_without]*len(episodes), label ="Baseline without ESS", color = "grey", linestyle = "dashed")
    
    plt.xticks(np.arange(0,len(episodes),1), episodes)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.xlabel('Number of training episodes')
    plt.ylabel('Costs')
    plt.legend(fontsize = "small")
    plt.savefig("plots/SARL/Discount_rate.png", dpi = 300)

# evaluates effect of epsilon decreasing function
# creates two subplots for the policy performances and the reward convergence
def test_epsilons():
    training_data, test_data = Data.get_training_test(7, False, False)
    epsilons = [1,3,5,7,9]
    mdp = MDP(1000,500,500,200,6000,7,7)
    episodes = [100,500,750,1000,2000,5000]
    
    results = np.zeros((len(epsilons), len(episodes)))
    
    rewards = np.zeros((max(episodes), len(epsilons)))
    print(rewards.shape)
    for j,x in enumerate(episodes):
        for i, e in enumerate(epsilons):
            Q, rewards_per_episode = QLearning.iterate(training_data,x,0.5,0.9,e, mdp)
            cost, policy, battery_states = mdp.find_policy(Q, test_data)
            results[i,j] = np.sum(cost)
            if x == max(episodes):
                rewards[:,i] = rewards_per_episode
    plt.figure(1)
    
    colors = ["lightcoral", "sandybrown", "yellowgreen", "lightslategrey", "royalblue"]
    labels = ["decreasing decay = 1", "decreasing decay = 3","decreasing decay = 5","decreasing decay = 7","decreasing decay = 9"]
    markers = ['^','s','x','o', 'd']

    for r in range(len(epsilons)):
        plt.plot(results[r,], color = str(colors[r]), label = str(labels[r]), linestyle = "solid", marker = markers[r], markersize = 5)
    baseline_rewards, baseline_states, baseline_policy, baseline_bat= Baseline.find_baseline_policy(test_data, mdp)
    plt.plot([np.sum(baseline_rewards)]*len(episodes), label ="rule-based Baseline", color = "purple", linestyle = "dashdot")
    bs_without = mdp.get_total_costs(test_data[:,1] - test_data[:,0])
    plt.plot([bs_without]*len(episodes), label ="Baseline without ESS", color = "grey", linestyle = "dashed")
    
    plt.xticks(np.arange(0,len(episodes),1), episodes)
    
    plt.xlabel('Number of training episodes')
    plt.ylabel('Costs')
    
    plt.legend(loc = 'upper left', fontsize = "small")
    plt.savefig("plots/SARL/epsilon_performance.png", dpi = 300)
    plt.figure(2)
    
    for r in range(len(epsilons)):
        plt.plot(rewards[:,r], color = str(colors[r]), label = str(labels[r]), linestyle = "solid", marker = markers[r], markersize = 4)
    plt.xlabel("training episode")
    plt.ylabel('reward per episode')
    
    plt.legend(fontsize ="small")
    plt.savefig('plots/SARL/rewards_convergence.png', dpi = 300)

    


##### SINGLE AGENT MICROGRID DESIGN PARAMETERS #####

# tests effect of discretization of continuous data
# uses pre-trained Q-tables from SARL/Q_SA_models
def test_binning(episodes):
    training_data, test_data = Data.get_training_test(7, False, False)
    bins = [3,5,7,10]
    labels = ['3 bins', '5 bins', '7 bins', '10 bins']
    results = np.zeros((len(bins), len(episodes)))
    subfolder_name = 'SARL/Q_SA_models'
    for i,b in enumerate(bins):
        for j,n in enumerate(episodes):
            mdp = MDP(1000, 500, 500, 200, 6000, b, b)
            file_path = os.path.join(subfolder_name, 'Q' +str(b) +  str(n)+ '.csv')
            Q =  np.genfromtxt(file_path, delimiter=',')
            costs, policy, battery = mdp.find_policy(Q, test_data)
            results[i,j] = np.sum(costs)
    
    colors = ["lightcoral", "sandybrown", "yellowgreen", "lightslategrey"]
    markers = ['^','s','x','o']
    
    x = np.arange(len(episodes))
    plt.figure()
    for r in range(len(bins)):
        plt.plot(x, results[r,], color = str(colors[r]), label = str(labels[r]), linestyle = "solid", marker = markers[r], markersize = 5)
    mdp_baseline = MDP(1000,500,500,200,6000,7,7)
    baseline_rewards, baseline_states, baseline_policy, baseline_bat = Baseline.find_baseline_policy(test_data, mdp_baseline)
    plt.plot([np.sum(baseline_rewards)]*len(episodes), label ="rule-based Baseline", color = "purple", linestyle = "dashdot")
    bs_without = mdp.get_total_costs(test_data[:,1] - test_data[:,0])
    plt.plot([bs_without]*len(episodes), label ="Baseline without ESS", color = "grey", linestyle = "dashdot")
    plt.xlabel('Number of training episodes')
    plt.ylabel('Costs')
    plt.xticks(x, episodes)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))

    plt.legend(fontsize = "small", loc = 'lower right', ncol = 2)
    plt.savefig("plots/SARL/binning.png", dpi = 300)
    
# tests effect of charging-discharging steps on baseline performance
def test_steps_baseline():
    training_data, test_data = Data.get_training_test(7, False, False)
    charge_high_steps =    [1500, 1000, 1000, 500]
    charge_low_steps =     [1000, 500, 500, 200]
    discharge_high_steps = [1000, 1000, 500, 200]
    discharge_low_steps =  [500, 500, 200, 100]
    costs = []
    plt.figure()
    for i in range(len(charge_high_steps)):
        mdp = MDP(charge_high_steps[i], discharge_high_steps[i], charge_low_steps[i], discharge_low_steps[i], 6000, 7,7)
        baseline_rewards, baseline_states, baseline_policy, baseline_bat = Baseline.find_baseline_policy(test_data, mdp)
        costs.append(np.sum(baseline_rewards))
    plt.plot(costs, marker = 'o', markersize = 5)
    x = np.arange(0,4,1)
    labels = ["(1500,1000,1000,500)", "(1000,500,1000,500)","(1000,500,500,200)", "(500,200,200,100)"]
    plt.xticks(x, labels, rotation=10)
    plt.xlabel('Step sizes')
    plt.ylabel('Costs')
    
    plt.savefig("plots/SARL/test_steps_baseline.png", dpi = 300)

# tests effect of charging-discharging step-sizes on performance and policy composition
# function returns performance graph and policy composition bar chart
# function takes list of number of training episodes to test
def test_steps(episodes):
    training_data, test_data = Data.get_training_test(7, False, False)
    charge_high_steps =    [1500, 1000, 1000, 500]
    charge_low_steps =     [1000, 500, 500, 200]
    discharge_high_steps = [1000, 1000, 500, 200]
    discharge_low_steps =  [500, 500, 200, 100]
    
    labels = [(1500,1000,1000,500), (1000,500,1000,500),(1000,500,500,200), (500,200,200,100)]
    results = np.zeros((len(charge_high_steps), len(episodes)))
    baseline = [0]*len(charge_high_steps)
    subfolder_name = 'SARL/Q_test_steps'
    policies = [None]*len(charge_high_steps)
    plt.figure()
    for i in range(len(charge_high_steps)):
        mdp = MDP(charge_high_steps[i], discharge_high_steps[i], charge_low_steps[i], discharge_low_steps[i], 6000, 5,5)
        
        for j,x in enumerate(episodes):
            file_path = os.path.join(subfolder_name, 'Q' + str(i) + str(j)+ '.csv')
            Q =  np.genfromtxt(file_path, delimiter=',')
            cost, policy, battery_states = mdp.find_policy(Q, test_data)
            results[i,j] = np.sum(cost)
            if x == max(episodes):
                policies[i] = policy
    # generate plot
    colors = ["lightcoral", "sandybrown", "yellowgreen", "lightslategrey"]
    markers = ['^','s','x','o']
    x = np.arange(len(episodes))
    for r in range(len(charge_high_steps)):
        plt.plot(x, results[r,], color = str(colors[r]), label = str(labels[r]), linestyle = "solid", marker = markers[r], markersize = 5)

    mdp_bs = MDP(1000,500,500,200,6000,7,7)
    baseline_rewards, baseline_states, policy_baseline, baseline_bat= Baseline.find_baseline_policy(test_data, mdp_bs)
    
    plt.plot([np.sum(baseline_rewards)]*len(episodes), label ="rule-based Baseline", color = "purple", linestyle = "dashdot")
    bs_without = mdp.get_total_costs(test_data[:,1] - test_data[:,0])
    plt.plot([bs_without]*len(episodes), label ="Baseline without ESS", color = "grey", linestyle = "dashed")
    
    plt.xlabel('Number of training episodes')
    plt.ylabel('Costs')
    plt.xticks(x, episodes)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    plt.legend(fontsize = 'small', loc = 'lower right', ncol = 3)
    plt.savefig("plots/SARL/test_steps.png", dpi = 300)

    # plot policies
    plt.figure()
    items5, counts5 = zip(*sorted(Counter(policies[0]).items()))
    items3, counts3 = zip(*sorted(Counter(policies[1]).items()))
    itemsd, countsd = zip(*sorted(Counter(policies[2]).items()))
    itemsr, countsr = zip(*sorted(Counter(policies[3]).items()))
    itemsbl, countsbl = zip(*sorted(Counter(policy_baseline).items()))
    
    plt.plot(items5+items3+ itemsd + itemsr + itemsbl, [5]*len(items5+items3+ itemsd + itemsr + itemsbl), visible=False)

    trans1 = matplotlib.transforms.Affine2D().translate(-0.2,0)
    trans2 = matplotlib.transforms.Affine2D().translate(-0.1,0) 
    trans3 = matplotlib.transforms.Affine2D().translate(+0.1,0)
    trans4 = matplotlib.transforms.Affine2D().translate(+0.2,0)
    ["lightcoral", "sandybrown", "yellowgreen", "lightslategrey"]
    print(items5, counts5)
    plt.bar(items5, counts5, label = str(labels[0]), width=0.1, transform=trans1+plt.gca().transData, color = "lightcoral")
    plt.bar(items3, counts3, label = str(labels[1]), width=0.1, transform=trans2+plt.gca().transData, color = "sandybrown")
    plt.bar(itemsd,countsd, label = str(labels[2]), width = 0.1, color = "yellowgreen")
    plt.bar(itemsr,countsr, label =  str(labels[3]), width = 0.1, transform=trans3+plt.gca().transData, color = "lightslategrey")
    plt.bar(itemsbl,countsbl, label = "rule-based baseline", width = 0.1, transform=trans4+plt.gca().transData, color = "purple")
    plt.legend()
    plt.savefig('plots/SARL/policies_steps.png', dpi = 300)
    

######## GENERAL PERFORMANCE #############

def test_seasons():
    training_summer, test_summer = Data.get_training_test(7, True, False)
    training_winter, test_winter = Data.get_training_test(7, False, True)
    training, test = Data.get_training_test(7, False, False)

    mdp = MDP(1000,500,500,200,6000,7,7)
    QS, rewards_per_episode = QLearning.iterate(training_summer,1000,0.5,0.9, 4,mdp)
    print("start find policy")
    cost_s, policy_s, battery_s = mdp.find_policy(QS, test_summer)
    bs_s = mdp.get_total_costs(test_summer[:,1] - test_summer[:,0])
    QW, rewards_per_episode = QLearning.iterate(training_winter,1000,0.5,0.9,4, mdp)
    cost_w, policy_w, battery_w = mdp.find_policy(QW, test_winter)
    bs_w = mdp.get_total_costs(test_winter[:,1] - test_winter[:,0])
    Q, rewards_per_episode = QLearning.iterate(training,1000,0.5,0.9,4, mdp)
    cost, policy, battery = mdp.find_policy(Q, test)
    bs = mdp.get_total_costs(test[:,1] - test[:,0])
    return np.sum(cost_s)/len(test_summer),bs_s / len(test_summer), np.sum(cost_w)/len(test_winter),bs_w/len(test_winter),np.sum(cost) / len(test),bs/len(test)

# helper function to find policy for different models    
def get_performances(n):
    training_data, test_data = Data.get_training_test(7, False, False)
    subfolder_name = 'SARL/Q_SA_models'

    # normal model with 5 bins
    mdp = MDP(1000,500,500,200,6000,5,5)
    file_path_5 = os.path.join(subfolder_name, 'Q5' + str(n)+ '.csv')
    Q5 =  np.genfromtxt(file_path_5, delimiter=',')
    costs_5, policy_5, battery_5 = mdp.find_policy(Q5, test_data)
    
    # normal model with 3 bins
    mdp = MDP(1000,500,500,200,6000,3,3)
    file_path_3 = os.path.join(subfolder_name, 'Q3' + str(n)+ '.csv')
    Q3 =  np.genfromtxt(file_path_3, delimiter=',')
    costs_3, policy_3, battery_3 = mdp.find_policy(Q3, test_data)
    
    # model with difference
    dmdp = dMDP(1000,500,500,200,6000)
    file_path_d = os.path.join(subfolder_name, 'dQ' + str(n)+ '.csv')
    dQ =  np.genfromtxt(file_path_d, delimiter=',')
    costs_d, policy_d, battery_d = dmdp.find_policy(dQ, test_data)
    
    # model without prediciton
    rmdp = rMDP(1000,500,500,200,6000,5,5)
    file_path_r = os.path.join(subfolder_name, 'rQ' + str(n)+ '.csv')
    rQ =  np.genfromtxt(file_path_r, delimiter=',')
    costs_r, policy_r, battery_r = rmdp.find_policy(rQ, test_data)
    
    # Baselines
    baseline_rewards, baseline_states, baseline_policy, baseline_bat = Baseline.find_baseline_policy(test_data, mdp)
    bl_without = mdp.get_total_costs(test_data[:,1] - test_data[:,0])
    return np.sum(costs_5), policy_5, battery_5, np.sum(costs_3), policy_3, battery_3, \
                np.sum(costs_d), policy_d, battery_d, costs_r, policy_r, battery_r, \
                np.sum(baseline_rewards), baseline_policy, baseline_bat, bl_without


# tests effect of MDP on performance of RL agents
def test_performances(episodes):
    results = np.zeros((6,len(episodes)))
    for i,n in enumerate(episodes):
        print("state spaces: " + str(n))
        costs_5, policy_5, battery_5, \
        costs_3, policy_3, battery_3,\
        costs_d, policy_d, battery_d, \
        costs_r, policy_r, battery_r, \
        baseline_rewards, baseline_policy, baseline_bat, bl_without = get_performances(n)
        
        # normal model with 5 bins
        results[0,i] = np.sum(costs_5)
        # normal model with 3 bins
        results[1,i] = np.sum(costs_3)
        # model with difference
        results[2,i] = np.sum(costs_d)
        # model without prediciton
        results[3,i] = np.sum(costs_r)
        # Baselines
        results[4,i] = baseline_rewards
        results[5,i] = bl_without

    fig, ax = plt.subplots()
    colors = ["lightcoral", "sandybrown", "yellowgreen", "lightslategrey"]
    labels = ["MDP with 5 bins", "MDP with 3 bins","MDP with difference","MDP without prediciton"]
    markers = ['^','s','x','o']

    for r in range(4):
        plt.plot(results[r,], color = str(colors[r]), label = str(labels[r]), linestyle = "solid", marker = markers[r], markersize = 5)
    
    plt.plot(results[4,], label ="rule-based Baseline", color = "purple", linestyle = "dashdot")
    plt.plot(results[5,], label ="Baseline without ESS", color = "grey", linestyle = "dashed")
    
    plt.xticks(np.arange(0,len(episodes),1),labels =  episodes)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    ax.set_xlabel('Number of training episodes')
    ax.set_ylabel('Costs')
    ax.legend(fontsize = 'small', bbox_to_anchor = (0.28, 0.4), ncol = 2)
    plt.savefig("plots/SARL/state_spaces.png", dpi = 300)
    
# compare policies for the same Q-tables, episodes has to be one number in this case
# episodes has to be an integer
def test_policies(episodes):
    costs_5, policy_5, battery_5, \
    costs_3, policy_3, battery_3,\
    costs_d, policy_d, battery_d, \
    costs_r, policy_r, battery_r, \
    baseline_rewards, baseline_policy, baseline_bat, bl_without = get_performances(episodes)

    items5, counts5 = zip(*sorted(Counter(policy_5).items()))
    items3, counts3 = zip(*sorted(Counter(policy_3).items()))
    itemsd, countsd = zip(*sorted(Counter(policy_d).items()))
    itemsr, countsr = zip(*sorted(Counter(policy_r).items()))
    itemsbl, countsbl = zip(*sorted(Counter(baseline_policy).items()))
    plt.figure()
    plt.plot(items5+items3+ itemsd + itemsr + itemsbl, [5]*len(items5+items3+ itemsd + itemsr + itemsbl), visible=False)

    trans1 = matplotlib.transforms.Affine2D().translate(-0.2,0)
    trans2 = matplotlib.transforms.Affine2D().translate(-0.1,0) 
    trans3 = matplotlib.transforms.Affine2D().translate(+0.1,0)
    trans4 = matplotlib.transforms.Affine2D().translate(+0.2,0)
    ["lightcoral", "sandybrown", "yellowgreen", "lightslategrey"]
    print(items5, counts5)
    plt.bar(items5, counts5, label="5 bins", width=0.1, transform=trans1+plt.gca().transData, color = "lightcoral")
    plt.bar(items3, counts3, label="3 bins", width=0.1, transform=trans2+plt.gca().transData, color = "sandybrown")
    plt.bar(itemsd,countsd, label = "with difference", width = 0.1, color = "yellowgreen")
    plt.bar(itemsr,countsr, label = "without prediction", width = 0.1, transform=trans3+plt.gca().transData, color = "lightslategrey")
    plt.bar(itemsbl,countsbl, label = "rule-based baseline", width = 0.1, transform=trans4+plt.gca().transData, color = "purple")
    
    plt.legend()
    plt.savefig("plots/SARL/policies.png", dpi = 300)

# plots ESS states in interval [start:end], note end < 2016
# episodes has to be an integer
def test_batteries(episodes, start, end):
   
    costs_5, policy_5, battery_5, \
    costs_3, policy_3, battery_3,\
    costs_d, policy_d, battery_d, \
    costs_r, policy_r, battery_r, \
    baseline_rewards, baseline_policy, baseline_bat, bl_without = get_performances(episodes)
    labels = ['5 bins MDP', '3 bins MDP', 'MDP with difference', 'MDP without prediction']
    batteries = [battery_5, battery_3, battery_d, battery_r]
    colors = ["lightcoral", "sandybrown", "yellowgreen", "lightslategrey"]
    fig = plt.figure()
    
    for i,b in enumerate(batteries):
        ax = fig.add_subplot(2,2,i+1)
        ax.plot(b[start:end], color = colors[i])
        ax.set_title(labels[i], fontsize = 10)
        ax.plot(baseline_bat[start:end], color = "darkgrey", linestyle = "dashdot")
        ax.set_xlabel('Days', fontsize = 8)
        ax.set_ylabel('State of battery', fontsize = 8)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
        ax.set_xticks(np.arange(12,180,24), np.arange(1,8,1))
        
    plt.tight_layout()
    plt.savefig("plots/SARL/batteries.png", dpi = 300)
    


########### FUNCTION CALL #################

# test_lr([100,500,1000])
# test_gamma([100,500,1000])

# test_binning([100,500,1000,2500,5000,10000])
# test_steps([100,500,1000,2500,5000,10000])

# test_performances([100,500,1000,2500,5000,10000])
# test_batteries(10000, 0,168)
# test_batteries(10000, 1000,1168)
# test_batteries(10000, 1500,1668)
# test_policies(10000)


# plt.show() # has to be commented in for the plots to be displayed


############### FUNCTION TO TRAIN DIFFERENT MODELS ########################
def train_models(episodes):
    training_data, test_data = Data.get_training_test(7, False, False)
    
    subfolder_name = 'SARL/Q_SA_models'
    os.makedirs(subfolder_name, exist_ok=True)
    for i,n in enumerate(episodes):
        
        # # normal model with 3 bins
        mdp = MDP(1000,500,500,200,6000,3,3)
        Q3, rewards_per_episode = QLearning.iterate(training_data,n,0.5,0.9,4, mdp)
        
        # normal model with 5 bins
        mdp = MDP(1000,500,500,200,6000,5,5)
        Q5, rewards_per_episode = QLearning.iterate(training_data,n,0.5,0.9,4, mdp)
        
        # normal model with 7 bins
        mdp = MDP(1000,500,500,200,6000,7,7)
        Q7, rewards_per_episode = QLearning.iterate(training_data,n,0.5,0.9,4, mdp)
        
        # normal model with 10 bins
        mdp = MDP(1000,500,500,200,6000,10,10)
        Q10, rewards_per_episode = QLearning.iterate(training_data,n,0.5,0.9,4, mdp)
        
        # model with difference
        dmdp = dMDP(1000,500,500,200,6000)
        dQ, rewards_per_episode = dQLearning.iterate(training_data,n,0.5,0.9, dmdp)
        
        # # model without prediciton
        rmdp = rMDP(1000,500,500,200,6000,5,5)
        rQ, rewards_per_episode = rQLearning.iterate(training_data,n,0.5,0.9, rmdp)
        
        # Define the file path within the subfolder and store generated Q-tables as .csv
        file_path = os.path.join(subfolder_name, 'Q3' + str(n)+ '.csv')
        np.savetxt(file_path, Q3, delimiter=',', fmt='%d')

        file_path = os.path.join(subfolder_name, 'Q5' + str(n)+ '.csv')
        np.savetxt(file_path, Q5, delimiter=',', fmt='%d')

        file_path = os.path.join(subfolder_name, 'Q7' + str(n)+ '.csv')
        np.savetxt(file_path, Q7, delimiter=',', fmt='%d')

        file_path = os.path.join(subfolder_name, 'Q10' + str(n)+ '.csv')
        np.savetxt(file_path, Q10, delimiter=',', fmt='%d')
        
        file_path = os.path.join(subfolder_name, 'dQ' + str(n)+ '.csv')
        np.savetxt(file_path, dQ, delimiter=',', fmt='%d')
        
        file_path = os.path.join(subfolder_name, 'rQ' + str(n)+ '.csv')
        np.savetxt(file_path, rQ, delimiter=',', fmt='%d')
    
# function which is relevant for comparison of different microgrid environments
def get_performances_SARL(episodes, agent):
    training_data, test_data = Data.get_training_test(7, False, False)
    subfolder_name = 'SARL/Q_SA_models' if agent == 'A' else 'Q_Agent_B'
   
    # normal model with 7 bins
    mdp = MDP(1000,500,500,200,6000,7,7)
    # Q, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(training_data,n,0.5,0.9, mdp)
    file_path_7 = os.path.join(subfolder_name, 'Q7' + str(episodes)+ '.csv')
    Q7 =  np.genfromtxt(file_path_7, delimiter=',')
    costs_7, policy, battery = mdp.find_policy(Q7, test_data)
    
    costs_5, policy_5, battery_5, \
    costs_3, policy_3, battery_3,\
    costs_d, policy_d, battery_d, \
    costs_r, policy_r, battery_r, \
    baseline_rewards, baseline_policy, baseline_bat, bl_without = get_performances(episodes)
       
    return np.sum(costs_5), np.sum(costs_7), np.sum(costs_d), np.sum(baseline_rewards), np.sum(bl_without)

