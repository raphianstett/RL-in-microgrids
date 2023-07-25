import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import os
from collections import Counter

# import original model
from MA3_learning import MA3_QLearning
from MA3_learning import Baseline_MA3

from MA3_environment import MDP
from MA3_environment import Policy

# import MDP with difference model
from Variations.MA3_environment_with_diff import Policy as dPolicy
from Variations.MA3_environment_with_diff import MDP as dMDP
from Variations.MA3_learning_with_diff import MA3_QLearning as dMA3_QLearning

# import data
from MA_data import Data_3
from MA_data import Data_2

'''
This file contains functions to generate the plots from the thesis and functions to train the models. As previously, some Q-tables for 
the iterations [100,500,1000,2500,5000, 10000] are already trained and stored in '3MARL/Q_3MARL'.
- train_MA3(): trains models with given list of training episodes
- get_performances(): helper function to return policy, ESS courses, cost outcomes
Plot functions:
- plot_total_performance()
- plot_battery_courses()
- plot_conflicts()
- plot_single_performances()
- plot_policies()

exemplary function calls are given below
'''

# function train MA on different models
# pre-trained for episodes = [100,500,1000,2500,5000, 10000]
def train_MA3(episodes):
    training_data, test_data = Data_2.split_data(Data_3.get_data(), 7)
    
    subfolder_name = 'Q_3MARL'
    os.makedirs(subfolder_name, exist_ok=True)
    for i,n in enumerate(episodes):
        
        # # 3 bins
        mdp3 = MDP(1000, 500, 500, 250, 12000, 3,3)
        Q_A3,Q_B3,Q_C3, rewards_per_episode = MA3_QLearning.iterate(training_data,n,mdp3)

        # # 5 bins
        mdp5 = MDP(1000, 500, 500, 250, 12000, 5,5)
        Q_A5,Q_B5,Q_C5, rewards_per_episode = MA3_QLearning.iterate(training_data,n,mdp5)

        # # 7 bins
        mdp7 = MDP(1000, 500, 500, 250, 12000, 7,7)
        Q_A7,Q_B7,Q_C7, rewards_per_episode = MA3_QLearning.iterate(training_data,n, mdp7)
        
        # # 10 bins
        mdp10 = MDP(1000, 500, 500, 250, 12000, 10,10)
        Q_A10,Q_B10,Q_C10, rewards_per_episode = MA3_QLearning.iterate(training_data,n, mdp10)


        # model with difference
        dmdp = dMDP(1000,500,500,250,12000)
        dQ_A, dQ_B,dQ_C, rewards_per_episode = dMA3_QLearning.iterate(training_data,n,dmdp)
        
        # # Define the file path within the subfolder
        file_path_A = os.path.join(subfolder_name, 'Q_A3_' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'Q_B3_' + str(n)+ '.csv')
        file_path_C = os.path.join(subfolder_name, 'Q_C3_' + str(n)+ '.csv')
        np.savetxt(file_path_A, Q_A3, delimiter=',', fmt='%d')
        np.savetxt(file_path_B, Q_B3, delimiter=',', fmt='%d')
        np.savetxt(file_path_C, Q_C3, delimiter=',', fmt='%d')


        file_path_A = os.path.join(subfolder_name, 'Q_A5_' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'Q_B5_' + str(n)+ '.csv')
        file_path_C = os.path.join(subfolder_name, 'Q_C5_' + str(n)+ '.csv')
        np.savetxt(file_path_A, Q_A5, delimiter=',', fmt='%d')
        np.savetxt(file_path_B, Q_B5, delimiter=',', fmt='%d')
        np.savetxt(file_path_C, Q_C5, delimiter=',', fmt='%d')

        file_path_A = os.path.join(subfolder_name, 'Q_A7_' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'Q_B7_' + str(n)+ '.csv')
        file_path_C = os.path.join(subfolder_name, 'Q_C7_' + str(n)+ '.csv')        
        np.savetxt(file_path_A, Q_A7, delimiter=',', fmt='%d')
        np.savetxt(file_path_B, Q_B7, delimiter=',', fmt='%d')
        np.savetxt(file_path_C, Q_C7, delimiter=',', fmt='%d')
        
        file_path_A = os.path.join(subfolder_name, 'Q_A10_' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'Q_B10_' + str(n)+ '.csv')
        file_path_C = os.path.join(subfolder_name, 'Q_C10_' + str(n)+ '.csv')        
        np.savetxt(file_path_A, Q_A10, delimiter=',', fmt='%d')
        np.savetxt(file_path_B, Q_B10, delimiter=',', fmt='%d')
        np.savetxt(file_path_C, Q_C10, delimiter=',', fmt='%d')
        
        file_path_A = os.path.join(subfolder_name, 'dQ_A_' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'dQ_B_' + str(n)+ '.csv')
        file_path_C = os.path.join(subfolder_name, 'dQ_C_' + str(n)+ '.csv')
        np.savetxt(file_path_A, dQ_A, delimiter=',', fmt='%d')
        np.savetxt(file_path_B, dQ_B, delimiter=',', fmt='%d')
        np.savetxt(file_path_C, dQ_C, delimiter=',', fmt='%d')



def get_performances(episodes):
    training_data, test_data = Data_2.split_data(Data_3.get_data(), 7)
    # episodes = [100,500,1000,2500,5000, 10000]
    
    results = np.zeros((7,len(episodes)))
    results_A = np.zeros((7,len(episodes)))
    results_B = np.zeros((7,len(episodes)))
    results_C = np.zeros((7,len(episodes)))
    

    confs = np.zeros((6,len(episodes)))
    subfolder_name = '3MARL/Q_3MARL'
    abc = ["A", "B", "C"]
    
    for i,n in enumerate(episodes):
        print(n)
        # # 3 bins
        mdp3 = MDP(1000, 500, 500, 250, 12000, 3,3)
        file_path_A = os.path.join(subfolder_name, 'Q_A3_' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'Q_B3_' + str(n)+ '.csv')
        file_path_C = os.path.join(subfolder_name, 'Q_C3_' + str(n)+ '.csv')
        
        QA3 =  np.genfromtxt(file_path_A, delimiter=',')
        QB3 =  np.genfromtxt(file_path_B, delimiter=',')
        QC3 =  np.genfromtxt(file_path_C, delimiter=',')
        costs_A,costs_B,costs_C, actions_A3, actions_B3, actions_C3, battery3, conflicts_3 = Policy.find_policies(mdp3, QA3,QB3, QC3,test_data)
        results[0,i] += np.sum(costs_A) + np.sum(costs_B) + np.sum(costs_C)
        results_A[0,i] += np.sum(costs_A)
        results_B[0,i] += np.sum(costs_B)
        results_C[0,i] += np.sum(costs_C)
        
        confs[0,i] = conflicts_3
       
        # # 5 bins
        mdp5 = MDP(1000, 500, 500, 250, 12000, 5,5)
        file_path_A = os.path.join(subfolder_name, 'Q_A5_' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'Q_B5_' + str(n)+ '.csv')
        file_path_C = os.path.join(subfolder_name, 'Q_C5_' + str(n)+ '.csv')
        
        QA5 =  np.genfromtxt(file_path_A, delimiter=',')
        QB5 =  np.genfromtxt(file_path_B, delimiter=',')
        QC5 =  np.genfromtxt(file_path_C, delimiter=',')
        costs_A,costs_B,costs_C, actions_A5, actions_B5, actions_C5, battery5, conflicts_5 = Policy.find_policies(mdp5, QA5,QB5, QC5,test_data)
        results[1,i] += np.sum(costs_A) + np.sum(costs_B) + np.sum(costs_C)
        confs[1,i] = conflicts_5
        results_A[1,i] += np.sum(costs_A)
        results_B[1,i] += np.sum(costs_B)
        results_C[1,i] += np.sum(costs_C)
    
        # # 7 bins
        mdp7 = MDP(1000, 500, 500, 250, 12000, 7,7)
        file_path_A = os.path.join(subfolder_name, 'Q_A7_' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'Q_B7_' + str(n)+ '.csv')
        file_path_C = os.path.join(subfolder_name, 'Q_C7_' + str(n)+ '.csv')
        
        QA7 =  np.genfromtxt(file_path_A, delimiter=',')
        QB7 =  np.genfromtxt(file_path_B, delimiter=',')
        QC7 =  np.genfromtxt(file_path_C, delimiter=',')
        costs_A,costs_B,costs_C, actions_A7, actions_B7, actions_C7, battery7, conflicts_7 = Policy.find_policies(mdp7, QA7,QB7, QC7,test_data)
        results[2,i] += np.sum(costs_A) + np.sum(costs_B) + np.sum(costs_C)
        confs[2,i] = conflicts_7
        results_A[2,i] += np.sum(costs_A)
        results_B[2,i] += np.sum(costs_B)
        results_C[2,i] += np.sum(costs_C)
        

        # # 10 bins
        mdp10 = MDP(1000, 500, 500, 250, 12000, 10,10)
        file_path_A = os.path.join(subfolder_name, 'Q_A10_' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'Q_B10_' + str(n)+ '.csv')
        file_path_C = os.path.join(subfolder_name, 'Q_C10_' + str(n)+ '.csv')
        
        QA10 =  np.genfromtxt(file_path_A, delimiter=',')
        QB10 =  np.genfromtxt(file_path_B, delimiter=',')
        QC10 =  np.genfromtxt(file_path_C, delimiter=',')
        costs_A,costs_B,costs_C, actions_A10, actions_B10, actions_C10, battery10, conflicts_10 = Policy.find_policies(mdp10, QA10,QB10, QC10,test_data)
        results[3,i] += np.sum(costs_A) + np.sum(costs_B) + np.sum(costs_C)
        confs[3,i] = conflicts_10
        results_A[3,i] += np.sum(costs_A)
        results_B[3,i] += np.sum(costs_B)
        results_C[3,i] += np.sum(costs_C)
        
        # # with difference
        dmdp = dMDP(1000,500,500,250,12000)
        file_path_A = os.path.join(subfolder_name, 'dQ_A_' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'dQ_B_' + str(n)+ '.csv')
        file_path_C = os.path.join(subfolder_name, 'dQ_C_' + str(n)+ '.csv')
        
        dQA =  np.genfromtxt(file_path_A, delimiter=',')
        dQB =  np.genfromtxt(file_path_B, delimiter=',')
        dQC =  np.genfromtxt(file_path_C, delimiter=',')
        costs_A,costs_B,costs_C, actions_Ad, actions_Bd, actions_Cd, batteryd, conflicts_d = dPolicy.find_policies(dmdp, dQA,dQB, dQC,test_data)
        results[4,i] += np.sum(costs_A) + np.sum(costs_B) + np.sum(costs_C)
        confs[4,i] = conflicts_d
        results_A[4,i] += np.sum(costs_A)
        results_B[4,i] += np.sum(costs_B)
        results_C[4,i] += np.sum(costs_C)
        
        # Baseline
        costs_A, costs_B, costs_C, policy_Abs, policy_Bbs, policy_Cbs, batterybs, conflicts_bs = Baseline_MA3.find_baseline(test_data, mdp5)
        results[5,i] += np.sum(costs_A) + np.sum(costs_B) + np.sum(costs_C)
        confs[5,i] = conflicts_bs
        results_A[5,i] += np.sum(costs_A)
        results_B[5,i] += np.sum(costs_B)
        results_C[5,i] += np.sum(costs_C)
        
        bs = mdp5.get_total_costs(test_data[:,3] - test_data[:,0])
        results_A[6,i]= mdp5.get_total_costs(test_data[:,3] - test_data[:,0])
        bs += mdp5.get_total_costs(test_data[:,4] - test_data[:,1])
        results_B[6,i]= mdp5.get_total_costs(test_data[:,4] - test_data[:,1])
        
        bs += mdp5.get_total_costs(test_data[:,5] - test_data[:,2])
        results_C[6,i]= mdp5.get_total_costs(test_data[:,5] - test_data[:,2])
        
        results[6,i] = bs
        
        batteries = [battery3, battery5, battery10, batteryd]
        policies_3 = [actions_A3, actions_B3, actions_C3]
        policies_5 = [actions_A5, actions_B5, actions_C5]
        policies_7 = [actions_A7, actions_B7, actions_C7]
        policies_10 = [actions_A10, actions_B10, actions_C10]
        policies_d = [actions_Ad, actions_Bd, actions_Cd]
        policies_bs = [policy_Abs, policy_Bbs, policy_Cbs]
    return results, results_A, results_B, results_C, batteries, batterybs, policies_3, policies_5, policies_7, policies_10, policies_d, policies_bs, confs


def plot_total_performance(episodes):
    # plot cost results
    plt.figure()
    results, results_A, results_B, results_C, \
    batteries, batterybs, policies_3, policies_5, policies_7, \
    policies_10, policies_d, policies_bs, confs = get_performances(episodes)

    colors = ["lightcoral", "sandybrown", 'royalblue',"lightslategrey","yellowgreen"]
    
    labels = ["MDP with 3 bins", "MDP with 5 bins",'MDP with 7 bins', "MDP with 10 bins", "MDP with difference"]
    markers = ['^','s','d','o','x']

    for r in range(5):
        plt.plot(results[r,], color = str(colors[r]), label = str(labels[r]), linestyle = "solid", marker = markers[r], markersize = 5)
    
    plt.plot(results[5,], label ="rule-based Baseline", color = "purple", linestyle = "dashdot")
    
    plt.plot(results[6,], label ="Baseline without ESS", color = "grey", linestyle = "dashed")
    
    plt.xticks(np.arange(0,len(episodes),1),labels =  episodes)

    plt.xlabel('Number of training episodes')
    plt.ylabel('Costs')
    plt.legend(fontsize = 'small')
    plt.savefig("plots/3MARL/state_spaces_3MARL.png", dpi = 300)


def plot_battery_courses(episodes, start, end):
    
    results, results_A, results_B, results_C, \
    batteries, batterybs, policies_3, policies_5, policies_7, \
    policies_10, policies_d, policies_bs, conf = get_performances([episodes])

    fig, axes = plt.subplots(2,2)
    
    labels = ["MDP with 3 bins", "MDP with 5 bins","MDP with 10 bins", "MDP with difference"]
    colors = ["lightcoral", "sandybrown","lightslategrey","yellowgreen"]
    
    for i,b in enumerate(batteries):
        row = i // 2
        col = i % 2
        ax = axes[row,col]
        ax.plot(b[start:end], color = colors[i])
        ax.set_title(labels[i], fontsize = 10)
        ax.plot(batterybs[start:end], color = "darkgrey", linestyle = "dashdot")
        ax.set_xlabel('Days', fontsize = 8)
        ax.set_ylabel('State of battery', fontsize = 8)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
        ax.set_xticks(np.arange(12,180,24), np.arange(1,8,1))
        
    plt.tight_layout()
    plt.savefig("plots/3MARL/batteries_3MARL_" + str(start)+ "_" + str(end) + "_.png", dpi = 300)
    
def plot_conflicts(episodes):
    results, results_A, results_B, results_C, \
    batteries, batterybs, policies_3, policies_5, policies_7, \
    policies_10, policies_d, policies_bs, confs = get_performances(episodes)
    print(confs)
    plt.figure()
    labels = ["MDP with 3 bins", "MDP with 5 bins","MDP with 7 bins","MDP with 10 bins", "MDP with difference"]
    colors = ["lightcoral", "royalblue","sandybrown","lightslategrey","yellowgreen"]
    markers = ['^','s','d','o','x']

    for i in range(5):
        print(i)
        plt.plot(confs[i,:], color = colors[i], label = labels[i], marker = markers[i], markersize = 5)
    plt.plot(confs[5,:], color = "purple",label = "rule-based baseline", linestyle = "dashdot")
    plt.xticks(np.arange(0,len(episodes),1),episodes)
    plt.ylabel('Number of conflicts')
    plt.xlabel('Number of training episodes')
    plt.legend(fontsize = 'small', ncol = 2, bbox_to_anchor = (1.0,0.4))
    plt.savefig('plots/3MARL/conflicts_in_3MARL.png', dpi = 300)
    
def plot_single_performances(episodes):
    results, results_A, results_B, results_C, \
    batteries, batterybs, policies_3, policies_5, policies_7, \
    policies_10, policies_d, policies_bs, conf = get_performances(episodes)
    
    plt.figure()
    title = ['(a)', '(b)', '(c)']
    fig = plt.figure(5, figsize=(15, 5))

    labels = ["MDP with 3 bins", "MDP with 5 bins","MDP with 7 bins","MDP with 10 bins", "MDP with difference"]
    colors = ["lightcoral", "royalblue","sandybrown","lightslategrey","yellowgreen"]
    markers = ['^','s','d','o','x']
    abc = ["A", "B", "C"]
    
    for i,x in enumerate(abc):
        ax = fig.add_subplot(1,3,i+1)
        results = results_A if x == "A" else (results_B if x == "B" else results_C)
        for r in range(5):
        
            ax.plot(results[r,], color = str(colors[r]), label = str(labels[r]), linestyle = "solid", marker = markers[r], markersize = 5)
            
        ax.plot([min(results[5,])]*len(episodes), label ="rule-based Baseline", color = "purple", linestyle = "dashdot")
        ax.plot([min(results[6,])]*len(episodes), label ="Baseline without ESS", color = "grey", linestyle = "dashed")
        ax.set_xticks(np.arange(0,len(episodes),1),labels =  episodes)
        ax.set_xlabel('Number of training episodes', fontsize = 10 )
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
        if x == 'A':
            ax.set_ylabel('Costs', fontsize = 10)
        
        ax.text(0.5, -0.2, title[i], transform=ax.transAxes, ha='center', fontsize = 14)
    plt.legend(ncol = 4, bbox_to_anchor=(0.2, 1.1))
    plt.savefig('plots/3MARL/performances_single.png')

def plot_policies(episodes):
    results, results_A, results_B, results_C, \
    batteries, batterybs, policies_3, policies_5, policies_7, \
    policies_10, policies_d, policies_bs, conf = get_performances([episodes])
    
    abc = ['A', 'B', 'C']

    for i,x in enumerate(abc):
        plt.figure()
        items3, counts3 = zip(*sorted(Counter(policies_3[i]).items()))
        items5, counts5 = zip(*sorted(Counter(policies_5[i]).items()))
        items10, counts10 = zip(*sorted(Counter(policies_10[i]).items()))
        itemsd, countsd = zip(*sorted(Counter(policies_d[i]).items()))
        itemsbl, countsbl = zip(*sorted(Counter(policies_bs[i]).items()))
        
        plt.plot(items3+items5+ items10 + itemsd + itemsbl, [6]*len(items3+items5 + items10 + itemsd + itemsbl), visible=False)

        trans1 = matplotlib.transforms.Affine2D().translate(-0.2,0)
        trans2 = matplotlib.transforms.Affine2D().translate(-0.1,0) 
        trans3 = matplotlib.transforms.Affine2D().translate(+0.1,0)
        trans4 = matplotlib.transforms.Affine2D().translate(+0.2,0)
       
        plt.bar(items3, counts3, label="MDP with 3 bins", width=0.1, transform=trans1+plt.gca().transData, color = "lightcoral")
        plt.bar(items5, counts5, label="MDP with 5 bins", width=0.1, transform=trans2+plt.gca().transData, color = "sandybrown")
        plt.bar(items10, counts10, label="MDP with 10 bins", width=0.1, color = "lightslategrey")
        plt.bar(itemsd,countsd, label = "MDP with difference", width = 0.1, transform=trans3+plt.gca().transData, color = "yellowgreen")
        plt.bar(itemsbl,countsbl, label = "rule-based baseline", width = 0.1, transform=trans4+plt.gca().transData, color = "purple")
        plt.title('Policies of Agents ' + str(x))
        plt.legend(fontsize = "small")
        plt.savefig("plots/3MARL/policy_" + str(x) + ".png", dpi = 300)
        
# train_MA3([3])
# plot_total_performance([100,500,1000])
plot_total_performance([100,500,1000,2500,5000,10000])
plot_battery_courses(10000, 0, 186)
# plot_battery_courses(10000, 700, 886)

# plot_battery_courses(10000, 1000, 1186)
# plot_battery_courses(10000, 1500, 1686)


plot_conflicts([100,500,1000,2500,5000,10000])

plot_single_performances([100,500,1000,2500,5000,10000])
# # plot_policies(10000)
# plt.show()


# helper functions for other tests
def get_performance_3MARL(episodes):
    training_data, test_data = Data_2.split_data(Data_3.get_data(), 7)
    
    subfolder_name = '3MARL/Q_3MARL'
    n = episodes[0]
    # # 5 bins
    mdp5 = MDP(1000, 500, 500, 250, 12000, 5,5)
    file_path_A = os.path.join(subfolder_name, 'Q_A5_' + str(n)+ '.csv')
    file_path_B = os.path.join(subfolder_name, 'Q_B5_' + str(n)+ '.csv')
    file_path_C = os.path.join(subfolder_name, 'Q_C5_' + str(n)+ '.csv')
    
    QA5 =  np.genfromtxt(file_path_A, delimiter=',')
    QB5 =  np.genfromtxt(file_path_B, delimiter=',')
    QC5 =  np.genfromtxt(file_path_C, delimiter=',')
    costs_A5,costs_B5,costs_C5, actions_A5, actions_B5, actions_C5, battery5, conflicts_5 = Policy.find_policies(mdp5, QA5,QB5, QC5,test_data)
    # # 7 bins
    mdp7 = MDP(1000, 500, 500, 250, 12000, 7,7)
    file_path_A = os.path.join(subfolder_name, 'Q_A7_' + str(n)+ '.csv')
    file_path_B = os.path.join(subfolder_name, 'Q_B7_' + str(n)+ '.csv')
    file_path_C = os.path.join(subfolder_name, 'Q_C7_' + str(n)+ '.csv')
    
    QA7 =  np.genfromtxt(file_path_A, delimiter=',')
    QB7 =  np.genfromtxt(file_path_B, delimiter=',')
    QC7 =  np.genfromtxt(file_path_C, delimiter=',')
    costs_A7,costs_B7,costs_C7, actions_A7, actions_B7, actions_C7, battery7, conflicts_7 = Policy.find_policies(mdp7, QA7,QB7, QC7,test_data)
     # # with difference
    dmdp = dMDP(1000,500,500,250,12000)
    file_path_A = os.path.join(subfolder_name, 'dQ_A_' + str(n)+ '.csv')
    file_path_B = os.path.join(subfolder_name, 'dQ_B_' + str(n)+ '.csv')
    file_path_C = os.path.join(subfolder_name, 'dQ_C_' + str(n)+ '.csv')
    
    dQA =  np.genfromtxt(file_path_A, delimiter=',')
    dQB =  np.genfromtxt(file_path_B, delimiter=',')
    dQC =  np.genfromtxt(file_path_C, delimiter=',')
    costs_Ad,costs_Bd,costs_Cd, actions_Ad, actions_Bd, actions_Cd, batteryd, conflicts_d = dPolicy.find_policies(dmdp, dQA,dQB, dQC,test_data)


    costs_Abl, costs_Bbl, costs_Cbl, policy_Abs, policy_Bbs, policy_Cbs, batterybs, conflicts_bs = Baseline_MA3.find_baseline(test_data, mdp5)
    bs_A = mdp5.get_total_costs(test_data[:,3] - test_data[:,0])
    
    bs_B = mdp5.get_total_costs(test_data[:,4] - test_data[:,1])
    bs_C = mdp5.get_total_costs(test_data[:,5] - test_data[:,2])

    return np.sum(costs_A5), np.sum(costs_B5), np.sum(costs_A7), np.sum(costs_B7), np.sum(costs_Ad), np.sum(costs_Bd), np.sum(costs_Abl), np.sum(costs_Bbl), np.sum(bs_A), np.sum(bs_B)


def get_performances_all(episodes):
    training_data, test_data = Data_2.split_data(Data_3.get_data(), 7)
    
    subfolder_name = '3MARL/Q_3MARL'
    n = episodes[0]
    # # 5 bins
    mdp5 = MDP(1000, 500, 500, 250, 12000, 5,5)
    file_path_A = os.path.join(subfolder_name, 'Q_A5_' + str(n)+ '.csv')
    file_path_B = os.path.join(subfolder_name, 'Q_B5_' + str(n)+ '.csv')
    file_path_C = os.path.join(subfolder_name, 'Q_C5_' + str(n)+ '.csv')
    
    QA5 =  np.genfromtxt(file_path_A, delimiter=',')
    QB5 =  np.genfromtxt(file_path_B, delimiter=',')
    QC5 =  np.genfromtxt(file_path_C, delimiter=',')
    costs_A5,costs_B5,costs_C5, actions_A5, actions_B5, actions_C5, battery5, conflicts_5 = Policy.find_policies(mdp5, QA5,QB5, QC5,test_data)
    # # 7 bins
    mdp7 = MDP(1000, 500, 500, 250, 12000, 7,7)
    file_path_A = os.path.join(subfolder_name, 'Q_A7_' + str(n)+ '.csv')
    file_path_B = os.path.join(subfolder_name, 'Q_B7_' + str(n)+ '.csv')
    file_path_C = os.path.join(subfolder_name, 'Q_C7_' + str(n)+ '.csv')
    
    QA7 =  np.genfromtxt(file_path_A, delimiter=',')
    QB7 =  np.genfromtxt(file_path_B, delimiter=',')
    QC7 =  np.genfromtxt(file_path_C, delimiter=',')
    costs_A7,costs_B7,costs_C7, actions_A7, actions_B7, actions_C7, battery7, conflicts_7 = Policy.find_policies(mdp7, QA7,QB7, QC7,test_data)
     # # with difference
    dmdp = dMDP(1000,500,500,250,12000)
    file_path_A = os.path.join(subfolder_name, 'dQ_A_' + str(n)+ '.csv')
    file_path_B = os.path.join(subfolder_name, 'dQ_B_' + str(n)+ '.csv')
    file_path_C = os.path.join(subfolder_name, 'dQ_C_' + str(n)+ '.csv')
    
    dQA =  np.genfromtxt(file_path_A, delimiter=',')
    dQB =  np.genfromtxt(file_path_B, delimiter=',')
    dQC =  np.genfromtxt(file_path_C, delimiter=',')
    costs_Ad,costs_Bd,costs_Cd, actions_Ad, actions_Bd, actions_Cd, batteryd, conflicts_d = dPolicy.find_policies(dmdp, dQA,dQB, dQC,test_data)


    costs_Abl, costs_Bbl, costs_Cbl, policy_Abs, policy_Bbs, policy_Cbs, batterybs, conflicts_bs = Baseline_MA3.find_baseline(test_data, mdp5)
    bs_A = mdp5.get_total_costs(test_data[:,3] - test_data[:,0])
    
    bs_B = mdp5.get_total_costs(test_data[:,4] - test_data[:,1])
    bs_C = mdp5.get_total_costs(test_data[:,5] - test_data[:,2])

    return np.sum(costs_A5) + np.sum(costs_B5) + np.sum(costs_C5), np.sum(costs_A7) + np.sum(costs_B7) + np.sum(costs_C7), np.sum(costs_Ad) + np.sum(costs_Bd) + np.sum(costs_Cd), np.sum(costs_Abl) + np.sum(costs_Bbl) + np.sum(costs_Cbl), np.sum(bs_A) + np.sum(bs_B) + np.sum(bs_C)

