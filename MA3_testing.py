from MA3_learning import MA_QLearning
from MA3_learning import Baseline_MA3
from learning import QLearning

from environment import State
from environment import MDP as SMDP

from MA3_environment import MDP
from MA3_environment import Policy
from MA3_environment import Reward

from data import RealData
from MA_data import Data_3
from MA_data import Data_2

import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import os
from collections import Counter
### TRAIN MODELS
from MA3_variations.MA3_learning_with_diff import MA_QLearning as dMA_QLearning
from MA3_variations.MA3_environment_with_diff import MDP as dMDP3
from MA3_variations.MA3_environment_with_diff import Policy as dPolicy

# train MA on different models
def train_MA3(iterations):
    training_data, test_data = Data_2.split_data(Data_3.get_data(), 7)
    
    subfolder_name = 'Q_3MARL'
    os.makedirs(subfolder_name, exist_ok=True)
    for i,n in enumerate(iterations):
        
        # # 3 bins
        mdp3 = MDP(1000, 500, 500, 250, 12000, 3,3)
        Q_A3,Q_B3,Q_C3, rewards_per_episode = MA_QLearning.iterate(training_data,n,mdp3)

        # # 5 bins
        mdp5 = MDP(1000, 500, 500, 250, 12000, 5,5)
        Q_A5,Q_B5,Q_C5, rewards_per_episode = MA_QLearning.iterate(training_data,n,mdp5)

        # # 7 bins
        mdp7 = MDP(1000, 500, 500, 250, 12000, 7,7)
        Q_A7,Q_B7,Q_C7, rewards_per_episode = MA_QLearning.iterate(training_data,n, mdp7)
        
        # # 10 bins
        mdp10 = MDP(1000, 500, 500, 250, 12000, 10,10)
        Q_A10,Q_B10,Q_C10, rewards_per_episode = MA_QLearning.iterate(training_data,n, mdp10)


        # model with difference
        dmdp = dMDP3(1000,500,500,250,12000)
        dQ_A, dQ_B,dQ_C, rewards_per_episode = dMA_QLearning.iterate(training_data,n,dmdp)
        
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


def plot_all(iterations, start, end):
    training_data, test_data = Data_2.split_data(Data_3.get_data(), 7)
    # iterations = [100,500,1000,2500,5000, 10000]
    # iterations = [1,2,3,4,5]
    results = np.zeros((6,len(iterations)))
    subfolder_name = 'Q_3MARL'
    abc = ["A", "B", "C"]
    
    for i,n in enumerate(iterations):
        print(n)
        # # 3 bins
        mdp3 = MDP(1000, 500, 500, 250, 12000, 3,3)
        file_path_A = os.path.join(subfolder_name, 'Q_A3_' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'Q_B3_' + str(n)+ '.csv')
        file_path_C = os.path.join(subfolder_name, 'Q_C3_' + str(n)+ '.csv')
        
        QA3 =  np.genfromtxt(file_path_A, delimiter=',')
        QB3 =  np.genfromtxt(file_path_B, delimiter=',')
        QC3 =  np.genfromtxt(file_path_C, delimiter=',')
        costs_A,costs_B,costs_C, actions_A3, actions_B3, actions_C3, battery3 = Policy.find_policies(mdp3, QA3,QB3, QC3,test_data)
        results[0,i] += np.sum(costs_A) + np.sum(costs_B) + np.sum(costs_C)
       
        # # 5 bins
        mdp5 = MDP(1000, 500, 500, 250, 12000, 5,5)
        file_path_A = os.path.join(subfolder_name, 'Q_A5_' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'Q_B5_' + str(n)+ '.csv')
        file_path_C = os.path.join(subfolder_name, 'Q_C5_' + str(n)+ '.csv')
        
        QA5 =  np.genfromtxt(file_path_A, delimiter=',')
        QB5 =  np.genfromtxt(file_path_B, delimiter=',')
        QC5 =  np.genfromtxt(file_path_C, delimiter=',')
        costs_A,costs_B,costs_C, actions_A5, actions_B5, actions_C5, battery5 = Policy.find_policies(mdp5, QA5,QB5, QC5,test_data)
        results[1,i] += np.sum(costs_A) + np.sum(costs_B) + np.sum(costs_C)

        # # 7 bins
        mdp7 = MDP(1000, 500, 500, 250, 12000, 7,7)
        file_path_A = os.path.join(subfolder_name, 'Q_A7_' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'Q_B7_' + str(n)+ '.csv')
        file_path_C = os.path.join(subfolder_name, 'Q_C7_' + str(n)+ '.csv')
        
        # QA7 =  np.genfromtxt(file_path_A, delimiter=',')
        # QB7 =  np.genfromtxt(file_path_B, delimiter=',')
        # QC7 =  np.genfromtxt(file_path_C, delimiter=',')
        # costs_A,costs_B,costs_C, actions_A7, actions_B7, actions_C7, battery7 = Policy.find_policies(mdp7, QA7,QB7, QC7,test_data)
        # results[2,i] += np.sum(costs_A) + np.sum(costs_B) + np.sum(costs_C)

        # # 10 bins
        mdp10 = MDP(1000, 500, 500, 250, 12000, 10,10)
        file_path_A = os.path.join(subfolder_name, 'Q_A10_' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'Q_B10_' + str(n)+ '.csv')
        file_path_C = os.path.join(subfolder_name, 'Q_C10_' + str(n)+ '.csv')
        
        QA10 =  np.genfromtxt(file_path_A, delimiter=',')
        QB10 =  np.genfromtxt(file_path_B, delimiter=',')
        QC10 =  np.genfromtxt(file_path_C, delimiter=',')
        costs_A,costs_B,costs_C, actions_A10, actions_B10, actions_C10, battery10 = Policy.find_policies(mdp10, QA10,QB10, QC10,test_data)
        results[2,i] += np.sum(costs_A) + np.sum(costs_B) + np.sum(costs_C)
        
         # # with difference
        dmdp = dMDP3(1000,500,500,250,12000)
        file_path_A = os.path.join(subfolder_name, 'dQ_A_' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'dQ_B_' + str(n)+ '.csv')
        file_path_C = os.path.join(subfolder_name, 'dQ_C_' + str(n)+ '.csv')
        
        dQA =  np.genfromtxt(file_path_A, delimiter=',')
        dQB =  np.genfromtxt(file_path_B, delimiter=',')
        dQC =  np.genfromtxt(file_path_C, delimiter=',')
        costs_A,costs_B,costs_C, actions_Ad, actions_Bd, actions_Cd, batteryd = dPolicy.find_policies(dmdp, dQA,dQB, dQC,test_data)
        results[3,i] += np.sum(costs_A) + np.sum(costs_B) + np.sum(costs_C)
        # Baseline
        cost_A, cost_B, cost_C, policy_Abs, policy_Bbs, policy_Cbs, batterybs = Baseline_MA3.find_baseline(test_data, mdp5)
        results[4,i] += np.sum(cost_A) + np.sum(cost_B) + np.sum(cost_C)
        bs = mdp5.get_total_costs(test_data[:,3] - test_data[:,0])
        bs += mdp5.get_total_costs(test_data[:,4] - test_data[:,1])
        bs += mdp5.get_total_costs(test_data[:,5] - test_data[:,2])
        
        results[5,i] = bs

    # plot cost results
    plt.figure(1)
    fig, ax = plt.subplots()
    colors = ["lightcoral", "sandybrown","lightslategrey","yellowgreen"]
    
    labels = ["MDP with 3 bins", "MDP with 5 bins","MDP with 10 bins", "MDP with difference"]
    markers = ['^','s','o','x']

    for r in range(4):
        plt.plot(results[r,], color = str(colors[r]), label = str(labels[r]), linestyle = "solid", marker = markers[r], markersize = 5)
    
    plt.plot(results[4,], label ="rule-based Baseline", color = "purple", linestyle = "dashdot")
    
    plt.plot(results[5,], label ="Baseline without ESS", color = "grey", linestyle = "dashed")
    
    plt.xticks(np.arange(0,len(iterations),1),labels =  iterations)
    # plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlabel('Number of training episodes')
    ax.set_ylabel('Costs')
    plt.title('Effect of MDP on RL performance')
    ax.legend()
    plt.savefig("state_spaces_3MARL.png", dpi = 300)
    #plt.show()
    
    # plot battery states

    plt.figure(2)
    fig, ax = plt.subplots()
    batteries = [battery3, battery5, battery10, batteryd]
    
    for i,b in enumerate(batteries):
        
        ax = fig.add_subplot(2,2,i+1)
        ax.plot(b[start:end], color = colors[i])
        ax.set_title(labels[i])
        ax.plot(batterybs[start:end], color = "darkgrey", linestyle = "dashdot")
        ax.set_xlabel('Days')
        ax.set_ylabel('State of battery')
        ax.set_xticks(np.arange(12,180,24), np.arange(1,8,1))
        
    plt.suptitle('Battery states for different Models')    
    plt.savefig("batteries_3MARL.png", dpi = 300)
    plt.tight_layout()
    
    # # plot policies
    

    policies_3 = [actions_A3, actions_B3, actions_C3]
    policies_5 = [actions_A5, actions_B5, actions_C5]
    policies_10 = [actions_A10, actions_B10, actions_C10]
    policies_d = [actions_Ad, actions_Bd, actions_Cd]
    policies_bs = [policy_Abs, policy_Bbs, policy_Cbs]
    abc = ['A', 'B', 'C']

    for i,x in enumerate(abc):
        fig, ax = plt.figure(i+3)
        items3, counts3 = zip(*sorted(Counter(policies_3[i]).items()))
        items5, counts5 = zip(*sorted(Counter(policies_5[i]).items()))
        items10, counts10 = zip(*sorted(Counter(policies_10[i]).items()))
        itemsd, countsd = zip(*sorted(Counter(policies_d[i]).items()))
        itemsbl, countsbl = zip(*sorted(Counter(policies_bs[i]).items()))
        
        ax.plot(items3+items5+ items10 + itemsd + itemsbl, [6]*len(items3+items5 + items10 + itemsd + itemsbl), visible=False)

        trans1 = matplotlib.transforms.Affine2D().translate(-0.2,0)
        trans2 = matplotlib.transforms.Affine2D().translate(-0.1,0) 
        trans3 = matplotlib.transforms.Affine2D().translate(+0.1,0)
        trans4 = matplotlib.transforms.Affine2D().translate(+0.2,0)
        # ["lightcoral", "sandybrown","lightslategrey", "yellowgreen"]
        # print(items5, counts5)
        ax.bar(items3, counts3, label="MDP with 3 bins", width=0.1, transform=trans1+plt.gca().transData, color = "lightcoral")
        ax.bar(items5, counts5, label="MDP with 5 bins", width=0.1, transform=trans2+plt.gca().transData, color = "sandybrown")
        ax.bar(items10, counts10, label="MDP with 10 bins", width=0.1, color = "lightslategrey")
        ax.bar(itemsd,countsd, label = "MDP with difference", width = 0.1, transform=trans3+plt.gca().transData, color = "yellowgreen")
        ax.bar(itemsbl,countsbl, label = "rule-based baseline", width = 0.1, transform=trans4+plt.gca().transData, color = "purple")
        ax.set_title('Policies of Agents ' + str(x) + ' of different models')
        plt.savefig("policy_" + str(x) + ".png", dpi = 300)
        plt.legend()

plot_all([100,500,1000,2500,5000,10000], 0, 186)
plt.show()