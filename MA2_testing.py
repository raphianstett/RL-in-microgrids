import matplotlib.pyplot as plt
import numpy as np
from learning import Baseline
import os
from collections import Counter
import matplotlib.transforms
from data import Data
from MA_data import Data_2

# import single agent model
from learning import QLearning as QLearning
from environment import MDP as sMDP
from Variations.learning_with_diff import QLearning as dQLearning
from Variations.environment_with_diff import MDP as sdMDP


# import original 2MARL
from MA2_learning import MA2_QLearning 
from MA2_environment import Reward
from MA2_environment import MDP
from MA2_environment import Policy
# import 2MARL with difference
from Variations.MA2_environment_diff import MDP as dMDP
from Variations.MA2_learning_diff import MA2_QLearning as MA2_dQLearning
from Variations.MA2_environment_diff import Policy as dPolicy

'''
File contains testing and training functions for the 2MARL.
Plots can be generated for:
    - pairwise comparison of SARL and 2MARL performances for agent A and B
    - comparison of the respective policy composition for agent A and B
'''

############## PLOTS #####################

def get_q(subfolder, table):
    file_path = os.path.join(subfolder, str(table) + '.csv')
    return np.genfromtxt(file_path, delimiter=',')

# function returns policies, ESS courses and cost outcomes
def get_performances(episodes):
    training_data, test_data =  Data_2.get_training_test(7, False, False)
    training_data_B, test_data_B = Data_2.split_data(Data_2.get_data_B(),7)
    training_data_A, test_data_A = Data.get_training_test(7, False, False)

    results = np.zeros((7,len(episodes)))
    subfolder_name_MA = '2MARL/Q_2MARL'
    subfolder_name_singleA = 'SARL/Q_SA_models'
    subfolder_name_singleB = 'SARL/Q_Agent_B'

    results_A_SA = np.zeros((4,len(episodes)))
    results_B_SA = np.zeros((4,len(episodes)))
    results_A_MA = np.zeros((3,len(episodes)))
    results_B_MA = np.zeros((3,len(episodes)))
    for i,n in enumerate(episodes):
        print(n)
        # # 5 bins
        mdp5 = MDP(1000, 500, 500, 200, 6000, 5,5)

        # MA Q-tables 
        QA5_MA =  get_q(subfolder_name_MA, 'Q_A5' + str(n))
        QB5_MA =  get_q(subfolder_name_MA, 'Q_B5' + str(n))

        costs_A_MA,costs_B_MA, policy_A5_MA, policy_B5_MA, battery_A5, battery_B5 = Policy.find_policies(QA5_MA,QB5_MA, test_data, mdp5, mdp5)
        results_A_MA[0,i] = np.sum(costs_A_MA)
        results_B_MA[0,i] = np.sum(costs_B_MA)
        
        # SA Q-tables
        QA5_SA =  get_q(subfolder_name_singleA, 'Q5' + str(n))
        QB5_SA =  get_q(subfolder_name_singleB, 'Q5' + str(n))

        mdp5_SA = sMDP(1000,500,500,200,6000,5,5)
        costs_A_SA, policy_A5_SA, battery_A5_SA = sMDP.find_policy(mdp5_SA, QA5_SA, test_data_A)
        costs_B_SA, policy_B5_SA, battery_B5_SA = sMDP.find_policy(mdp5_SA, QB5_SA, test_data_B)
        
        results_A_SA[0,i] = np.sum(costs_A_SA)
        results_B_SA[0,i] = np.sum(costs_B_SA)
        
        # # 7 bins
        mdp7 = MDP(1000, 500, 500, 200, 6000,7,7)

        # MA Q-tables 
        QA7_MA =  get_q(subfolder_name_MA, 'Q_A7' + str(n))
        QB7_MA =  get_q(subfolder_name_MA, 'Q_B7' + str(n))

        costs_A_MA,costs_B_MA,policy_A7_MA, policy_B7_MA,battery_A7, battery_B7 = Policy.find_policies(QA7_MA,QB7_MA, test_data, mdp7, mdp7)
        results_A_MA[1,i] = np.sum(costs_A_MA)
        results_B_MA[1,i] = np.sum(costs_B_MA)
        
        # SA q-tables
        QA7_SA =  get_q(subfolder_name_singleA, 'Q7' + str(n))
        QB7_SA =  get_q(subfolder_name_singleB, 'Q7' + str(n))

        mdp7_SA = sMDP(1000,500,500,200,6000,7,7)
        costs_A_SA, policy_A7_SA, battery_A7_SA = sMDP.find_policy(mdp7_SA, QA7_SA, test_data_A)
        costs_B_SA, policy_B7_SA, battery_B7_SA = sMDP.find_policy(mdp7_SA, QB7_SA, test_data_B)
        
        results_A_SA[1,i] = np.sum(costs_A_SA)
        results_B_SA[1,i] = np.sum(costs_B_SA)
        
        # with difference

        # MA q-tables 
        dmdp_A = dMDP(1000,500,500,200,6000)
        dmdp_B = dMDP(1000,500,500,200,6000)
        dQA_MA =  get_q(subfolder_name_MA, 'dQ_A' + str(n))
        dQB_MA =  get_q(subfolder_name_MA, 'dQ_B' + str(n))

        costs_A_dMA,costs_B_dMA,policy_Ad_MA, policy_Bd_MA,battery_Ad, battery_Bd = dPolicy.find_policies(dQA_MA, dQB_MA, test_data, dmdp_A, dmdp_B)
        results_A_MA[2,i] = np.sum(costs_A_dMA)
        results_B_MA[2,i] = np.sum(costs_B_dMA)
        
        # SA q-tables
        dQA_SA =  get_q(subfolder_name_singleA, 'dQ' + str(n))
        dQB_SA =  get_q(subfolder_name_singleB, 'dQ' + str(n))

        dmdp_SA = dMDP(1000,500,500,200,6000)
        costs_A_SA, policy_Ad_SA, battery_Ad_SA = dMDP.find_policy(dmdp_SA, dQA_SA, test_data_A)
        costs_B_SA, policy_Bd_SA, battery_Bd_SA = dMDP.find_policy(dmdp_SA, dQB_SA, test_data_B)
        
        results_A_SA[2,i] = np.sum(costs_A_SA)
        results_B_SA[2,i] = np.sum(costs_B_SA)
    
    mdpbs = sMDP(1000,500,500,200,6000,7,7)    
    baseline_costs_A, baseline_states, baseline_policy_A, baseline_bat_A = Baseline.find_baseline_policy(test_data_A, mdpbs)
    results_A_SA[3,:] = [np.sum(baseline_costs_A)]*len(episodes)
    baseline_costs_B, baseline_states, baseline_policy_B, baseline_bat_B = Baseline.find_baseline_policy(test_data_B, mdpbs)
    results_B_SA[3,:] = [np.sum(baseline_costs_B)]*len(episodes)

    policy_A = [policy_A5_SA,policy_A5_MA, policy_A7_SA,policy_A7_MA, policy_Ad_SA,policy_Ad_MA, baseline_policy_A]
    policy_B= [policy_B5_SA,policy_B5_MA, policy_B7_SA,policy_B7_MA, policy_Bd_SA, policy_Bd_MA, baseline_policy_B]
    
    return results_A_SA, results_B_SA, results_A_MA, results_B_MA, policy_A, policy_B

# plots comparison between single agents and 2MARL agents
def plot_pairwise_performance(episodes):
    training_data, test_data =  Data_2.get_training_test(7, False, False)
    training_data_B, test_data_B = Data_2.split_data(Data_2.get_data_B(),7)
    training_data_A, test_data_A = Data.get_training_test(7, False, False)

    results_A_SA, results_B_SA, results_A_MA, results_B_MA, policy_A, policy_B = get_performances(episodes)

    colors =  ["royalblue", "sandybrown", "yellowgreen"]
    markers = ['o', '^', 'x']
    labels = ["5 bins", "7 bins", "difference", 'minimal MDP']

    # PLOT A
    plt.figure()
    for r in range(3):
        plt.plot(results_A_MA[r,], color = str(colors[r]), marker = 'o', markersize = 5, label = "2MARL with " + str(labels[r]))
        plt.plot(results_A_SA[r,], color = str(colors[r]), marker = 'x', markersize = 5, label = "SARL with " + str(labels[r]))
    plt.plot(results_A_SA[3,],label = "rule-based Baseline", color = "purple", linestyle = "dashdot")
    plt.legend(fontsize = 'small')
    plt.xlabel("Number of training episodes")
    plt.ylabel("Costs")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    plt.xticks(np.arange(0,len(episodes),1), episodes)
    plt.savefig('performances_agentA.png', dpi = 300)
   
    # PLOT B    
    plt.figure()
    for r in range(3):
        plt.plot(results_B_MA[r,], color = str(colors[r]),marker = 'o', markersize = 5, label = "2MARL with " + str(labels[r]))
        plt.plot(results_B_SA[r,], color = str(colors[r]),marker = 'x', markersize = 5, label = "SARL with " + str(labels[r]))
    
    plt.plot(results_B_SA[3,], color = "purple",label = "rule-based Baseline", linestyle = "dashdot")
    plt.legend(fontsize = 'small')
    plt.xlabel("Number of training episodes")
    plt.ylabel("Costs")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    plt.xticks(np.arange(0,len(episodes),1), episodes)
    plt.savefig('plots/2MARL/performances_agentB.png', dpi = 300)

# compares policies pairwise between SARL and 2MARL agents
def plot_policies_between(agent):
    training_data, test_data =  Data_2.get_training_test(7, False, False)
    training_data_B, test_data_B = Data_2.split_data(Data_2.get_data_B(),7)
    training_data_A, test_data_A = Data.get_training_test(7, False, False)

    results_A_SA, results_B_SA, results_A_MA, results_B_MA, policy_A, policy_B = get_performances([10000])
    policy = policy_A if agent == 'A' else policy_B

    items5_SA, counts5_SA = zip(*sorted(Counter(policy[0]).items()))
    items5_MA, counts5_MA = zip(*sorted(Counter(policy[1]).items()))
    items7_SA, counts7_SA = zip(*sorted(Counter(policy[2]).items()))
    items7_MA, counts7_MA = zip(*sorted(Counter(policy[3]).items()))
    itemsd_SA, countsd_SA = zip(*sorted(Counter(policy[4]).items()))
    itemsd_MA, countsd_MA = zip(*sorted(Counter(policy[5]).items()))
    items_bs, counts_bs = zip(*sorted(Counter(policy[6]).items()))

    plt.figure(figsize=(10,5))
    plt.plot(items5_SA+items5_MA+ items7_SA + items7_MA + itemsd_SA + itemsd_MA + items_bs, [6]*len(items5_SA+items5_MA+ items7_SA + items7_MA + itemsd_SA + itemsd_MA+ items_bs), visible=False)

    trans1 = matplotlib.transforms.Affine2D().translate(-0.3,0)
    trans2 = matplotlib.transforms.Affine2D().translate(-0.2,0)
    trans3 = matplotlib.transforms.Affine2D().translate(-0.1,0)
    trans4 = matplotlib.transforms.Affine2D().translate(0,0)
    trans5 = matplotlib.transforms.Affine2D().translate(+0.1,0)
    trans6 = matplotlib.transforms.Affine2D().translate(+0.2,0)
    trans7 = matplotlib.transforms.Affine2D().translate(+0.3,0)
    
    plt.bar(items5_MA, counts5_MA, label="MARL with 5 bins", width=0.1, transform=trans1+plt.gca().transData, color = "royalblue")
    plt.bar(items5_SA, counts5_SA, label="SARL with 5 bins", width=0.1, transform=trans2+plt.gca().transData, color = "royalblue", hatch = '//', edgecolor = 'white')
    plt.bar(items7_MA, counts7_MA, label="MARL with 7 bins", width=0.1, transform=trans3+plt.gca().transData, color = "sandybrown")
    plt.bar(items7_SA, counts7_SA, label="SARL with 7 bins", width=0.1, transform=trans4+plt.gca().transData, color = "sandybrown", hatch = '//', edgecolor = 'white')
    plt.bar(itemsd_MA, countsd_MA, label="MARL with difference", width=0.1, transform=trans5+plt.gca().transData, color = "yellowgreen")
    plt.bar(itemsd_SA, countsd_SA, label="SARL with difference", width=0.1, transform=trans6+plt.gca().transData, color = "yellowgreen", hatch = '//', edgecolor = 'white')
    plt.bar(items_bs, counts_bs, label="rule-based baseline", width=0.1, transform=trans7+plt.gca().transData, color = "purple", edgecolor = 'white')
    
    plt.legend(fontsize = "small")
    plt.savefig('plots/2MARL/policies_agent_' + str(agent) + '.png', dpi = 300)



###########   FUNCTION CALLS    ####################
# plot_pairwise_performance([100,500,1000,2500,5000,10000])
# plot_policies_between('A')
# plot_policies_between('B')

# plt.show()

############ TRAINING FUNCTIONS ####################

# Agent B trained on single agent model for comparison
# already trained for [100,500,1000,2500,5000,10000]
def train_B(episodes):
    training_data, test_data = Data_2.split_data(Data_2.get_data_B(),7)
    subfolder_name = 'Q_Agent_B'
    os.makedirs(subfolder_name, exist_ok=True)
    for i,n in enumerate(episodes):
        print(n)
        # normal model with 5 bins
        mdp = sMDP(1000,500,500,200,6000,5,5)
        Q5, rewards_per_episode = QLearning.iterate(training_data,n,0.5,0.9,4, mdp)
        
        # # normal model with 7 bins
        mdp = sMDP(1000,500,500,200,6000,7,7)
        Q7, rewards_per_episode = QLearning.iterate(training_data,n,0.5,0.9,4, mdp)
        
        # model with difference
        dmdp = sdMDP(1000,500,500,200,6000)
        dQ, rewards_per_episode = dQLearning.iterate(training_data,n,0.5,0.9, dmdp)

        # # Define the file path within the subfolder
        file_path = os.path.join(subfolder_name, 'Q5' + str(n)+ '.csv')
        np.savetxt(file_path, Q5, delimiter=',', fmt='%d')

        file_path = os.path.join(subfolder_name, 'Q7' + str(n)+ '.csv')
        np.savetxt(file_path, Q7, delimiter=',', fmt='%d')
        
        file_path = os.path.join(subfolder_name, 'dQ' + str(n)+ '.csv')
        np.savetxt(file_path, dQ, delimiter=',', fmt='%d')
        
# Train 2MARL model
# already trained for [100,500,1000,2500,5000,10000]
def train_MA2(episodes):
    training_data, test_data = Data_2.get_training_test(7, False, False)
    subfolder_name = '2MARL/Q_2MARL'
    os.makedirs(subfolder_name, exist_ok=True)
    for i,n in enumerate(episodes):
        
        # # 5 bins
        mdp_A5 = MDP(1000, 500, 500, 200, 6000, 5,5)
        mdp_B5 = MDP(1000, 500, 500, 200, 6000, 5,5)
        Q_A5,Q_B5, rewards_per_episode = MA2_QLearning.iterate(training_data,n, mdp_A5, mdp_B5, 0.5, 0.9)

        # # 7 bins
        mdp_A7 = MDP(1000, 500, 500, 200, 6000, 7,7)
        mdp_B7 = MDP(1000, 500, 500, 200, 6000, 7,7)
        Q_A7,Q_B7, rewards_per_episode = MA2_QLearning.iterate(training_data,n, mdp_A7, mdp_B7, 0.5, 0.9)
        
        # # model with difference
        dmdp_A = dMDP(1000,500,500,200,6000)
        dmdp_B = dMDP(1000,500,500,200,6000)
        dQ_A, dQ_B, rewards_per_episode = MA2_dQLearning.iterate(training_data,n,dmdp_A, dmdp_B, 0.5,0.9)
        
        # Define the file path within the subfolder
        file_path_A = os.path.join(subfolder_name, 'Q_A5' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'Q_B5' + str(n)+ '.csv')
        np.savetxt(file_path_A, Q_A5, delimiter=',', fmt='%d')
        np.savetxt(file_path_B, Q_B5, delimiter=',', fmt='%d')

        file_path_A = os.path.join(subfolder_name, 'Q_A7' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'Q_B7' + str(n)+ '.csv')
        np.savetxt(file_path_A, Q_A7, delimiter=',', fmt='%d')
        np.savetxt(file_path_B, Q_B7, delimiter=',', fmt='%d')
        
        file_path_A = os.path.join(subfolder_name, 'dQ_A' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'dQ_B' + str(n)+ '.csv')
        np.savetxt(file_path_A, dQ_A, delimiter=',', fmt='%d')
        np.savetxt(file_path_B, dQ_B, delimiter=',', fmt='%d')



# Helper function for a later
def get_performances_2MARL(episodes):
    training_data, test_data =  Data_2.get_training_test(7, False, False)
    subfolder_name_MA = 'Q_2MARL'
    n = episodes[0]
    results_A_SA, results_B_SA, results_A_MA, results_B_MA, policy_A, policy_B = get_performances([n])
    costs_A5, costs_A7, costs_Ad = results_A_MA[:,0]
    baseline_costs_A = results_A_SA[3,0]
    costs_B5, costs_B7, costs_Bd = results_B_MA[:,0]
    baseline_costs_B = results_B_SA[3,0]

    # baseline without
    mdp5 = sMDP(1000,500,500,200,6000,5,5)
    bs_A = mdp5.get_total_costs(test_data[:,2] - test_data[:,0])
    bs_B = mdp5.get_total_costs(test_data[:,3] - test_data[:,1])
    

    return costs_A5 + costs_B5, costs_A7 + costs_B7, costs_Ad + costs_Bd,baseline_costs_A + baseline_costs_B, bs_A + bs_B


