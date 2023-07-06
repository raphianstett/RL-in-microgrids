from MA2_learning import MA_QLearning

from MA2_environment import Reward
from MA2_environment import MDP
from MA2_environment import Policy
from data import RealData
from MA_data import Data_2
import matplotlib.pyplot as plt
import numpy as np
from learning import Baseline
import os

# df_training, test = Data_2.get_training_test(7, True, False)

# # load data for baselines
# x, df_bs_B = Data_2.split_data(RealData.get_summer_pd(Data_2.get_data_B()), 7)

# x, df_bs_A = Data_2.split_data(RealData.get_summer_pd(RealData.get_real_data()), 7)


# mdp_A = MDP(1000, 500, 500, 200, 6000, 7,7)
# mdp_B = MDP(1000, 500, 500, 200, 6000, 7,7)

# Q_A,Q_B, rewards_per_episode = MA_QLearning.iterate(df_training,10000, mdp_A, mdp_B, 0.5, 0.9)

# # plt.plot(rewards_per_episode)
# # plt.show()

# cost_A, cost_B, policy_A, policy_B, battery_A, battery_B  = Policy.find_policies(Q_A, Q_B, test, mdp_A, mdp_B)

# plt.hist(policy_A)
# plt.show()
# plt.hist(policy_B)
# plt.show()

# baseline_rewards_A, baseline_states, baseline_policy_A, baseline_bat_A, difference= Baseline.find_baseline_policy(df_bs_A, mdp_A)
# baseline_rewards_B, baseline_states, baseline_policy_B, baseline_bat_B, difference= Baseline.find_baseline_policy(df_bs_B, mdp_B)


# print("Baseline A:        " + str(np.sum(baseline_rewards_A)))
# print("Baseline B:        " + str(np.sum(baseline_rewards_B)))
# print("total Baseline: " + str(np.sum(baseline_rewards_A) + np.sum(baseline_rewards_B)))

# print("Agent A connected: " + str(np.sum(cost_A)))
# print("Agent B connected: " + str(np.sum(cost_B)))
# print("total Q: " + str(np.sum(cost_A+ cost_B)))
# print("without battery A: " + str(mdp_A.get_total_costs(df_bs_A[:,1] - df_bs_A[:,0])))
# print("without battery B: " + str(mdp_B.get_total_costs(df_bs_B[:,1] - df_bs_B[:,0])))
# print("total without: "+ str(mdp_A.get_total_costs(df_bs_A[:,1] - df_bs_A[:,0]) + mdp_B.get_total_costs(df_bs_B[:,1] - df_bs_B[:,0])))
# plt.plot(battery_A[:240], color = "red")
# plt.plot(baseline_bat_A[:240], color = "black")
# plt.show()
# plt.plot(baseline_bat_B[:240], color = "black")
# plt.plot(battery_B[:240], color = "green")
# plt.show()


################## FUNCIONS FOR RESULT FIGURES BELOW ############
iterations = [100,500,1000,2500,5000]
########### TEST B SINGLE ################
from SA_variations.environment_with_diff import MDP as dMDP
from SA_variations.environment_without_pred import MDP as rMDP
from SA_variations.learning_without_pred import QLearning as rQLearning
from SA_variations.learning_with_diff import QLearning as dQLearning
from learning import QLearning
from environment import MDP as sMDP


def train_B(iterations):
    training_data, test_data = Data_2.split_data(Data_2.get_data_B(),7)
    
    subfolder_name = 'Q_Agent_B'
    os.makedirs(subfolder_name, exist_ok=True)
    for i,n in enumerate(iterations):
        
        # normal model with 5 bins
        mdp = sMDP(1000,500,500,200,6000,5,5)
        Q5, rewards_per_episode = QLearning.iterate(training_data,n,0.5,0.9,4, mdp)
        
        # normal model with 7 bins
        mdp = sMDP(1000,500,500,200,6000,7,7)
        Q3, rewards_per_episode = QLearning.iterate(training_data,n,0.5,0.9,4, mdp)
        
        # model with difference
        dmdp = dMDP(1000,500,500,200,6000)
        dQ, rewards_per_episode = dQLearning.iterate(training_data,n,0.5,0.9, dmdp)

         # Define the file path within the subfolder
        file_path = os.path.join(subfolder_name, 'Q5' + str(n)+ '.csv')
        np.savetxt(file_path, Q5, delimiter=',', fmt='%d')

        file_path = os.path.join(subfolder_name, 'Q7' + str(n)+ '.csv')
        np.savetxt(file_path, Q3, delimiter=',', fmt='%d')
        
        file_path = os.path.join(subfolder_name, 'dQ' + str(n)+ '.csv')
        np.savetxt(file_path, dQ, delimiter=',', fmt='%d')
        
        # file_path = os.path.join(subfolder_name, 'rQ' + str(n)+ '.csv')
        # np.savetxt(file_path, rQ, delimiter=',', fmt='%d')

########## TRAIN 2MARL MODEL ##############
from MA2_learning_diff import MA_QLearning as dMA_QLearning
from MA2_environment_diff import MDP as dMDP2
from MA2_environment_diff import Policy as dPolicy

# train MA on different models
def train_MA2(iterations):
    training_data, test_data = Data_2.get_training_test(7, False, False)
    
    subfolder_name = 'Q_2MARL'
    os.makedirs(subfolder_name, exist_ok=True)
    for i,n in enumerate(iterations):
        
        # # 5 bins
        mdp_A5 = MDP(1000, 500, 500, 200, 6000, 5,5)
        mdp_B5 = MDP(1000, 500, 500, 200, 6000, 5,5)
        Q_A5,Q_B5, rewards_per_episode = MA_QLearning.iterate(training_data,n, mdp_A5, mdp_B5, 0.5, 0.9)

        # # 7 bins
        mdp_A7 = MDP(1000, 500, 500, 200, 6000, 7,7)
        mdp_B7 = MDP(1000, 500, 500, 200, 6000, 7,7)
        Q_A7,Q_B7, rewards_per_episode = MA_QLearning.iterate(training_data,n, mdp_A7, mdp_B7, 0.5, 0.9)
        
        # model with difference
        dmdp_A = dMDP2(1000,500,500,200,6000)
        dmdp_B = dMDP2(1000,500,500,200,4000)
        dQ_A, dQ_B, rewards_per_episode = dMA_QLearning.iterate(training_data,n,dmdp_A, dmdp_B, 0.5,0.9)
        
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
        
# train_MA2([10])
# training_data, test_data = Data_2.split_data(Data_2.get_data_B(),7)
# print(max(Data_2.get_data_B()["Production"] - Data_2.get_data_B()["Consumption"]))
# print(min(Data_2.get_data_B()["Production"] - Data_2.get_data_B()["Consumption"]))

########## TEST DIFFERENCE BETWEEN BINS, PERFORMANCE AND POLICY ##################
def train_MA_bins(iterations):
    training_data, test_data = Data_2.get_training_test(7, False, False)
    
    subfolder_name = 'Q_MA_bins'
    os.makedirs(subfolder_name, exist_ok=True)
    for i,n in enumerate(iterations):
        # # 3 bins
        mdp_A3 = MDP(1000, 500, 500, 200, 6000, 3,3)
        mdp_B3 = MDP(1000, 500, 500, 200, 6000, 3,3)
        Q_A3,Q_B3, rewards_per_episode = MA_QLearning.iterate(training_data,n, mdp_A3, mdp_B3, 0.5, 0.9)
        
        # # 10 bins
        mdp_A10 = MDP(1000, 500, 500, 200, 6000, 10,10)
        mdp_B10 = MDP(1000, 500, 500, 200, 6000, 10,10)
        Q_A10,Q_B10, rewards_per_episode = MA_QLearning.iterate(training_data,n, mdp_A10, mdp_B10, 0.5, 0.9)
        
        # Define the file path within the subfolder
        file_path_A = os.path.join(subfolder_name, 'Q_A3' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'Q_B3' + str(n)+ '.csv')
        np.savetxt(file_path_A, Q_A3, delimiter=',', fmt='%d')
        np.savetxt(file_path_B, Q_B3, delimiter=',', fmt='%d')
        
        file_path_A = os.path.join(subfolder_name, 'Q_A10' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'Q_B10' + str(n)+ '.csv')
        np.savetxt(file_path_A, Q_A10, delimiter=',', fmt='%d')
        np.savetxt(file_path_B, Q_B10, delimiter=',', fmt='%d')
# train_MA_bins([10])


############## PLOTS #####################
