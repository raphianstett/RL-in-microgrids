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
from collections import Counter
import matplotlib.transforms
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
        print(n)
        # normal model with 5 bins
        # mdp = sMDP(1000,500,500,200,6000,5,5)
        # Q5, rewards_per_episode = QLearning.iterate(training_data,n,0.5,0.9,4, mdp)
        
        # # normal model with 7 bins
        # mdp = sMDP(1000,500,500,200,6000,7,7)
        # Q7, rewards_per_episode = QLearning.iterate(training_data,n,0.5,0.9,4, mdp)
        
        # model with difference
        dmdp = dMDP(1000,500,500,200,6000)
        dQ, rewards_per_episode = dQLearning.iterate(training_data,n,0.5,0.9, dmdp)

        # # Define the file path within the subfolder
        # file_path = os.path.join(subfolder_name, 'Q5' + str(n)+ '.csv')
        # np.savetxt(file_path, Q5, delimiter=',', fmt='%d')

        # file_path = os.path.join(subfolder_name, 'Q7' + str(n)+ '.csv')
        # np.savetxt(file_path, Q7, delimiter=',', fmt='%d')
        
        file_path = os.path.join(subfolder_name, 'dQ' + str(n)+ '.csv')
        np.savetxt(file_path, dQ, delimiter=',', fmt='%d')
        
        # file_path = os.path.join(subfolder_name, 'rQ' + str(n)+ '.csv')
        # np.savetxt(file_path, rQ, delimiter=',', fmt='%d')

########## TRAIN 2MARL MODEL ##############
from MA2_variations.MA2_learning_diff import MA_QLearning as dMA_QLearning
from MA2_variations.MA2_environment_diff import MDP as dMDP2
from MA2_variations.MA2_environment_diff import Policy as dPolicy

# train MA on different models
def train_MA2(iterations):
    training_data, test_data = Data_2.get_training_test(7, False, False)
    
    subfolder_name = 'Q_2MARL'
    os.makedirs(subfolder_name, exist_ok=True)
    for i,n in enumerate(iterations):
        
        # # # 5 bins
        # mdp_A5 = MDP(1000, 500, 500, 200, 6000, 5,5)
        # mdp_B5 = MDP(1000, 500, 500, 200, 6000, 5,5)
        # Q_A5,Q_B5, rewards_per_episode = MA_QLearning.iterate(training_data,n, mdp_A5, mdp_B5, 0.5, 0.9)

        # # # 7 bins
        # mdp_A7 = MDP(1000, 500, 500, 200, 6000, 7,7)
        # mdp_B7 = MDP(1000, 500, 500, 200, 6000, 7,7)
        # Q_A7,Q_B7, rewards_per_episode = MA_QLearning.iterate(training_data,n, mdp_A7, mdp_B7, 0.5, 0.9)
        
        # model with difference
        dmdp_A = dMDP2(1000,500,500,200,6000)
        dmdp_B = dMDP2(1000,500,500,200,6000)
        dQ_A, dQ_B, rewards_per_episode = dMA_QLearning.iterate(training_data,n,dmdp_A, dmdp_B, 0.5,0.9)
        
        # # Define the file path within the subfolder
        # file_path_A = os.path.join(subfolder_name, 'Q_A5' + str(n)+ '.csv')
        # file_path_B = os.path.join(subfolder_name, 'Q_B5' + str(n)+ '.csv')
        # np.savetxt(file_path_A, Q_A5, delimiter=',', fmt='%d')
        # np.savetxt(file_path_B, Q_B5, delimiter=',', fmt='%d')

        # file_path_A = os.path.join(subfolder_name, 'Q_A7' + str(n)+ '.csv')
        # file_path_B = os.path.join(subfolder_name, 'Q_B7' + str(n)+ '.csv')
        # np.savetxt(file_path_A, Q_A7, delimiter=',', fmt='%d')
        # np.savetxt(file_path_B, Q_B7, delimiter=',', fmt='%d')
        
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
    training_data_B, test_data_B = Data_2.split_data(Data_2.get_data_B(),7)
    training_data_A, test_data_A = RealData.get_training_test(7, False, False)

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
# train_MA_bins([5000])
# train_MA_bins([10])

############## PLOTS #####################

def get_q(subfolder, table):
    file_path = os.path.join(subfolder, str(table) + '.csv')
    return np.genfromtxt(file_path, delimiter=',')

from environment import MDP as sMDP
# from environment import State as sState

def plot_pairwise_performance(iterations):
    
    training_data, test_data =  Data_2.get_training_test(7, False, False)
    training_data_B, test_data_B = Data_2.split_data(Data_2.get_data_B(),7)
    training_data_A, test_data_A = RealData.get_training_test(7, False, False)

    results = np.zeros((6,len(iterations)))
    subfolder_name_MA = 'Q_2MARL'
    subfolder_name_singleA = 'Q_SA_models'
    subfolder_name_singleB = 'Q_Agent_B'
    
    MA_models = ["Q_A5", "Q_A7", "Q_B5", "Q_B7", "dQ_A", "dQ_B"]
    SA_models = ["Q5", "Q7", "dQ"]

    ab = ["A", "B"]
    results_A_SA = np.zeros((3,len(iterations)))
    results_B_SA = np.zeros((3,len(iterations)))
    results_A_MA = np.zeros((3,len(iterations)))
    results_B_MA = np.zeros((3,len(iterations)))
    print(len(results_A_SA))
    print(len(results_A_MA))
    for i,n in enumerate(iterations):
        print(n)
        # # 5 bins
        mdp5 = MDP(1000, 500, 500, 200, 6000, 5,5)

        # MA q-tables 
        
        QA5_MA =  get_q(subfolder_name_MA, 'Q_A5' + str(n))
        QB5_MA =  get_q(subfolder_name_MA, 'Q_B5' + str(n))

        costs_A_MA,costs_B_MA,actions_A5_MA, actions_B5_MA,battery_A5, battery_B5 = Policy.find_policies(QA5_MA,QB5_MA, test_data, mdp5, mdp5)
        results_A_MA[0,i] = np.sum(costs_A_MA)
        results_B_MA[0,i] = np.sum(costs_B_MA)
        
        # SA q-tables
        QA5_SA =  get_q(subfolder_name_singleA, 'Q5' + str(n))
        QB5_SA =  get_q(subfolder_name_singleB, 'Q5' + str(n))

        mdp5_SA = sMDP(1000,500,500,200,6000,5,5)
        costs_A_SA, policy_A_SA, battery_A_SA = sMDP.find_policy(mdp5_SA, QA5_SA, test_data_A)
        costs_B_SA, policy_B_SA, battery_B_SA = sMDP.find_policy(mdp5_SA, QB5_SA, test_data_B)
        
        results_A_SA[0,i] = np.sum(costs_A_SA)
        results_B_SA[0,i] = np.sum(costs_B_SA)
        
        # # 7 bins
        mdp7 = MDP(1000, 500, 500, 200, 6000,7,7)

        # MA q-tables 
        
        QA7_MA =  get_q(subfolder_name_MA, 'Q_A7' + str(n))
        QB7_MA =  get_q(subfolder_name_MA, 'Q_B7' + str(n))

        costs_A_MA,costs_B_MA,actions_A7_MA, actions_B7_MA,battery_A7, battery_B7 = Policy.find_policies(QA7_MA,QB7_MA, test_data, mdp7, mdp7)
        results_A_MA[1,i] = np.sum(costs_A_MA)
        results_B_MA[1,i] = np.sum(costs_B_MA)
        
        # SA q-tables
        QA7_SA =  get_q(subfolder_name_singleA, 'Q7' + str(n))
        QB7_SA =  get_q(subfolder_name_singleB, 'Q7' + str(n))

        mdp7_SA = sMDP(1000,500,500,200,6000,7,7)
        costs_A_SA, policy_A_SA, battery_A_SA = sMDP.find_policy(mdp7_SA, QA7_SA, test_data_A)
        costs_B_SA, policy_B_SA, battery_B_SA = sMDP.find_policy(mdp7_SA, QB7_SA, test_data_B)
        
        results_A_SA[1,i] = np.sum(costs_A_SA)
        results_B_SA[1,i] = np.sum(costs_B_SA)
        
        # with difference

        # MA q-tables 
        dmdp_A = dMDP2(1000,500,500,200,6000)
        dmdp_B = dMDP2(1000,500,500,200,6000)
        dQA_MA =  get_q(subfolder_name_MA, 'dQ_A' + str(n))
        dQB_MA =  get_q(subfolder_name_MA, 'dQ_B' + str(n))

        costs_A_MA,costs_B_MA,actions_Ad_MA, actions_Bd_MA,battery_Ad, battery_Bd = dPolicy.find_policies(dQA_MA, dQB_MA, test_data, dmdp_A, dmdp_B)
        results_A_MA[2,i] = np.sum(costs_A_MA)
        results_B_MA[2,i] = np.sum(costs_B_MA)
        
        # SA q-tables
        dQA_SA =  get_q(subfolder_name_singleA, 'dQ' + str(n))
        dQB_SA =  get_q(subfolder_name_singleB, 'dQ' + str(n))

        dmdp_SA = dMDP(1000,500,500,200,6000)
        costs_A_SA, policy_A_SA, battery_A_SA = dMDP.find_policy(dmdp_SA, dQA_SA, test_data_A)
        costs_B_SA, policy_B_SA, battery_B_SA = dMDP.find_policy(dmdp_SA, dQB_SA, test_data_B)
        
        results_A_SA[2,i] = np.sum(costs_A_SA)
        results_B_SA[2,i] = np.sum(costs_B_SA)

    
    colors =  ["sandybrown", "lightslategrey", "yellowgreen"]
    markers = ['o', '^', 'x']
    labels = ["5 bins", "7 bins", "difference"]
    # PLOT A
    plt.figure(1)
    for r in range(3):
        print(r)
        plt.plot(results_A_MA[r,], color = str(colors[r]), marker = 'o', markersize = 5, label = "2MARL with " + str(labels[r]))
        plt.plot(results_A_SA[r,], color = str(colors[r]), marker = 'x', markersize = 5, label = "SARL with " + str(labels[r]))
    mdpbs = sMDP(1000,500,500,200,6000,7,7)
    baseline_rewards, baseline_states, baseline_actions, baseline_bat, difference= Baseline.find_baseline_policy(test_data_A, mdpbs)
    plt.plot([np.sum(baseline_rewards)]*len(iterations), color = "purple", linestyle = "dashdot")
    plt.legend()
    plt.xlabel("Number of training episodes")
    plt.ylabel("Costs")
    plt.xticks(np.arange(0,len(iterations),1), iterations)
    plt.title("Performances for Agents A")
    # PLOT B    
    plt.figure(2)
    for r in range(3):
        print(r)
        plt.plot(results_B_MA[r,], color = str(colors[r]),marker = 'o', markersize = 5, label = "2MARL with " + str(labels[r]))
        plt.plot(results_B_SA[r,], color = str(colors[r]),marker = 'x', markersize = 5, label = "SARL with " + str(labels[r]))
    mdpbs = sMDP(1000,500,500,200,6000,7,7)
    baseline_rewards, baseline_states, baseline_actions, baseline_bat, difference= Baseline.find_baseline_policy(test_data_B, mdpbs)
    plt.plot([np.sum(baseline_rewards)]*len(iterations), color = "purple", linestyle = "dashdot")
    plt.legend()
    plt.xlabel("Number of training episodes")
    plt.ylabel("Costs")
    plt.xticks(np.arange(0,len(iterations),1), iterations)
    plt.title("Performances for Agents B")

def plot_bin_policies():
    training_data, test_data =  Data_2.get_training_test(7, False, False)
    subfolder_name_MA = 'Q_2MARL'
    subfolder_name_other = 'Q_MA_bins'
    # # 3 bins
    mdp3 = MDP(1000, 500, 500, 200, 6000, 3,3)
    QA3 =  get_q(subfolder_name_other, 'Q_A35000')
    QB3 =  get_q(subfolder_name_other, 'Q_B35000')
    costs_A3,costs_B3,policy_A3, policy_B3,battery_A3, battery_B3 = Policy.find_policies(QA3,QB3, test_data, mdp3, mdp3)

    # # 5 bins
    mdp5 = MDP(1000, 500, 500, 200, 6000, 5,5)
    QA5 =  get_q(subfolder_name_MA, 'Q_A55000')
    QB5 =  get_q(subfolder_name_MA, 'Q_B55000')

    costs_A5,costs_B5,policy_A5, policy_B5,battery_A5, battery_B5 = Policy.find_policies(QA5,QB5, test_data, mdp5, mdp5)
    
    # # 5 bins
    mdp7 = MDP(1000, 500, 500, 200, 6000, 7,7)
    QA7 =  get_q(subfolder_name_MA, 'Q_A75000')
    QB7 =  get_q(subfolder_name_MA, 'Q_B75000')

    costs_A7,costs_B7,policy_A7, policy_B7,battery_A7, battery_B7 = Policy.find_policies(QA7,QB7, test_data, mdp7, mdp7)
    
    # # 3 bins
    mdp10 = MDP(1000, 500, 500, 200, 6000, 10,10)
    QA10 =  get_q(subfolder_name_other, 'Q_A105000')
    QB10 =  get_q(subfolder_name_other, 'Q_B105000')
    costs_A10,costs_B10,policy_A10, policy_B10,battery_A10, battery_B10 = Policy.find_policies(QA10,QB10, test_data, mdp10, mdp10)

    # # Difference
    # MA q-tables 
    dmdp_A = dMDP2(1000,500,500,200,6000)
    dmdp_B = dMDP2(1000,500,500,200,6000)
    dQA_MA =  get_q(subfolder_name_MA, 'dQ_A5000')
    dQB_MA =  get_q(subfolder_name_MA, 'dQ_B5000')

    costs_Ad,costs_B_MA,policy_Ad, policy_Bd,battery_Ad, battery_Bd = dPolicy.find_policies(dQA_MA, dQB_MA, test_data, dmdp_A, dmdp_B)

    #fig, ax = plt.subplots()
    #plt.style.use('seaborn-deep')
    # print(len(policy_5), len(policy_r), len(policy_3), len(policy_d), len(policy_baseline))
    items5, counts5 = zip(*sorted(Counter(policy_A5).items()))
    items3, counts3 = zip(*sorted(Counter(policy_A3).items()))
    items10, counts10 = zip(*sorted(Counter(policy_A10).items()))
    items7, counts7 = zip(*sorted(Counter(policy_A7).items()))
    itemsd, countsd = zip(*sorted(Counter(policy_Ad).items()))

    plt.plot(items3+items5+ items7 + items10 + itemsd, [5]*len(items3+items5+ items7 + items10 + itemsd), visible=False)

    trans1 = matplotlib.transforms.Affine2D().translate(-0.2,0)
    trans2 = matplotlib.transforms.Affine2D().translate(-0.1,0) 
    trans3 = matplotlib.transforms.Affine2D().translate(+0.1,0)
    trans4 = matplotlib.transforms.Affine2D().translate(+0.2,0)
    ["lightcoral", "sandybrown", "yellowgreen", "lightslategrey"]
    print("7: ")
    print(items7, counts7)
    print(np.sum(costs_A7))
    print("5: ")
    print(items5, counts5)
    print(np.sum(costs_A5))
    print("10: ")
    print(items10, counts10)
    print(np.sum(costs_A10))
    print("3: ")
    print(items3, counts3)
    print(np.sum(costs_A3))
    print("difference: ")
    print(itemsd, countsd)
    print(np.sum(costs_Ad))
    plt.bar(items3, counts3, label="3 bins", width=0.1, transform=trans1+plt.gca().transData, color = "lightcoral")
    plt.bar(items5, counts5, label="5 bins", width=0.1, transform=trans2+plt.gca().transData, color = "sandybrown")
    plt.bar(items7,counts7, label = "5 bins", width = 0.1, color = "yellowgreen")
    plt.bar(items10,counts10, label = "10 bins", width = 0.1, transform=trans3+plt.gca().transData, color = "lightslategrey")
    plt.bar(itemsd,countsd, label = "with difference", width = 0.1, transform=trans4+plt.gca().transData, color = "purple")
    plt.title('Policies of different models')
    plt.legend()


# plot_pairwise_performance([100,500,1000,2500,5000,10000])
# plt.show()
train_B([10000])
# plot_bin_policies()
# plt.show()
# training_data_B, test_data_B = Data_2.split_data(Data_2.get_data_B(),7)
# training_data_A, test_data_A = RealData.get_training_test(7, False, False) 
# print(test_data_B[:100])
# print(training_data_B[:100])
# train_B([100,500,1000,2500,5000,10000])
# train_MA2([100,500,1000,2500,5000,10000])