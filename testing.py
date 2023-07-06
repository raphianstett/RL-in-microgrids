from learning import QLearning
from learning import Baseline

from environment import MDP
from environment import State
from SA_variations.environment_with_diff import MDP as dMDP
from SA_variations.environment_with_diff import State as dState

from data import StepFunctions
from data import RealData

import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import FormatStrFormatter


##### TESTING FOR RL EVALUATION ####
# import test data
dat_test = StepFunctions.get_test_data()

# # import real data
# data = RealData.get_real_data()

# summer_data = RealData.get_summer(data)

training_data, test_data = RealData.get_training_test(7,True,False)

# initialize MDP
# MDP(max_charge, max_discharge, charge_low, discharge_low, max_battery, bins_cons, bins_prod)

mdp_3 = MDP(1000, 1000, 500, 500, 6000, 3,3)
mdp_5 = MDP(1000, 1000, 500, 500, 6000, 5,5)
mdp_7 = MDP(1000, 500, 500, 200, 4000, 7,7)
mdp_10 = MDP(1000, 500, 500, 200, 4000, 10,10)
mdp = mdp_5
mdp_d = dMDP(1000, 500, 200, 500, 4000)

# Q_table_sol, rewards_per_episode= QLearning.iterate(training_data,1, 0.5,0.9,4,mdp)
# cost, applied_actions, battery_states = MDP.find_policy(mdp_5,Q_table_sol, test_data)
# print(np.sum(cost))
# baseline_rewards, baseline_states, baseline_actions, baseline_bat, difference= Baseline.find_baseline_policy(test_data, mdp)
# print(np.sum(baseline_rewards))
### TEST BINNING IN RL CONVERGENCE ###
def test_binning():
    training_data, test_data = RealData.get_training_test(7, False, False)
    bins = [3,5,7,10]
    # bins = [3,5]
    iterations = [100,500,1000,2500,5000, 10000]
    # iterations = [3,5,6,10]
    labels = ['3 bins', '5 bins', '7 bins', '10 bins']
    # test_iterations = [1,2]
    results = np.zeros((len(bins), len(iterations)))
    
    for i,b in enumerate(bins):
        
        for j,x in enumerate(iterations):
            print("bin testing")
            mdp = MDP(1000, 500, 500, 200, 6000, b, b)
            Q_table_sol, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(training_data,x, 0.5,0.9,mdp)
            cost, applied_actions, battery_states, dis, loss, states = mdp.find_policy(Q_table_sol, test_data)
            results[i,j] = np.sum(cost)
    
    colors = ["lightcoral", "sandybrown", "yellowgreen", "lightslategrey"]
    markers = ['^','s','x','o']
    fig, ax = plt.subplots()
    x = np.arange(len(iterations))
    fig, ax = plt.subplots()
    
    for r in range(len(bins)):
        plt.plot(x, results[r,], color = str(colors[r]), label = str(labels[r]), linestyle = "solid", marker = markers[r], markersize = 5)
    mdp_baseline = MDP(1000,500,500,200,6000,7,7)
    baseline_rewards, baseline_states, baseline_actions, baseline_bat, difference= Baseline.find_baseline_policy(test_data, mdp_baseline)
    plt.plot([np.sum(baseline_rewards)]*len(iterations), label ="rule-based Baseline", color = "purple", linestyle = "dashdot")
    bs_without = mdp.get_total_costs(test_data["Production"] - test_data["Consumption"])
    plt.plot([bs_without]*len(iterations), label ="Baseline without ESS", color = "grey", linestyle = "dashdot")
    ax.set_xlabel('Number of training episodes')
    ax.set_ylabel('Costs')
    plt.xticks(x, iterations, rotation=45)
    plt.title('Effect of binning on RL performance')
    ax.legend()
    plt.savefig("binning.png")
    # plt.show()
    

#test_binning()
### TEST CHARGING STEP SIZES ###
def test_steps():
    charge_high_steps =    [1500, 1000, 1000, 500]
    charge_low_steps =     [1000, 500, 500, 200]
    discharge_high_steps = [1000, 1000, 500, 200]
    discharge_low_steps =  [500, 500, 200, 100]
    iterations =  [100,500,750,1000,2000,5000, 10000]
    # iterations = [1,2]
    labels = [(1500,1000,1000,500), (1000,500,1000,500),(1000,500,500,200), (500,200,200,100)]
    results = np.zeros((len(charge_high_steps), len(iterations)))
    baseline = [0]*len(charge_high_steps)
    for i in range(len(charge_high_steps)):
        mdp = MDP(charge_high_steps[i], discharge_high_steps[i], charge_low_steps[i], discharge_low_steps[i], 6000, 7,7)
        
        
        for j,x in enumerate(iterations):
            print("step testing")
            Q_table_sol, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(training_data,x, 0.5,0.9, mdp)
            cost, applied_actions, battery_states, dis, loss, states = mdp.find_policy(Q_table_sol, test_data)
            results[i,j] = np.sum(cost)
    # do result plot
    colors = ["lightcoral", "sandybrown", "yellowgreen", "lightslategrey"]
    markers = ['^','s','x','o']
    fig, ax = plt.subplots()
    x = np.arange(len(iterations))
    for r in range(len(charge_high_steps)):
        plt.plot(x, results[r,], color = str(colors[r]), label = str(labels[r]), linestyle = "solid", marker = markers[r], markersize = 5)

    mdp_bs = MDP(1000,500,500,200,6000,7,7)
    baseline_rewards, baseline_states, baseline_actions, baseline_bat, difference= Baseline.find_baseline_policy(test_data, mdp_bs)
    
    plt.plot([np.sum(baseline_rewards)]*len(charge_high_steps), label ="rule-based Baseline", color = "purple", linestyle = "dashdot")
    bs_without = mdp.get_total_costs(test_data["Production"] - test_data["Consumption"])
    plt.plot([bs_without]*len(iterations), label ="Baseline without ESS", color = "grey", linestyle = "dashed")
    
    ax.set_xlabel('Number of training episodes')
    ax.set_ylabel('Costs')
    plt.xticks(x, iterations, rotation=45)
    plt.title('Effect of ESS Model step sizes on RL performance')
    ax.legend()
    plt.savefig("test_steps.png")
#test_steps()
    
##### TEST LEARNING RATE #####
# can be copied to test gamma
def test_lr():
    lrs = [0.1,0.3,0.5,0.7,0.9]
    mdp = MDP(1000,500,500,200,6000,7,7)
    iterations = [100,750,500,1000,5000, 10000]
    # iterations = [5,10,15]
    results = np.zeros((len(iterations), len(lrs)))
    for j,x in enumerate(iterations):
        for i, lr in enumerate(lrs):
            print("learning rate")
            Q, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(training_data,x,lr,0.9, mdp)
            cost, applied_actions, battery_states, dis, loss, states = mdp.find_policy(Q, test_data)
            results[j,i] = np.sum(cost)
    
    fig, ax = plt.subplots()
    colors = ["lightcoral", "sandybrown", "yellowgreen", "lightslategrey"]
    labels = ["Q-learning - 100 episodes", "Q-learning - 500 episodes","Q-learning - 1000 episodes","Q-learning - 5000 episodes"]
    markers = ['^','s','x','o']

    for r in range(len(iterations)):
        plt.plot(results[r,], color = str(colors[r]), label = str(labels[r]), linestyle = "solid", marker = markers[r], markersize = 5)
    baseline_rewards, baseline_states, baseline_actions, baseline_bat, difference= Baseline.find_baseline_policy(test_data, mdp)
    plt.plot([np.sum(baseline_rewards)]*len(lrs), label ="rule-based Baseline", color = "purple", linestyle = "dashdot")
    bs_without = mdp.get_total_costs(test_data["Production"] - test_data["Consumption"])
    plt.plot([bs_without]*len(lrs), label ="Baseline without ESS", color = "grey", linestyle = "dashdot")
    
    plt.xticks(np.arange(0,10,1), lrs)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Costs')
    plt.title('Effect of Learning rate on RL performance')
    ax.legend()
    plt.savefig("learning_rate.png")
    # plt.show()
#test_lr()

def test_gamma():
    training_data, test_data = RealData.get_training_test(7, False, False)
    gammas = [0.1,0.3,0.5,0.7,0.9]
    mdp = MDP(1000,500,500,200,6000,5,5)
    iterations = [100,750,500,1000]
    #iterations = [1,2,3,4,5,6]
    results = np.zeros((len(gammas), len(iterations)))
    for j,x in enumerate(iterations):
        for i, g in enumerate(gammas):
            print("gamma")
            Q, rewards_per_episode = QLearning.iterate(training_data,x,0.5,g,4, mdp)
            cost, applied_actions, battery_states = mdp.find_policy(Q, test_data)
            results[i,j] = np.sum(cost)
    
    fig, ax = plt.subplots()
    colors = ["lightcoral", "sandybrown", "yellowgreen", "lightslategrey", "royalblue"]
    labels = ["gamma = 0.1", "gamma = 0.3","gamma = 0.5","gamma = 0.7", "gamma = 0.9"]
    markers = ['^','s','x','o','d']

    for r in range(len(gammas)):
        plt.plot(results[r,], color = str(colors[r]), label = str(labels[r]), linestyle = "solid", marker = markers[r], markersize = 5)
    baseline_rewards, baseline_states, baseline_actions, baseline_bat, difference= Baseline.find_baseline_policy(test_data, mdp)
    plt.plot([np.sum(baseline_rewards)]*len(iterations), label ="rule-based Baseline", color = "purple", linestyle = "dashdot")
    bs_without = mdp.get_total_costs(test_data[:,0] - test_data[:,1])
    plt.plot([bs_without]*len(iterations), label ="Baseline without ESS", color = "grey", linestyle = "dashed")
    
    plt.xticks(np.arange(0,len(iterations),1), iterations)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlabel('Number of training episodes')
    ax.set_ylabel('Costs')
    plt.title('Discount rate and RL performance')
    ax.legend()
    plt.savefig("Discount_rate.png")
    # plt.show()
test_gamma()


######## TEST SEASONAL DIFFERENCES IN DATA AND RL PERFORMANCE  #############
def test_seasons():
    data = RealData.get_real_data()
    # summer_data = RealData.get_summer(RealData.get_real_data())
    # winter_data = RealData.get_winter(RealData.get_real_data())

    training_summer, test_summer = RealData.get_training_test(7, True, False)
    training_winter, test_winter = RealData.get_training_test(7, False, True)
    training, test = RealData.get_training_test(7, False, False)
    
    # use optimal values for binning, step-sizes and learning rate
    mdp = MDP(1000,500,500,200,6000,7,7)
    QS, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(training_summer,1000,0.5,0.9, 4,mdp)
    print("start find policy")
    cost_s, applied_actions, battery_s, dis, loss, states = mdp.find_policy(QS, test_summer)
    bs_s = mdp.get_total_costs(test_summer[:,1] - test_summer[:,0])
    QW, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(training_winter,1000,0.5,0.9,4, mdp)
    cost_w, applied_actions, battery_s, dis, loss, states = mdp.find_policy(QW, test_winter)
    bs_w = mdp.get_total_costs(test_winter[:,1] - test_winter[:,0])
    Q, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(training,1000,0.5,0.9,4, mdp)
    cost, applied_actions, battery_s, dis, loss, states = mdp.find_policy(Q, test)
    bs = mdp.get_total_costs(test[:,1] - test[:,0])
    return np.sum(cost_s)/len(test_summer),bs_s / len(test_summer), np.sum(cost_w)/len(test_winter),bs_w/len(test_winter),np.sum(cost) / len(test),bs/len(test)
# print(test_seasons())

############### TEST DIFFERENT STATE SPACES ########################
from SA_variations.environment_with_diff import MDP as dMDP
from SA_variations.environment_without_pred import MDP as rMDP
from SA_variations.learning_without_pred import QLearning as rQLearning
from SA_variations.learning_with_diff import QLearning as dQLearning

def train_models(iterations):
    training_data, test_data = RealData.get_training_test(7, False, False)
    # iterations = [100,500,1000,2500,5000]
   
    subfolder_name = 'Q_SA_models'
    os.makedirs(subfolder_name, exist_ok=True)
    for i,n in enumerate(iterations):
        
        # normal model with 5 bins
        mdp = MDP(1000,500,500,200,6000,5,5)
        Q5, rewards_per_episode = QLearning.iterate(training_data,n,0.5,0.9,4, mdp)
        
        # normal model with 10 bins
        mdp = MDP(1000,500,500,200,6000,3,3)
        Q3, rewards_per_episode = QLearning.iterate(training_data,n,0.5,0.9,4, mdp)
        
        # model with difference
        dmdp = dMDP(1000,500,500,200,6000)
        dQ, rewards_per_episode = dQLearning.iterate(training_data,n,0.5,0.9, dmdp)
        
        # model without prediciton
        rmdp = rMDP(1000,500,500,200,6000,5,5)
        rQ, rewards_per_episode = rQLearning.iterate(training_data,n,0.5,0.9, rmdp)
        
        # Define the file path within the subfolder
        file_path = os.path.join(subfolder_name, 'Q5' + str(n)+ '.csv')
        np.savetxt(file_path, Q5, delimiter=',', fmt='%d')

        file_path = os.path.join(subfolder_name, 'Q3' + str(n)+ '.csv')
        np.savetxt(file_path, Q3, delimiter=',', fmt='%d')
        
        file_path = os.path.join(subfolder_name, 'dQ' + str(n)+ '.csv')
        np.savetxt(file_path, dQ, delimiter=',', fmt='%d')
        
        file_path = os.path.join(subfolder_name, 'rQ' + str(n)+ '.csv')
        np.savetxt(file_path, rQ, delimiter=',', fmt='%d')
    

def test_state_spaces(iterations):
    training_data, test_data = RealData.get_training_test(7, False, False)
    # iterations = [100,500,1000,2500,5000, 10000]
    # iterations = [1,2,3,4,5]
    results = np.zeros((6,len(iterations)))
    subfolder_name = 'Q_SA_models'
    for i,n in enumerate(iterations):
        print("state spaces: " + str(n))
        # normal model with 5 bins
        mdp = MDP(1000,500,500,200,6000,5,5)
        # Q, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(training_data,n,0.5,0.9, mdp)
        file_path_5 = os.path.join(subfolder_name, 'Q5' + str(n)+ '.csv')
        Q5 =  np.genfromtxt(file_path_5, delimiter=',')
        costs_5, applied_actions, battery_s = mdp.find_policy(Q5, test_data)
        results[0,i] = np.sum(costs_5)

        # normal model with 3 bins
        mdp = MDP(1000,500,500,200,6000,3,3)
        # Q, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(training_data,n,0.5,0.9, mdp)
        file_path_3 = os.path.join(subfolder_name, 'Q3' + str(n)+ '.csv')
        Q3 =  np.genfromtxt(file_path_3, delimiter=',')
        costs_3, applied_actions, battery_s = mdp.find_policy(Q3, test_data)
        results[1,i] = np.sum(costs_3)

        # model with difference
        dmdp = dMDP(1000,500,500,200,6000)
        # dQ_table, rewards_per_episode, all_rewards, actions, states_id, states, battery = dQ.iterate(training_data,n,0.5,0.9, dmdp)
        file_path_d = os.path.join(subfolder_name, 'dQ' + str(n)+ '.csv')
        dQ =  np.genfromtxt(file_path_d, delimiter=',')
        dcosts, applied_actions, battery_s = dmdp.find_policy(dQ, test_data)
        results[2,i] = np.sum(dcosts)

        # model without prediciton
        rmdp = rMDP(1000,500,500,200,6000,5,5)
        # rQ_table, rewards_per_episode, all_rewards, actions, states_id, states, battery = rQ.iterate(training_data,n,0.5,0.9, rmdp)
        file_path_r = os.path.join(subfolder_name, 'rQ' + str(n)+ '.csv')
        rQ =  np.genfromtxt(file_path_r, delimiter=',')
        rcosts, applied_actions, battery_s = rmdp.find_policy(rQ, test_data)
        results[3,i] = np.sum(rcosts)

        # Baseline
        baseline_rewards, baseline_states, baseline_actions, baseline_bat, difference= Baseline.find_baseline_policy(test_data, mdp)
        results[4,i] = np.sum(baseline_rewards)
        bs = mdp.get_total_costs(test_data[:,1] - test_data[:,0])
        results[5,i] = bs
    fig, ax = plt.subplots()
    colors = ["lightcoral", "sandybrown", "yellowgreen", "lightslategrey"]
    
    labels = ["MDP with 5 bins", "MDP with 3 bins","MDP with difference","MDP without prediciton"]
    markers = ['^','s','x','o', ]

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
    plt.savefig("state_spaces.png")
    #plt.show()

#train_models([500])
#test_state_spaces([500])
# plt.show()
# test_state_spaces()
### Policy comparisons ####
def test_policies(iterations):
    training_data, test_data = RealData.get_training_test(7, False, False)
    subfolder_name = 'Q_SA_models'
    # difference
    # without prediction, 5 production
    # 5 bins normal
    # rule based baseline
    # normal model with 5 bins
    mdp = MDP(1000,500,500,200,6000,5,5)
    # Q, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(training_data,5000,0.5,0.9, mdp)
    
    file_path_5 = os.path.join(subfolder_name, 'Q5' + str(max(iterations))+ '.csv')
    Q5 =  np.genfromtxt(file_path_5, delimiter=',')
    costs_5, policy_5, battery_s = mdp.find_policy(Q5, test_data)

    # normal model with 3 bins
    mdp = MDP(1000,500,500,200,6000,3,3)
    # Q, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(training_data,n,0.5,0.9, mdp)
    file_path_3 = os.path.join(subfolder_name, 'Q3' + str(max(iterations))+ '.csv')
    Q3=  np.genfromtxt(file_path_3, delimiter=',')
    costs_3, policy_3, battery_s = mdp.find_policy(Q3, test_data)
    
    # model with difference
    dmdp = dMDP(1000,500,500,200,6000)
    # dQ_table, rewards_per_episode, all_rewards, actions, states_id, states, battery = dQ.iterate(training_data,n,0.5,0.9, dmdp)
    file_path_d = os.path.join(subfolder_name, 'dQ' + str(max(iterations))+ '.csv')
    dQ =  np.genfromtxt(file_path_d, delimiter=',')
    dcosts, policy_d, battery_s = dmdp.find_policy(dQ, test_data)
    
    # model without prediciton and 3 bins
    rmdp = rMDP(1000,500,500,200,6000,5,5)
    # rQ_table, rewards_per_episode, all_rewards, actions, states_id, states, battery = rQ.iterate(training_data,n,0.5,0.9, rmdp)
    file_path_r = os.path.join(subfolder_name, 'rQ' + str(max(iterations))+ '.csv')
    rQ =  np.genfromtxt(file_path_r, delimiter=',')
    rcosts, policy_r, battery_s = rmdp.find_policy(rQ, test_data)
    
    # Baseline
    baseline_rewards, baseline_states, policy_baseline, baseline_bat, difference= Baseline.find_baseline_policy(test_data, mdp)
    fig, ax = plt.subplots()
    #plt.style.use('seaborn-deep')
    
    plt.hist([policy_5, policy_3, policy_d, policy_r, policy_baseline], label= ['5 bins','3 bins', 'with difference', 'without prediction', 'rule-based baseline'], color = ["lightcoral", "sandybrown", "yellowgreen", "lightslategrey","purple"], rwidth=0.8)
    ax.legend(loc = 'upper right')

    # print(np.sum(costs_5), np.sum(dcosts), np.sum(rcosts), np.sum(baseline_rewards))
# test_policies([500])
# plt.show()

def test_batteries(iterations, start, end):
    training_data, test_data = RealData.get_training_test(7, False, False)
    subfolder_name = 'Q_SA_models'
    # difference
    # without prediction, 5 production
    # 5 bins normal
    # rule based baseline

    # normal model with 5 bins
    mdp = MDP(1000,500,500,200,6000,5,5)
    file_path_5 = os.path.join(subfolder_name, 'Q5' + str(max(iterations))+ '.csv')
    Q5 =  np.genfromtxt(file_path_5, delimiter=',')
    costs_5, policy_5, battery_5 = mdp.find_policy(Q5, test_data)

    # normal model with 3 bins
    mdp = MDP(1000,500,500,200,6000,3,3)
    file_path_3 = os.path.join(subfolder_name, 'Q3' + str(max(iterations))+ '.csv')
    Q3 =  np.genfromtxt(file_path_3, delimiter=',')
    costs_3, policy_3, battery_3 = mdp.find_policy(Q3, test_data)
    
    # model with difference
    dmdp = dMDP(1000,500,500,200,6000)
    file_path_d = os.path.join(subfolder_name, 'dQ' + str(max(iterations))+ '.csv')
    dQ =  np.genfromtxt(file_path_d, delimiter=',')
    dcosts, policy_d, battery_d = dmdp.find_policy(dQ, test_data)

     # model without prediciton and 3 bins
    rmdp = rMDP(1000,500,500,200,6000,5,5)
    # rQ_table, rewards_per_episode, all_rewards, actions, states_id, states, battery = rQ.iterate(training_data,n,0.5,0.9, rmdp)
    file_path_r = os.path.join(subfolder_name, 'rQ' + str(max(iterations))+ '.csv')
    rQ =  np.genfromtxt(file_path_r, delimiter=',')
    rcosts, policy_r, battery_r = rmdp.find_policy(rQ, test_data)
    
    # Baseline
    baseline_rewards, baseline_states, policy_baseline, baseline_bat, difference= Baseline.find_baseline_policy(test_data, mdp)
    labels = ['5 bins MDP', '3 bins MDP', 'MDP with difference', 'MDP without prediction']
    batteries = [battery_5, battery_3, battery_d, battery_r]
    colors = ["lightcoral", "sandybrown", "yellowgreen", "lightslategrey"]
    fig = plt.figure()
    
    for i,b in enumerate(batteries):
        
        ax = fig.add_subplot(2,2,i+1)
        ax.plot(b[start:end], color = colors[i])
        ax.set_title(labels[i])
        ax.plot(baseline_bat[start:end], color = "lightgrey", linestyle = "dashdot")
        ax.set_xlabel('Days')
        ax.set_ylabel('State of battery')
        ax.set_xticks(np.arange(12,200,24), np.arange(0,8,1))
        
    plt.suptitle('Battery states for different Models')    
    plt.savefig("batteries.png")
    plt.tight_layout()
    print(np.sum(costs_5), np.sum(dcosts), np.sum(rcosts), np.sum(baseline_rewards))
# first week of june
# test_batteries([500], 0,186)
# # first week of december
# test_batteries([500], 1008, 1194)
plt.show()

############## TEST EPSILONS ##########################
def test_epsilons():
    epsilons = [1,3,5,7,9]
    mdp = MDP(1000,500,500,200,6000,7,7)
    iterations = [100,500,750,1000,2000,5000]
    # iterations = [5,10,15]
    results = np.zeros((len(epsilons), len(iterations)))
    for j,x in enumerate(iterations):
        for i, e in enumerate(epsilons):
            
            Q, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(training_data,x,0.5,0.9,e, mdp)
            cost, applied_actions, battery_states, dis, loss, states = mdp.find_policy(Q, test_data)
            results[i,j] = np.sum(cost)
    
    fig, ax = plt.subplots()
    colors = ["lightcoral", "sandybrown", "yellowgreen", "lightslategrey", "royalblue"]
    labels = ["decreasing decay = 1", "decreasing decay = 3","decreasing decay = 5","decreasing decay = 7","decreasing decay = 9"]
    markers = ['^','s','x','o', 'd']

    for r in range(len(epsilons)):
        plt.plot(results[r,], color = str(colors[r]), label = str(labels[r]), linestyle = "solid", marker = markers[r], markersize = 5)
    baseline_rewards, baseline_states, baseline_actions, baseline_bat, difference= Baseline.find_baseline_policy(test_data, mdp)
    plt.plot([np.sum(baseline_rewards)]*len(iterations), label ="rule-based Baseline", color = "purple", linestyle = "dashdot")
    bs_without = mdp.get_total_costs(test_data[:,1] - test_data[:,0])
    plt.plot([bs_without]*len(iterations), label ="Baseline without ESS", color = "grey", linestyle = "dashed")
    
    plt.xticks(np.arange(0,len(iterations),1), iterations)
    #plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlabel('Number of training episodes')
    ax.set_ylabel('Costs')
    plt.title('Effect of exploration-exploitation trade-off')
    ax.legend(loc = 'upper left')
    plt.savefig("epsilon.png")
    # plt.show()
# test_epsilons()
# plt.show()
############### ACTIONS #################################
# print(actions)
# daytime: 9 am - 6 pm
def check_daytime_actions(actions):
    daytime_actions = []
    daytime = [0]*8 + [1]*10 + [0]*6
   
    for i in range(len(actions)):
        if daytime[i % 24] == 1:
            daytime_actions.append(actions[i])
    return daytime_actions



def plot_batteries(a, b, battery_states, baseline_bat):
    plt.plot(battery_states[a:b], color = "red")
    plt.plot(baseline_bat[a:b], color = "green")
    plt.show()


def compare_actions_hist(mdp):
    Q_table_sol, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(training_data,10, mdp)
    reward, applied_actions, battery_states, dis = MDP.find_policy(mdp, Q_table_sol, training_data)
    fig, ax = plt.subplots()
    # Define the bins
    bins = ['charge_high', 'charge_low', 'discharge_high', 'discharge_low', 'do nothing']

    daytime_actions = check_daytime_actions(applied_actions)

    # Map the bin labels to numerical values
    bin_values = np.arange(len(bins))
    baseline_rewards, baseline_states, baseline_actions, baseline_bat, difference= Baseline.find_baseline_policy(training_data, mdp)

    # Convert the data lists to numerical values based on the bin mapping
    # choose between all actions and baseline actions
    applied_actions_numeric = np.array([bin_values[bins.index(value)] for value in applied_actions])
    # daytime_actions_numeric = np.array([bin_values[bins.index(value)] for value in daytime_actions])
    daytime_actions_numeric = np.array([bin_values[bins.index(value)] for value in baseline_actions])

    # Set up the figure and subplots
    fig, ax = plt.subplots()

    # Create the histograms

    hist1, bins, _ = ax.hist(applied_actions_numeric, bins=bin_values, alpha=0.5, label='Applied Actions', density=True)
    hist2, _, _ = ax.hist(daytime_actions_numeric, bins=bin_values, alpha=0.5, label='Daytime Actions', density=True)

    # Set the x-axis tick locations and labels
    bin_centers = bin_values + 0.5  # Center the bars
    ax.set_xticks(bins)
    ax.set_xticklabels(bins, rotation=45)

    # Add labels and legend
    ax.set_xlabel('Category')
    ax.set_ylabel('Normalized Frequency')
    ax.legend()

    plt.show()



################ BASELINE TESTING ##########################
baseline_rewards, baseline_states, baseline_actions, baseline_bat, difference= Baseline.find_baseline_policy(test_data, mdp)

# plt.show()
# plt.hist(baseline_actions)
# plt.show()


# print(difference.count(True))
# plt.show()
#print(baseline_rewards)

# print(baseline_bat)
# print(baseline_actions)
# print("baseline: ")
# print(baseline_rewards)
# print(MDP.get_total_costs(baseline_rewards))
unique = set(baseline_states)
print(len(unique))
# print(baseline_states.count())
# print(baseline_ids)
# print(baseline_actions.count("do nothing"))

#  print(MDP.get_total_costs(baseline_rewards))
# actions_id = [MDP.get_action_id(x) for x in baseline_actions]
# print(actions_id)


# plt.plot(baseline_ids[:(24*2)])
# plt.show()


# print(dat[:(24*3)])

############# Baseline RL Comparison #################
# diff = [baseline_rewards[i] - reward[i] for i in range(len(reward))]
# print()
# print(diff)
# # print(np.sum(diff))
# Q2, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning_d.iterate(training_data,500, mdp_d)

# reward2, policy2, battery_states2, dis2, loss, visited_states = mdp_d.find_policy(Q2, test_data)


# Q1, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(training_data,1000, mdp)

# reward1, policy1, battery_states1, dis, loss, visited_states = mdp.find_policy(Q1, test_data)


# print("Q-Learning normal: " + str(np.sum(reward1)))
# # print("Q-Learning with difference: " + str(np.sum(reward2)))

# print("Baseline:   " + str(np.sum(baseline_rewards)))
# print("without battery: " + str(mdp.get_total_costs(test_data["Production"] - test_data["Consumption"])))

# # diff = [data["Purchased"][i] + reward[i] for i in range(len(reward))]
# # print("diff: " + str(diff))
# # print(len(reward))
# # print(len(data["Purchased"]))
# # plot_batteries(0,240,battery_states, baseline_bat)
# print("amount discharged 1: " + str(dis))
# # print("amount discharged2 : " + str(dis2))

# # print("amount wasted when discharging" + str(loss))
# # print(applied_actions)

# plt.plot(rewards_per_episode)
# plt.show()
# cons = test_data["Consumption"]
# scaled_cons = [x/100 for x in cons]
# prod = test_data["Production"]
# scaled_prod = [x/100 for x in prod]


# # plt.plot(battery_states1[:186], color = "red")
# # plt.plot(scaled_cons[1000:1240], color = "green")
# # plt.plot(scaled_prod[1000:1240], color = "yellow")
# # plt.show()
# plt.plot(battery_states1[:186], color = "red")

# # plt.plot(battery_states2[:186], color = "green")
# plt.plot(baseline_bat[:186], color = "black")
# plt.show()

# plt.plot(battery_states1[200:386], color = "red")
# plt.plot(baseline_bat[200:386], color = "black")
# plt.show()


# plt.hist(policy1, color = "red")
# plt.show()

# plt.hist(policy2, color = "green")

def data_to_states(mdp, data):
    states = []
    for i in range(len(data)-1):
        #print(MDP.get_production(data["Production"][i]))
        # data["Consumption"][i], data["Production"][i], , data["Time"][i]
        states.append((mdp.get_consumption(data["Consumption"][i]), mdp.get_production(data["Production"][i]),mdp.get_production(data["Production"][i+1]), data["Time"][i]))
        # print(data["Consumption"][i])
        # print(State(data["Consumption"][i], data["Production"][i], 2, data["Production"][i+1],data["Time"][i]).consumption)
    # return states   
    return Counter(states)
# print(len(Counter(baseline_states)))
#print(baseline_states[:100])
# plt.hist(baseline_actions)
# plt.show()