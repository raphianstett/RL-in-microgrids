from learning import QLearning
from learning import Baseline

from environment import MDP
from environment import State
from SA_variations.environment_with_diff import MDP as dMDP
from SA_variations.environment_with_diff import State as dState

from data import StepFunctions
from data import RealData

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import FormatStrFormatter


##### TESTING FOR RL EVALUATION ####
# import test data
dat_test = StepFunctions.get_test_data()

# import real data
data = RealData.get_real_data()

summer_data = RealData.get_summer(data)

training_data, test_data = RealData.split_data(summer_data, 7)

# initialize MDP
# MDP(max_charge, max_discharge, charge_low, discharge_low, max_battery, bins_cons, bins_prod)

mdp_3 = MDP(1000, 1000, 500, 500, 6000, 3,3)
mdp_5 = MDP(1000, 1000, 500, 500, 6000, 5,5)
mdp_7 = MDP(1000, 500, 500, 200, 4000, 7,7)
mdp_10 = MDP(1000, 500, 500, 200, 4000, 10,10)
mdp = mdp_7
mdp_d = dMDP(1000, 500, 200, 500, 4000,  10)



### TEST BINNING IN RL CONVERGENCE ###
def test_binning():
    bins = [3,5,7,10]
    # bins = [3,5]
    iterations = [100,500,750,1000,2000,5000, 10000]
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
    

test_binning()
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
test_steps()
    
##### TEST LEARNING RATE #####
# can be copied to test gamma
def test_lr():
    lrs = np.arange(0.1,1.1,0.1)
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
test_lr()

def test_gamma():
    gammas = np.arange(0.1,1.1,0.1)
    mdp = MDP(1000,500,500,200,6000,7,7)
    iterations = [100,500,1000,5000]
    # iterations = [5,10,15]
    results = np.zeros((len(iterations), len(gammas)))
    for j,x in enumerate(iterations):
        for i, g in enumerate(gammas):
            print("gamma")
            Q, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(training_data,x,0.5,g, mdp)
            cost, applied_actions, battery_states, dis, loss, states = mdp.find_policy(Q, test_data)
            results[j,i] = np.sum(cost)
    
    fig, ax = plt.subplots()
    colors = ["lightcoral", "sandybrown", "yellowgreen", "lightslategrey"]
    labels = ["Q-learning - 100 episodes", "Q-learning - 500 episodes","Q-learning - 1000 episodes","Q-learning - 5000 episodes"]
    markers = ['^','s','x','o']

    for r in range(len(iterations)):
        plt.plot(results[r,], color = str(colors[r]), label = str(labels[r]), linestyle = "solid", marker = markers[r], markersize = 5)
    baseline_rewards, baseline_states, baseline_actions, baseline_bat, difference= Baseline.find_baseline_policy(test_data, mdp)
    plt.plot([np.sum(baseline_rewards)]*len(gammas), label ="rule-based Baseline", color = "purple", linestyle = "dashdot")
    bs_without = mdp.get_total_costs(test_data["Production"] - test_data["Consumption"])
    plt.plot([bs_without]*len(gammas), label ="Baseline without ESS", color = "grey", linestyle = "dashed")
    
    plt.xticks(np.arange(0,10,1), gammas)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlabel('Discount rate')
    ax.set_ylabel('Costs')
    plt.title('Discount rate and RL performance')
    ax.legend()
    plt.savefig("Discount_rate.png")
    # plt.show()
test_gamma()


######## TEST SEASONAL DIFFERENCES IN DATA AND RL PERFORMANCE  #############
def test_seasons():
    data = RealData.get_real_data()
    summer_data = RealData.get_summer(RealData.get_real_data())
    winter_data = RealData.get_winter(RealData.get_real_data())

    training_summer, test_summer = RealData.split_data(summer_data, 7)
    training_winter, test_winter = RealData.split_data(winter_data, 7)
    training, test = RealData.split_data(data, 7)
    
    # use optimal values for binning, step-sizes and learning rate
    mdp = MDP(1000,500,500,200,6000,7,7)
    QS, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(training_summer,1000,0.5,0.9, mdp)
    cost_s, applied_actions, battery_s, dis, loss, states = mdp.find_policy(QS, test_summer)
    bs_s = mdp.get_total_costs(test_summer["Production"] - test_summer["Consumption"])
    QW, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(training_winter,1000,0.5,0.9, mdp)
    cost_w, applied_actions, battery_s, dis, loss, states = mdp.find_policy(QW, test_winter)
    bs_w = mdp.get_total_costs(test_winter["Production"] - test_winter["Consumption"])
    Q, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(training,1000,0.5,0.9, mdp)
    cost, applied_actions, battery_s, dis, loss, states = mdp.find_policy(Q, test)
    bs = mdp.get_total_costs(test["Production"] - test["Consumption"])
    return np.sum(cost_s)/len(test_summer),bs_s / len(test_summer), np.sum(cost_w)/len(test_winter),bs_w/len(test_winter),np.sum(cost) / len(test),bs/len(test)
# print(test_seasons())

############### TEST DIFFERENT STATE SPACES ########################
from SA_variations.environment_with_diff import MDP as dMDP
from SA_variations.environment_without_pred import MDP as rMDP
from SA_variations.learning_without_pred import QLearning as rQ
from SA_variations.learning_with_diff import QLearning as dQ

def test_state_spaces():
    summer_data = RealData.get_summer(RealData.get_real_data())

    training_data, test_data = RealData.split_data(summer_data, 7)
    iterations = [100,500,1000,2500,5000]
    # iterations = [1,5,10]
    results = np.zeros((6,len(iterations)))
    for i,n in enumerate(iterations):
        print("state spaces")
        # normal model with 5 bins
        mdp = MDP(1000,500,500,200,6000,5,5)
        Q, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(training_data,n,0.5,0.9, mdp)
        costs_5, applied_actions, battery_s, dis, loss, states = mdp.find_policy(Q, test_data)
        results[0,i] = np.sum(costs_5)
        # normal model with 10 bins
        mdp = MDP(1000,500,500,200,6000,10,10)
        Q, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(training_data,n,0.5,0.9, mdp)
        costs_10, applied_actions, battery_s, dis, loss, states = mdp.find_policy(Q, test_data)
        results[1,i] = np.sum(costs_10)

        # model with difference
        dmdp = dMDP(1000,500,500,200,6000,10)
        dQ_table, rewards_per_episode, all_rewards, actions, states_id, states, battery = dQ.iterate(training_data,n,0.5,0.9, dmdp)
        dcosts, applied_actions, battery_s, dis, loss = dmdp.find_policy(dQ_table, test_data)
        results[2,i] = np.sum(dcosts)

        # model without prediciton
        rmdp = rMDP(1000,500,500,200,6000,10,10)
        rQ_table, rewards_per_episode, all_rewards, actions, states_id, states, battery = rQ.iterate(training_data,n,0.5,0.9, rmdp)
        rcosts, applied_actions, battery_s, dis, loss = rmdp.find_policy(rQ_table, test_data)
        results[3,i] = np.sum(rcosts)
        # Baseline
        baseline_rewards, baseline_states, baseline_actions, baseline_bat, difference= Baseline.find_baseline_policy(test_data, mdp)
        results[4,i] = np.sum(baseline_rewards)
        bs = mdp.get_total_costs(test_data["Production"] - test_data["Consumption"])
        results[5,i] = bs
        fig, ax = plt.subplots()
    colors = ["lightcoral", "sandybrown", "yellowgreen", "lightslategrey"]
    
    labels = ["MDP with 5 bins", "MDP with 10 bins","MDP with difference","MDP without prediciton"]
    markers = ['^','s','x','o', ]

    for r in range(4):
        plt.plot(results[r,], color = str(colors[r]), label = str(labels[r]), linestyle = "solid", marker = markers[r], markersize = 5)
    
    plt.plot(results[4,], label ="rule-based Baseline", color = "purple", linestyle = "dashdot")
    
    plt.plot(results[5,], label ="Baseline without ESS", color = "grey", linestyle = "dashed")
    
    plt.xticks(np.arange(0,3,1),labels =  iterations)
    # plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlabel('Number of training episodes')
    ax.set_ylabel('Costs')
    plt.title('Effect of MDP on RL performance')
    ax.legend()
    plt.savefig("state_spaces.png")
    #plt.show()

    
test_state_spaces()
plt.show()
################## APPLIED Q-TABLE #################################
# print(Q_table_sol)
rewards_it = [None]*40
# for i in range(40):
#     reward, applied_actions, battery_states = MDP.find_policy(Q_table_sol, training_data)
    
#     #print(reward)
#     print(np.sum(reward))
#     rewards_it[i] = np.sum(reward)


# print(rewards_it)
# plt.plot(rewards_it)
# plt.show()
# plt.hist(MDP.iterate_q(Q_table_sol))
# plt.show()
# print(MDP.iterate_q(Q_table_sol))
# print(reward)

# print(applied_actions)

# plt.plot(battery_states[1:220])
# plt.show()

# plt.show()
#print(reward)

# print(applied_actions)
# print("amount discharged:" + str(dis))

# print(Q_table_sol)
# print(len(Q_table_sol))
# print(len(applied_actions))
# # plt.plot(applied_actions)
# plt.show()
# print(Q_table_sol)
# plt.plot(battery_states[:240])
# plt.show()

# print(battery_states)
## APPLIED ACTIONS
def analyse_actions(applied_actions):
    l = int(np.ceil(len(applied_actions)/24))
    av_day = [0]*24
    for i in range(l-1):
        for j in range(24):
            av_day[j] += applied_actions[i*24+j]
    day = [x/l for x in av_day]
    return day

# print(analyse_actions(applied_actions))
# plt.plot(analyse_actions(applied_actions))
# plt.show()
#print(np.ceil(len(applied_actions)/24))
# ################## REWARDS #################################
#plt.plot(rewards_per_episode)
# rewards = [x/len(data["Purchased"]) for x in rewards_per_episode]
# plt.plot(rewards)
# plt.show()
# print(all_rewards)
# # print(rewards)

# print(len(dat))

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

################ Q-TABLE #################################
# print(Q_table_sol)


############## BATTERY #################################

# plt.plot(battery[:240])
# plt.show()

################## STATES #################################  
# print(states[:1000])
# print(len(states))

#plt.show()

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