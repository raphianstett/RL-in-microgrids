from learning import QLearning
from learning import Baseline
from learning_with_diff import QLearning as QLearning_d
from environment import MDP
from environment import State
from environment_with_diff import MDP as dMDP
from environment_with_diff import State as dState

from data import StepFunctions
from data import RealData

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def test_exploration(i):
    exp = [0]*i
    min_exploration_proba = 0.01
    #exploration_decreasing_decay = 0.01
    exploration_decreasing_decay = 5 / i
    exploration_proba = 1
    for i in range(i):  
          
        exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*i))
        exp[i] = exploration_proba
    return exp

##### TESTING FOR RL EVALUATION ####
# import test data
dat_test = StepFunctions.get_test_data()

# import real data
data = RealData.get_real_data()
summer_data = RealData.get_summer(data)
training_data, test_data = RealData.split_data(summer_data, 7)

# initialize MDP
# MDP(max_charge, max_discharge, discharge_low, charge_low, max_battery, bins_cons, bins_prod)

mdp_3 = MDP(1000, 1000, 500, 500, 6000, 3,3)
mdp_5 = MDP(1000, 1000, 500, 500, 6000, 5,5)
mdp_7 = MDP(2000, 500, 200, 200, 6000, 7,7)
mdp_10 = MDP(1000, 500, 500, 200, 4000, 10,10)
mdp = mdp_10
mdp_d = dMDP(1000, 500, 200, 500, 4000, 10, 10)



### TEST BINNING IN RL CONVERGENCE ###


def test_binning():
    bins = [3,5,7,10]
    # bins = [3,5]
    test_iterations = [25,50,75,100,125,150,200,250,300,350]
    # test_iterations = [20,50,75,100]
    # test_iterations = [1,2]
    solutions = np.zeros((len(bins), len(test_iterations)))
    
    for b in range(len(bins)):
        for i in range(len(test_iterations)):
            mdp = MDP(1000, 1000, 500, 500, 6, bins[b], bins[b])
            Q_table_sol, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(training_data,test_iterations[i], mdp)
            reward, applied_actions, battery_states, dis = MDP.find_policy(mdp, Q_table_sol, training_data)
            solutions[b,i] = np.sum(reward)
    
    colors = ["red", "yellow", "blue", "green"]
    fig, ax = plt.subplots()
    for b in range(len(bins)):
        plt.plot(solutions[b,], color = str(colors[b]), label = str(bins[b]) + " bins")
        
    ax.legend()
    baseline_rewards, baseline_states, baseline_actions, baseline_bat, difference= Baseline.find_baseline_policy(training_data, mdp)
    plt.plot([np.sum(baseline_rewards)]*len(test_iterations), label ="Baseline", color = "purple", linestyle = "dashdot")
    
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Sum of purchased energy')
    plt.show()
    

# test_binning()


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


Q1, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(training_data,1000, mdp)

reward1, policy1, battery_states1, dis, loss, visited_states = mdp.find_policy(Q1, test_data)


print("Q-Learning normal: " + str(np.sum(reward1)))
# print("Q-Learning with difference: " + str(np.sum(reward2)))

print("Baseline:   " + str(np.sum(baseline_rewards)))
print("without battery: " + str(mdp.get_total_costs(test_data["Production"] - test_data["Consumption"])))

# diff = [data["Purchased"][i] + reward[i] for i in range(len(reward))]
# print("diff: " + str(diff))
# print(len(reward))
# print(len(data["Purchased"]))
# plot_batteries(0,240,battery_states, baseline_bat)
print("amount discharged 1: " + str(dis))
# print("amount discharged2 : " + str(dis2))

# print("amount wasted when discharging" + str(loss))
# print(applied_actions)

# plt.plot(rewards_per_episode)
# plt.show()
cons = test_data["Consumption"]
scaled_cons = [x/100 for x in cons]
prod = test_data["Production"]
scaled_prod = [x/100 for x in prod]


# plt.plot(battery_states1[:186], color = "red")
# plt.plot(scaled_cons[1000:1240], color = "green")
# plt.plot(scaled_prod[1000:1240], color = "yellow")
# plt.show()
plt.plot(battery_states1[:186], color = "red")

# plt.plot(battery_states2[:186], color = "green")
plt.plot(baseline_bat[:186], color = "black")
plt.show()
plt.hist(policy1, color = "red")
plt.show()

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
plt.hist(baseline_actions)
plt.show()