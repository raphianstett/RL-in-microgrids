import MDP
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from MDP import State
from MDP import MDP
from data import StepFunctions
from data import RealData

np.set_printoptions(threshold=np.inf)


# import test data
dat_test = StepFunctions.get_test_data()

# import real data
data = RealData.get_real_data()
# training_data, test_data = RealData.split_data(dat, 1)

# implementation of q-learning algorithm
class ValueIteration:
# define parameters for q-learning
    def value_iteration(data, n_episodes):
        print("####GO#####")

        #initialize the exploration probability to 1
        exploration_proba = 1

        #exploartion decreasing decay for exponential decreasing
        exploration_decreasing_decay = 4 / n_episodes

        # minimum of exploration proba
        min_exploration_proba = 0.01

        #discounted factor
        gamma = 0.8

        #learning rate
        lr = 0.4

        # initialize rewards per episode

        rewards_per_episode = []
        all_rewards = []
        chosen_actions = []
        battery = []
        states_id = []
        states = []

        # initialize Q-table
        Q_table = np.zeros((MDP.n_states, MDP.n_actions))
        
        # we initialize the first state of the episode
        current_state = State(data["Consumption"][0], data["Production"][0], 2, data["Production"][1] ,data["Time"][0])
            
            
        for e in range(n_episodes):
            
            #sum the rewards that the agent gets from the environment
            total_episode_reward = 0
            
            for i in range(0, len(data["Consumption"])): 
            
                # sample a float from a uniform distribution over 0 and 1
                # if the sampled flaot is less than the exploration proba
                #     the agent selects arandom action
                # else
                #     he exploits his knowledge using the bellman equation 
                
                if np.random.uniform(0,1) < exploration_proba:
                    action = MDP.action_space[np.random.randint(0,MDP.n_actions)]
                else:
                    a = Q_table[State.get_id(current_state),:]
                    action = MDP.action_space[MDP.get_best_action(a)]
                    # action = MDP.action_space[np.argmax(Q_table[State.get_id(current_state),:])]
                
                # The environment runs the chosen action and returns the next state and the reward for the action in the current state.
                reward = State.get_reward(action, current_state)

                # if i == len(data["Consumption"])-1:
                #     next_state = State.get_next_state(current_state, action, data["Consumption"][0], data["Production"][0], data["Time"][0])
                # else:
                next_state = State.get_next_state(current_state, action, data["Consumption"][(i+1)%len(data["Consumption"])], data["Production"][(i+1)%len(data["Consumption"])], data["Production"][(i+2)%len(data["Consumption"])], data["Time"][(i+1)%len(data["Consumption"])])

            
                # We update our Q-table using the Q-learning iteration
                Q_table[State.get_id(current_state), MDP.get_action_id(action)] = (1-lr) * Q_table[State.get_id(current_state), MDP.get_action_id(action)] + lr*(reward + gamma*max(Q_table[State.get_id(next_state),:]))
                total_episode_reward = total_episode_reward + reward
                
                all_rewards.append(reward)
                chosen_actions.append(MDP.get_action_id(action))
                battery.append(current_state.battery)
                states.append((current_state.consumption, current_state.production, current_state.battery, current_state.time))
                states_id.append(State.get_id(current_state))
                # states.append(current_state)

                current_state = next_state
            #We update the exploration proba using exponential decay formula 
            exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*e))
            rewards_per_episode.append(total_episode_reward)
            print(e)
        
        return Q_table, rewards_per_episode, all_rewards, chosen_actions, states_id, states, battery


class Baseline:
    def test(data):
        rewards = []
        states = []
        actions = []
        action_ids = []
        current_state = State(data["Consumption"][0], data["Production"][0], 2,data["Production"][1], data["Time"][0])
        
        for i in range(1,len(data["Consumption"])):
            # current_state = State(data["Consumption"][i], data["Production"][i], 2, data["Time"][i])

            if current_state.p > current_state.c and current_state.p - current_state.c > 1000 and current_state.battery < MDP.max_battery:
                action = "charge"
            elif current_state.p > current_state.c and current_state.c - current_state.p < 1000:
                action = "do nothing"
            elif current_state.p < current_state.c and not (current_state.p - current_state.c) > 1000 and current_state.battery > 0:
                action = "discharge"
            else:
                action = "do nothing"
        
            rewards.append(State.get_cost(action, current_state))
            states.append((current_state.consumption, current_state.production, current_state.battery,current_state.time))
            actions.append(action)
            action_ids.append(MDP.get_action_id(action))
            current_state = State.get_next_state(current_state, action, data["Consumption"][i], data["Production"][i],data["Production"][(i+1)%len(data["Production"])], data["Time"][i])

        return rewards, states, actions, action_ids


Q_table_sol, rewards_per_episode, all_rewards, actions, states_id, states, battery = ValueIteration.value_iteration(data,10)



################## APPLIED Q-TABLE #################################
# print(Q_table_sol)

reward, applied_actions, battery_states = MDP.apply_q_table(Q_table_sol, data)

# print(reward)

# print(applied_actions)
# plt.hist(applied_actions)
# plt.show()
# print(MDP.get_total_costs(reward))
# print(len(applied_actions))


# print(Q_table_sol)
# print(len(Q_table_sol))
# print(len(applied_actions))
# # plt.plot(applied_actions)
# plt.show()
# print(Q_table_sol)
plt.hist(battery_states)
plt.show()

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
rewards = [x/len(data["Purchased"]) for x in rewards_per_episode]
# plt.plot(rewards)
# plt.show()
# print(all_rewards)
# print(rewards)

# print(len(dat))

############### ACTIONS #################################
# print(actions)
# plt.hist(actions)
# plt.show()


################ Q-TABLE #################################
# print(Q_table_sol)


############## BATTERY #################################

# plt.hist(battery)
# plt.show()
################## STATES #################################  
# print(states[:1000])
# print(len(states))

#plt.show()

################ BASELINE TESTING ##########################
baseline_rewards, baseline_states, baseline_actions, baseline_ids= Baseline.test(data)
# print("baseline: ")
# print(baseline_rewards)
# print(MDP.get_total_costs(baseline_rewards))
# # print(baseline_states)

# actions_id = [MDP.get_action_id(x) for x in baseline_actions]
# print(actions_id)


# plt.plot(baseline_ids[:(24*2)])
# plt.show()


# print(dat[:(24*3)])

############# Baseline RL Comparison #################
# diff = [baseline_rewards[i] - reward[i] for i in range(len(reward))]
# print()
# print(diff)
# print(np.sum(diff))


print("Q-Learning: " + str(np.sum(reward)))
print("Baseline:   " + str(MDP.get_total_costs(baseline_rewards)))
print("without battery: -" + str(np.sum(data["Purchased"])))


# diff = [data["Purchased"][i] + reward[i] for i in range(len(reward))]
# print("diff: " + str(diff))
# print(len(reward))
# print(len(data["Purchased"]))
