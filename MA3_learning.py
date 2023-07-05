import environment
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from MA3_environment import Reward
from MA3_environment import State
from environment import State as State2

class MA_QLearning:
# define parameters for q-learning
    def iterate(data, n_episodes, mdp):
        print("####GO#####")

        #initialize the exploration probability to 1
        exploration_proba = 1

        #exploartion decreasing decay for exponential decreasing
        exploration_decreasing_decay = 4/n_episodes #4 / n_episodes

        # minimum of exploration proba
        min_exploration_proba = 0.05

        #discounted factor
        gamma = 0.8

        #learning rate
        lr = 0.5

        rewards_per_episode = []
        all_rewards = []
        chosen_actions = []
        battery = []
        states_id = []
        states = []
        changed = 0

        # initialize Q-table
        Q_A = np.zeros((mdp.n_states, mdp.n_actions))
        Q_B = np.zeros((mdp.n_states, mdp.n_actions))
        Q_C = np.zeros((mdp.n_states, mdp.n_actions_c))
        # initialize the first state of the episode
        state_A = State(data["Consumption_A"][0], data["Production_A"][0], 6000, data["Production_A"][1] ,data["Time"][0], mdp)
        state_B = State(data["Consumption_B"][0], data["Production_B"][0], 6000, data["Production_B"][1] ,data["Time"][0], mdp)
        state_C = State(data["Consumption_C"][0], data["Production_C"][0], 6000, data["Production_C"][1] ,data["Time"][0], mdp)
            
        l = len(data["Consumption_A"])
        for e in range(n_episodes):
            
            #sum the rewards that the agent gets from the environment
            total_reward = 0
            
            for i in range(0, l): 
                print(i)
                state_A_id = State2.get_id(state_A, mdp)
                state_B_id = State2.get_id(state_B, mdp)
                state_C_id = State2.get_id(state_C, mdp)
                # exploration 
                if np.random.uniform(0,1) < exploration_proba:
                    action_A = mdp.action_space[np.random.randint(0,mdp.n_actions)]
                    action_B = mdp.action_space[np.random.randint(0,mdp.n_actions)]
                    action_C = mdp.action_space_c[np.random.randint(0, mdp.n_actions_c)]
                # exploitation
                else:
                    # choose best action for given state from Q-table
                    a_A = Q_A[state_A_id,:]
                    action_A = mdp.action_space[mdp.get_best_action(a_A)]
                    a_B = Q_B[state_B_id,:]
                    action_B = mdp.action_space[mdp.get_best_action(a_B)]
                    a_C = Q_C[state_C_id,:]
                    action_C = mdp.action_space_c[mdp.get_best_action(a_C)]
                    
                action_A_id = mdp.get_action_id(action_A)
                prev_id_A = action_A_id
                action_B_id = mdp.get_action_id(action_B)
                action_C_id = mdp.get_action_id(action_C)
                
                
                # run the chosen action and return the next state and the reward for the action in the current state.
                reward_A, reward_B, reward_C =  Reward.get_reward(state_A, state_B, state_C, action_A, action_B, action_C, mdp)
                
                all_rewards.append(reward_A)
                all_rewards.append(reward_B)
                all_rewards.append(reward_C)

                # reward_B = Reward.get_reward(state_B, action_B,mdp)
                
                next_state_A = State.get_next_state(state_A, data["Consumption_A"][(i+1)%l], data["Production_A"][(i+1)%l], data["Production_A"][(i+2)%l], data["Time"][(i+1)%l], mdp, action_A, action_B, action_C)
                next_state_B = State.get_next_state(state_B, data["Consumption_B"][(i+1)%l], data["Production_B"][(i+1)%l], data["Production_B"][(i+2)%l], data["Time"][(i+1)%l], mdp, action_A, action_B, action_C)
                next_state_C = State.get_next_state(state_C, data["Consumption_C"][(i+1)%l], data["Production_C"][(i+1)%l], data["Production_C"][(i+2)%l], data["Time"][(i+1)%l], mdp, action_A, action_B, action_C)
                
                # get best next expected reward (only from already explored states)
                max_next_A = mdp.get_best_next(Q_A[State2.get_id(next_state_A, mdp),:])
                max_next_B = mdp.get_best_next(Q_B[State2.get_id(next_state_B, mdp),:])
                max_next_C = mdp.get_best_next(Q_C[State2.get_id(next_state_C, mdp),:])

                # update Q-tables with Bellman equation
                Q_A[state_A_id, action_A_id] = (1-lr) * Q_A[state_A_id, action_A_id] + lr*(reward_A + gamma*max_next_A - Q_A[state_A_id, action_A_id])
                Q_B[state_B_id, action_B_id] = (1-lr) * Q_B[state_B_id, action_B_id] + lr*(reward_B + gamma*max_next_B - Q_B[state_B_id, action_B_id])
                Q_C[state_C_id, action_C_id] = (1-lr) * Q_C[state_C_id, action_C_id] + lr*(reward_C + gamma*max_next_C - Q_C[state_C_id, action_C_id])

                # sum reward
                total_reward = total_reward + reward_A + reward_B + reward_C
                
                # all_rewards.append(reward)
                # chosen_actions.append(mdp.get_action_id(action))
                battery.append(state_A.battery)
                # states.append((current_state.consumption, current_state.production, current_state.battery, current_state.time))
                # states_id.append(State.get_id(current_state, mdp))
                # states.append(current_state)
                
                # move to next state
                state_A = next_state_A
                state_B = next_state_B
                state_C = next_state_C

            # update the exploration proba using exponential decay formula after each episode
            exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*e))
            rewards_per_episode.append(total_reward)
            print(e)
        
        return Q_A, Q_B,Q_C, rewards_per_episode, changed, all_rewards, battery #, all_rewards, chosen_actions, states_id, states, battery


class Baseline_MA3:
    
    def find_baseline(data, mdp):
        costs_A = []
        costs_B = []
        costs_C = []
        actions_A = []
        actions_B = []
        actions_C = []
        battery = []
        diffb = []
        diffb2 = []
        state_A = State(data["Consumption_A"][0], data["Production_A"][0], 6000, data["Production_A"][1] ,data["Time"][0], mdp)
        state_B = State(data["Consumption_B"][0], data["Production_B"][0], 6000, data["Production_B"][1] ,data["Time"][0], mdp)
        state_C = State(data["Consumption_C"][0], data["Production_C"][0], 6000, data["Production_C"][1] ,data["Time"][0], mdp)
        l = len(data["Consumption_A"])
        for i in range(1,l):
            action_A = Baseline_MA3.find_best_action(state_A, mdp)
            action_B = Baseline_MA3.find_best_action(state_B, mdp)
            action_C = Baseline_MA3.find_best_action(state_C, mdp)
            print(action_A, action_B, action_C)
            deltaA, deltaB, deltaC = State.check_actions(state_A, action_A, action_B, action_C, mdp)
            action_A = state_A.get_action_for_delta(deltaA, mdp)
            action_B = state_B.get_action_for_delta(deltaB, mdp)
            action_C = state_C.get_action_for_delta(deltaC, mdp)
            cost_A, cost_B, cost_C = Reward.get_cost(state_A, state_B, state_C, action_A, action_B, action_C, mdp)
            costs_A.append(cost_A)
            costs_B.append(cost_B)
            costs_C.append(cost_C)
            
            actions_A.append(action_A)
            actions_B.append(action_B)
            actions_C.append(action_C)
            

            battery.append(state_A.battery)
            diffb.append(data["Production_B"][i] - data["Consumption_B"][i])
            diffb2.append(state_B.p - state_B.c)
            state_A = state_A.get_next_state(data["Consumption_A"][i], data["Production_A"][i],data["Production_A"][(i+1)%l], data["Time"][i], mdp, action_A, action_B, action_C)
            state_B = state_B.get_next_state(data["Consumption_B"][i], data["Production_B"][i],data["Production_B"][(i+1)%l], data["Time"][i], mdp, action_A, action_B, action_C)
            state_C = state_C.get_next_state(data["Consumption_C"][i], data["Production_C"][i],data["Production_C"][(i+1)%l], data["Time"][i], mdp, action_A, action_B, action_C)
            
        return costs_A,costs_B,costs_C, actions_A, actions_B, actions_C, battery, diffb, diffb2
    
    def find_best_action(state, mdp):
        print(state.p - state.c)
        if state.p - state.c >= mdp.charge_high and state.battery + mdp.charge_high <= mdp.max_battery:
            action = "charge_high"
            print("charge_high")
        elif state.p - state.c >= mdp.charge_low and state.battery + mdp.charge_low <= mdp.max_battery:
            action = "charge_low"
            print("charge_low")
        elif (state.c - state.p) >= mdp.discharge_low and state.battery - mdp.discharge_high >= 0:
            action = "discharge_high"
            print("discharge_high")
        elif (state.c - state.p) > 0 and state.battery - mdp.discharge_low >= 0:
            action = "discharge_low"
            print("discharge_low")
        else:
            action = "do nothing"
            print("do nothing")
        return action