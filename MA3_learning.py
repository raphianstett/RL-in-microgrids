import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from MA3_environment import Reward
from MA3_environment import State


class MA3_QLearning:
# define parameters for q-learning
    def iterate(data, n_episodes, mdp):
        print("Start Training")

        exploration_proba = 1
        exploration_decreasing_decay = 4/n_episodes #4 / n_episodes
        min_exploration_proba = 0.05
        gamma = 0.8
        lr = 0.5
        rewards_per_episode = []
    
        Q_A = np.zeros((mdp.n_states, mdp.n_actions))
        Q_B = np.zeros((mdp.n_states, mdp.n_actions))
        Q_C = np.zeros((mdp.n_states, mdp.n_actions_c))
        
        state_A = State(data[0,0], data[0,3], 2000, data[1,3] ,data[0,7], mdp)
        state_B = State(data[0,1], data[0,4], 2000, data[1,4] ,data[0,7], mdp)
        state_C = State(data[0,2], data[0,5], 2000, data[1,5] ,data[0,7], mdp)
        
        l = data.shape[0]
        for e in range(n_episodes):
            
            total_reward = 0
            
            for i in range(0, l): 
               
                state_A_id = int(State.get_id(state_A, mdp))
                state_B_id = int(State.get_id(state_B, mdp))
                state_C_id = int(State.get_id(state_C, mdp))
                # exploration 
                if np.random.uniform(0,1) < exploration_proba:
                    action_A = mdp.action_space[np.random.randint(0,mdp.n_actions)]
                    action_B = mdp.action_space[np.random.randint(0,mdp.n_actions)]
                    action_C = mdp.action_space_c[np.random.randint(0, mdp.n_actions_c)]
                # exploitation
                else:
                    a_A = Q_A[state_A_id,:]
                    action_A = mdp.action_space[mdp.get_best_action(a_A)]
                    a_B = Q_B[state_B_id,:]
                    action_B = mdp.action_space[mdp.get_best_action(a_B)]
                    a_C = Q_C[state_C_id,:]
                    action_C = mdp.action_space_c[mdp.get_best_action(a_C)]
                    
                action_A_id = int(mdp.get_action_id(action_A))
                action_B_id = int(mdp.get_action_id(action_B))
                action_C_id = int(mdp.get_action_id(action_C))
                
                
                reward_A, reward_B, reward_C =  Reward.get_reward(state_A, state_B, state_C, action_A, action_B, action_C, mdp)

                # reward_B = Reward.get_reward(state_B, action_B,mdp)
                
                next_state_A = state_A.get_next_state(data[i,0], data[i,3], data[(i+1)%l,3] ,data[i,7],mdp, action_A, action_B, action_C)
                next_state_B = state_B.get_next_state(data[i,1], data[i,4], data[(i+1)%l,4] ,data[i,7],mdp, action_A, action_B, action_C)
                next_state_C = state_C.get_next_state(data[i,2], data[i,5], data[(i+1)%l,5] ,data[i,7],mdp, action_A, action_B, action_C)

                # get best next expected reward (only from already explored states)
                max_next_A = mdp.get_best_next(Q_A[int(State.get_id(next_state_A, mdp)),:])
                max_next_B = mdp.get_best_next(Q_B[int(State.get_id(next_state_B, mdp)),:])
                max_next_C = mdp.get_best_next(Q_C[int(State.get_id(next_state_C, mdp)),:])

                # update Q-tables with Bellman equation
                Q_A[state_A_id, action_A_id] = (1-lr) * Q_A[state_A_id, action_A_id] + lr*(reward_A + gamma*max_next_A - Q_A[state_A_id, action_A_id])
                Q_B[state_B_id, action_B_id] = (1-lr) * Q_B[state_B_id, action_B_id] + lr*(reward_B + gamma*max_next_B - Q_B[state_B_id, action_B_id])
                Q_C[state_C_id, action_C_id] = (1-lr) * Q_C[state_C_id, action_C_id] + lr*(reward_C + gamma*max_next_C - Q_C[state_C_id, action_C_id])

                total_reward = total_reward + reward_A + reward_B + reward_C
                
                # move to next state
                state_A = next_state_A
                state_B = next_state_B
                state_C = next_state_C

            exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*e))
            rewards_per_episode.append(total_reward)
            print(e) if e%100 == 0 else None
        
        return Q_A, Q_B,Q_C, rewards_per_episode 


class Baseline_MA3:
    def find_baseline(data, mdp):
        costs_A = []
        costs_B = []
        costs_C = []
        actions_A = []
        actions_B = []
        actions_C = []
        battery = []
        conflicts = 0

        state_A = State(data[0,0], data[0,3], 2000, data[1,3] ,data[0,7], mdp)
        state_B = State(data[0,1], data[0,4], 2000, data[1,4] ,data[0,7], mdp)
        state_C = State(data[0,2], data[0,5], 2000, data[1,5] ,data[0,7], mdp)
        l = data.shape[0]

        for i in range(1,l):
            action_A = Baseline_MA3.find_best_action(state_A, mdp)
            action_B = Baseline_MA3.find_best_action(state_B, mdp)
            action_C = Baseline_MA3.find_best_action(state_C, mdp)
            action_A_old = action_A
            action_B_old = action_B
            action_C_old = action_C
            
            deltaA, deltaB, deltaC = State.check_actions(state_A, action_A, action_B, action_C, mdp)
            
            action_A = state_A.get_action_for_delta(deltaA, mdp)
            action_B = state_B.get_action_for_delta(deltaB, mdp)
            action_C = state_C.get_action_for_delta(deltaC, mdp)

            conflicts = conflicts + 1 if action_A != action_A_old or action_B != action_B_old or action_C != action_C_old else conflicts
            cost_A, cost_B, cost_C = Reward.get_cost(state_A, state_B, state_C, action_A, action_B, action_C, mdp)
            
            costs_A.append(cost_A)
            costs_B.append(cost_B)
            costs_C.append(cost_C)
            
            actions_A.append(action_A)
            actions_B.append(action_B)
            actions_C.append(action_C)
            
            battery.append(state_A.battery)
            
            state_A = State.get_next_state(state_A,data[i,0], data[i,3], data[(i+1)%l,3] ,data[i,7], mdp, action_A, action_B, action_C)
            state_B = State.get_next_state(state_B,data[i,1], data[i,4], data[(i+1)%l,4] ,data[i,7], mdp, action_A, action_B, action_C)
            state_C = State.get_next_state(state_C,data[i,2], data[i,5], data[(i+1)%l,5] ,data[i,7], mdp, action_A, action_B, action_C)

        return costs_A,costs_B,costs_C, actions_A, actions_B, actions_C, battery, conflicts
    
    def find_best_action(state, mdp):
        if state.p - state.c >= mdp.charge_high and state.battery + mdp.charge_high <= mdp.max_battery:
            action = "charge_high"
        elif state.p - state.c >= mdp.charge_low and state.battery + mdp.charge_low <= mdp.max_battery:
            action = "charge_low"
        elif (state.c - state.p) >= mdp.discharge_low and state.battery - mdp.discharge_high >= 0:
            action = "discharge_high"
        elif (state.c - state.p) > 0 and state.battery - mdp.discharge_low >= 0:
            action = "discharge_low"
        else:
            action = "do nothing"
        return action