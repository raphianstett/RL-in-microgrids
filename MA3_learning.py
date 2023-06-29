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
            
            
        for e in range(n_episodes):
            
            #sum the rewards that the agent gets from the environment
            total_reward = 0
            
            for i in range(0, len(data["Consumption_A"])): 
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
                action_A,reward_A,action_B, reward_B, action_C, reward_C =  Reward.get_reward(state_A, state_B, state_C, action_A, action_B, action_C, mdp)
                
                all_rewards.append(reward_A)
                all_rewards.append(reward_B)
                all_rewards.append(reward_C)
                
                action_A_id = mdp.get_action_id(action_A)
                changed = changed + 1 if prev_id_A != action_A_id else changed
                action_B_id = mdp.get_action_id(action_B)
                action_C_id = mdp.get_action_id(action_C)

                # reward_B = Reward.get_reward(state_B, action_B,mdp)
                
                next_state_A = State2.get_next_state(state_A, action_A, data["Consumption_A"][(i+1)%len(data["Consumption_A"])], data["Production_A"][(i+1)%len(data["Consumption_A"])], data["Production_A"][(i+2)%len(data["Consumption_A"])], data["Time"][(i+1)%len(data["Consumption_A"])], mdp)
                next_state_B = State2.get_next_state(state_B, action_B, data["Consumption_B"][(i+1)%len(data["Consumption_B"])], data["Production_B"][(i+1)%len(data["Consumption_B"])], data["Production_B"][(i+2)%len(data["Consumption_B"])], data["Time"][(i+1)%len(data["Consumption_B"])], mdp)
                next_state_C = State2.get_next_state(state_C, action_C, data["Consumption_C"][(i+1)%len(data["Consumption_C"])], data["Production_C"][(i+1)%len(data["Consumption_C"])], data["Production_C"][(i+2)%len(data["Consumption_C"])], data["Time"][(i+1)%len(data["Consumption_C"])], mdp)

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


