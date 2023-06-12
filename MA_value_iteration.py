import environment
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from environment import MDP
from environment import State
from MA_environment import Reward

from MA_data import Data

class MA_ValueIteration:
# define parameters for q-learning
    def value_iteration(data, n_episodes, mdp_A, mdp_B):
        print("####GO#####")

        #initialize the exploration probability to 1
        exploration_proba = 1

        #exploartion decreasing decay for exponential decreasing
        exploration_decreasing_decay = 5/n_episodes #4 / n_episodes

        # minimum of exploration proba
        min_exploration_proba = 0.01

        #discounted factor
        gamma = 0.8

        #learning rate
        lr = 0.8

        rewards_per_episode = []
        all_rewards = []
        chosen_actions = []
        battery = []
        states_id = []
        states = []
        

        # initialize Q-table
        Q_A = np.zeros((mdp_A.n_states, mdp_A.n_actions))
        Q_B = np.zeros((mdp_B.n_states, mdp_B.n_actions))
        
        # initialize the first state of the episode
        state_A = State(data["Consumption_A"][0], data["Production_A"][0], 2.0, data["Production_A"][1] ,data["Time"][0], mdp_A)
        state_B = State(data["Consumption_B"][0], data["Production_B"][0], 2.0, data["Production_B"][1] ,data["Time"][0], mdp_B)
            
            
        for e in range(n_episodes):
            
            #sum the rewards that the agent gets from the environment
            total_reward = 0
            
            for i in range(0, len(data["Consumption_A"])): 
            
                # exploration 
                if np.random.uniform(0,1) < exploration_proba:
                    action_A = mdp_A.action_space[np.random.randint(0,mdp_A.n_actions)]
                    action_B = mdp_B.action_space[np.random.randint(0,mdp_B.n_actions)]
                # exploitation
                else:
                    # choose best action for given state from Q-table
                    a_A = Q_A[State.get_id(state_A, mdp_A),:]
                    action_A = mdp_A.action_space[mdp_A.get_best_action(a_A)]
                    a_B = Q_B[State.get_id(state_B, mdp_B),:]
                    action_B = mdp_B.action_space[mdp_B.get_best_action(a_B)]
                
                # run the chosen action and return the next state and the reward for the action in the current state.
                reward_A = Reward.get_reward(action_A, action_B, state_A, state_B, mdp_A, mdp_B)
                reward_B = Reward.get_reward(action_B, action_A, state_B, state_A, mdp_B, mdp_A)
                
                next_state_A = State.get_next_state(state_A, action_A, data["Consumption_A"][(i+1)%len(data["Consumption_A"])], data["Production_A"][(i+1)%len(data["Consumption_A"])], data["Production_A"][(i+2)%len(data["Consumption_A"])], data["Time"][(i+1)%len(data["Consumption_A"])], mdp_A)
                next_state_B = State.get_next_state(state_B, action_B, data["Consumption_B"][(i+1)%len(data["Consumption_B"])], data["Production_B"][(i+1)%len(data["Consumption_B"])], data["Production_B"][(i+2)%len(data["Consumption_B"])], data["Time"][(i+1)%len(data["Consumption_B"])], mdp_B)

    
                # update Q-tables with Bellman equation
                Q_A[State.get_id(state_A, mdp_A), mdp_A.get_action_id(action_A)] = (1-lr) * Q_A[State.get_id(state_A, mdp_A), mdp_A.get_action_id(action_A)] + lr*(reward_A + gamma*max(Q_A[State.get_id(next_state_A, mdp_A),:]) - Q_A[State.get_id(state_A, mdp_A), mdp_A.get_action_id(action_A)])
                Q_B[State.get_id(state_B, mdp_B), mdp_B.get_action_id(action_B)] = (1-lr) * Q_B[State.get_id(state_B, mdp_B), mdp_B.get_action_id(action_B)] + lr*(reward_B + gamma*max(Q_B[State.get_id(next_state_B, mdp_B),:]) - Q_B[State.get_id(state_B, mdp_B), mdp_B.get_action_id(action_B)])
                

                # sum reward
                total_reward = total_reward + reward_A + reward_B
                
                # all_rewards.append(reward)
                # chosen_actions.append(mdp.get_action_id(action))
                # battery.append(current_state.battery)
                # states.append((current_state.consumption, current_state.production, current_state.battery, current_state.time))
                # states_id.append(State.get_id(current_state, mdp))
                # states.append(current_state)
                
                # move to next state
                state_A = next_state_A
                state_B = next_state_B

            # update the exploration proba using exponential decay formula after each episode
            exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*e))
            rewards_per_episode.append(total_reward)
            print(e)
        
        return Q_A, Q_B, rewards_per_episode #, all_rewards, chosen_actions, states_id, states, battery

data = Data()
df = data.get_data()

mdp_A = MDP(1000, 1000, 500, 500, 6, 5,5)
mdp_B = MDP(1000, 1000, 500, 500, 6, 5,5)

Q_A,Q_B, rewards_per_episode = MA_ValueIteration.value_iteration(df, 2, mdp_A, mdp_B)

print(rewards_per_episode)

