import environment
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from environment import MDP
from environment import State
from MA_environment import Reward

from MA_data import Data

class ValueIteration:
# define parameters for q-learning
    def value_iteration(data, n_episodes, mdp):
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
        Q_table = np.zeros((mdp.n_states, mdp.n_actions))
        
        # initialize the first state of the episode
        current_state = State(data["Consumption"][0], data["Production"][0], 2.0, data["Production"][1] ,data["Time"][0], mdp)
            
            
        for e in range(n_episodes):
            
            #sum the rewards that the agent gets from the environment
            total_episode_reward = 0
            
            for i in range(0, len(data["Consumption"])): 
            
                # exploration 
                if np.random.uniform(0,1) < exploration_proba:
                    action = mdp.action_space[np.random.randint(0,mdp.n_actions)]
                    
                # exploitation
                else:
                    # choose best action for given state from Q-table
                    a = Q_table[State.get_id(current_state, mdp),:]
                    action = mdp.action_space[mdp.get_best_action(a)]
                    
                
                # run the chosen action and return the next state and the reward for the action in the current state.
                reward = State.get_reward(action, current_state, mdp)

                next_state = State.get_next_state(current_state, action, data["Consumption"][(i+1)%len(data["Consumption"])], data["Production"][(i+1)%len(data["Consumption"])], data["Production"][(i+2)%len(data["Consumption"])], data["Time"][(i+1)%len(data["Consumption"])], mdp)

    
                # update Q-table with Bellman equation
                Q_table[State.get_id(current_state, mdp), mdp.get_action_id(action)] = (1-lr) * Q_table[State.get_id(current_state, mdp), mdp.get_action_id(action)] + lr*(reward + gamma*max(Q_table[State.get_id(next_state, mdp),:]) - Q_table[State.get_id(current_state, mdp), mdp.get_action_id(action)])
                
                # sum reward
                total_episode_reward = total_episode_reward + reward
                
                all_rewards.append(reward)
                chosen_actions.append(mdp.get_action_id(action))
                battery.append(current_state.battery)
                states.append((current_state.consumption, current_state.production, current_state.battery, current_state.time))
                states_id.append(State.get_id(current_state, mdp))
                # states.append(current_state)
                
                # move to next state
                current_state = next_state

            # update the exploration proba using exponential decay formula after each episode
            exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*e))
            rewards_per_episode.append(total_episode_reward)
            print(e)
        
        return Q_table, rewards_per_episode, all_rewards, chosen_actions, states_id, states, battery
