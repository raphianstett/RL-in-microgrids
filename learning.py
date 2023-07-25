import numpy as np
from environment import State
from environment import Reward

# implementation of Q-Learning algorithm
class QLearning:

    def iterate(data, n_episodes, lr, gamma,epsilon, mdp):
        print("Start Training")

        # initialize the exploration probability to 1
        exploration_proba = 1

        # exploration decreasing decay function for exponential decreasing epsilon
        exploration_decreasing_decay = epsilon/n_episodes

        # minimum of exploration proba
        min_exploration_proba = 0.05

        rewards_per_episode = []
        
        # initialize Q-table with zeros
        Q_table = np.zeros((mdp.n_states, mdp.n_actions))
        
        # initialize the first state of the episode, first data point and initial ESS state = 2kWh
        current_state = State(data[0,0], data[0,1], 2000, data[1,1] ,data[0,2], mdp)
        
        l = data.shape[0]
            
        for e in range(n_episodes):
            state_id = int(State.get_id(current_state, mdp))
    
            total_episode_reward = 0
            
            # loop over dataset
            for i in range(0, l): 
                
                # exploration, select random action
                if np.random.uniform(0,1) < exploration_proba:
                    action = mdp.action_space[np.random.randint(0,mdp.n_actions)]
                    
                # exploitation
                else:
                    # choose best action for given state from Q-table
                    a = Q_table[state_id,:]
                    action = mdp.action_space[mdp.get_best_action(a)]
                
                action_id = int(mdp.get_action_id(action))
               
                # execute the chosen action to receive reward and move to next state
                reward = Reward.get_reward(current_state, action, mdp)

                next_state = State.get_next_state(current_state, action, data[i,0], data[i,1],data[(i+1)%l,1] ,data[i,2], mdp)
                
                # get maximum expected future rewards from already explored states
                max_next = mdp.get_best_next(Q_table[int(State.get_id(next_state, mdp)),:])

                # update Q-table with Bellman equation
                Q_table[state_id,action_id] = (1-lr) * Q_table[state_id,action_id] + lr* (reward + gamma*max_next - Q_table[state_id,action_id])
                
                # sum reward
                total_episode_reward += reward

                # move to next state
                current_state = next_state
                state_id = int(State.get_id(current_state, mdp))

            # update the exploration proba using exponential decay formula after each episode
            exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*e))
            rewards_per_episode.append(total_episode_reward)
            print(e) if e % 100 == 0  else None 
        
        return Q_table, rewards_per_episode

class Baseline:
    def find_baseline_policy(data, mdp):
        rewards = []
        states = []
        actions = []
        battery = []
        current_state = State(data[0,1], data[0,2], 2000, data[1,2] ,data[0,3], mdp)
        l = data.shape[0]

        # action selection based on rules
        for i in range(0,l):
            if current_state.p - current_state.c >= mdp.charge_high and current_state.battery + mdp.charge_high <= mdp.max_battery:
                action = "charge_high"
            elif current_state.p - current_state.c >= mdp.charge_low and current_state.battery + mdp.charge_low <= mdp.max_battery:
                action = "charge_low"
            elif (current_state.c - current_state.p) >= mdp.discharge_low and current_state.battery - mdp.discharge_high >= 0:
                action = "discharge_high"
            elif (current_state.c - current_state.p) > 0 and current_state.battery - mdp.discharge_low >= 0:
                action = "discharge_low"
            else:
                action = "do nothing"

            rewards.append(Reward.get_cost(current_state,action, mdp))
            actions.append(action)
            battery.append(current_state.battery)

            if i == l-1:
                continue
            else:
                current_state = State.get_next_state(current_state, action, data[i+1,0], data[i+1,1],data[(i+2)%l,2] ,data[i+1,3], mdp)

        return rewards, states, actions, battery

