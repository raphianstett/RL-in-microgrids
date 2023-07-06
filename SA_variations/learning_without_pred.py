import numpy as np
from SA_variations.environment_without_pred import State

np.set_printoptions(threshold=np.inf)

# implementation of q-learning algorithm
class QLearning:
# define parameters for q-learning
    def iterate(data, n_episodes, lr, gamma, mdp):
        print("####GO#####")

        #initialize the exploration probability to 1
        exploration_proba = 1

        #exploartion decreasing decay for exponential decreasing
        exploration_decreasing_decay = 4/n_episodes #4 / n_episodes

        # minimum of exploration proba
        min_exploration_proba = 0.05


        rewards_per_episode = []
        all_rewards = []
        chosen_actions = []
        battery = []
        states_id = []
        states = []
        

        # initialize Q-table
        Q_table = np.zeros((mdp.n_states, mdp.n_actions))
        
        # initialize the first state of the episode
        current_state = State(data[0,0], data[0,1], 2000, data[0,2], mdp)
        l = data.shape[0]
            
        for e in range(n_episodes):
            state_id = int(State.get_id(current_state, mdp))
            #sum the rewards that the agent gets from the environment
            total_episode_reward = 0
            
            for i in range(l): 
                
                # exploration 
                if np.random.uniform(0,1) < exploration_proba:
                    action = mdp.action_space[np.random.randint(0,mdp.n_actions)]
                    
                # exploitation
                else:
                    # choose best action for given state from Q-table
                    a = Q_table[state_id,:]
                    action = mdp.action_space[mdp.get_best_action(a)]
                
                action_id = int(mdp.get_action_id(action))
               

                # run the chosen action and return the next state and the reward for the action in the current state.
                reward = State.get_reward(current_state, action, mdp)

                next_state = State.get_next_state(current_state, action, data[i,0], data[i,1],data[i,2], mdp)
                
                # get max expected future reward (only already explored states are included)
                max_next = mdp.get_best_next(Q_table[int(State.get_id(next_state, mdp)),:])

                # update Q-table with Bellman equation
                Q_table[state_id,action_id] = (1-lr) * Q_table[state_id,action_id] + lr* (reward + gamma*max_next - Q_table[state_id,action_id])
                
                # sum reward
                total_episode_reward = total_episode_reward + reward
                
                # move to next state
                current_state = next_state
                state_id = int(State.get_id(current_state, mdp))

            # update the exploration proba using exponential decay formula after each episode
            exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*e))
            rewards_per_episode.append(total_episode_reward)
            # print(e) if e%10 == 0 else None
        
        return Q_table, rewards_per_episode

class Baseline:
    def find_baseline_policy(data, mdp):
        rewards = []
        states = []
        actions = []
        
        battery = []
        current_state = State(data[0,1], data[0,2], 2000, data[1,2] ,data[0,3], mdp)
        l = data.shape[0]
        for i in range(1,l):
            
            if current_state.p - current_state.c >= 1000 and current_state.battery + mdp.charge_high <= mdp.max_battery:
                action = "charge_high"
                # print("charge_high")
            elif current_state.p - current_state.c >= mdp.charge_low and current_state.battery + mdp.charge_low <= mdp.max_battery:
                action = "charge_low"
                # print("charge_low")
            elif (current_state.c - current_state.p) >= mdp.discharge_low and current_state.battery - mdp.discharge_high >= 0:
                action = "discharge_high"
                # print("discharge_high")
            elif (current_state.c - current_state.p) > 0 and current_state.battery - mdp.discharge_low >= 0:
                action = "discharge_low"
                # print("discharge_low")
            else:
                action = "do nothing"
                # print("Do nothing")
            # print(current_state.battery)

            rewards.append(State.get_cost(current_state,action, mdp))
            actions.append(action)
    
            battery.append(current_state.battery)
            
            current_state = State.get_next_state(current_state, action, data[i,0], data[i,1],data[(i+1)%l,2] ,data[i,3], mdp)
           
        return rewards, states, actions, battery

