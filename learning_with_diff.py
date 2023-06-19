import numpy as np
from environment_with_diff import State


np.set_printoptions(threshold=np.inf)

# implementation of q-learning algorithm
class QLearning:
# define parameters for q-learning
    def iterate(data, n_episodes, mdp):
        print("####GO#####")

        #initialize the exploration probability to 1
        exploration_proba = 1

        #exploartion decreasing decay for exponential decreasing
        exploration_decreasing_decay = 10/n_episodes #4 / n_episodes

        # minimum of exploration proba
        min_exploration_proba = 0.02

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
        current_state = State(data["Consumption"][0], data["Production"][0], 20, data["Production"][1] ,data["Time"][0], mdp)
        l = len(data["Consumption"])
            
        for e in range(n_episodes):
            state_id = State.get_id(current_state, mdp)
            #sum the rewards that the agent gets from the environment
            total_episode_reward = 0
            
            for i in range(0, l): 
                
                # exploration 
                if np.random.uniform(0,1) < exploration_proba:
                    action = mdp.action_space[np.random.randint(0,mdp.n_actions)]
                    
                # exploitation
                else:
                    # choose best action for given state from Q-table
                    a = Q_table[state_id,:]
                    action = mdp.action_space[mdp.get_best_action(a)]
                
                action_id = mdp.get_action_id(action)
                

                # run the chosen action and return the next state and the reward for the action in the current state.
                reward = State.get_reward(current_state, action, mdp)

                next_state = State.get_next_state(current_state, action, data["Consumption"][(i+1)%l], data["Production"][(i+1)%l], data["Production"][(i+2)%l], data["Time"][(i+1)%l], mdp)

                # update Q-table with Bellman equation
                Q_table[state_id,action_id] = (1-lr) * Q_table[state_id,action_id] + lr*(reward + gamma*max(Q_table[State.get_id(next_state, mdp),:]) - Q_table[state_id,action_id])
                
                # sum reward
                total_episode_reward = total_episode_reward + reward
                
                # all_rewards.append(reward)
                chosen_actions.append(mdp.get_action_id(action))
                battery.append(current_state.battery)
                # states.append((current_state.consumption, current_state.production, current_state.battery, current_state.time))
                # states_id.append(State.get_id(current_state, mdp))
                # states.append(current_state)
                
                # move to next state
                current_state = next_state
                state_id = State.get_id(current_state, mdp)

            # update the exploration proba using exponential decay formula after each episode
            exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*e))
            rewards_per_episode.append(total_episode_reward)
            print(e)
        
        return Q_table, rewards_per_episode, all_rewards, chosen_actions, states_id, states, battery


class Baseline:
    def find_baseline_policy(data, mdp):
        rewards = []
        states = []
        actions = []
        diffs = []
        battery = []
        current_state = State(data["Consumption"][0], data["Production"][0], 20,data["Production"][1], data["Time"][0], mdp)
        
        for i in range(1,len(data["Consumption"])):
            
            if current_state.p - current_state.c >= 1000 and current_state.battery + mdp.step_high_charge <= mdp.max_battery:
                action = "charge_high"
                # print("charge_high")
            elif current_state.p - current_state.c >= mdp.charge_low and current_state.battery + mdp.step_low_charge <= mdp.max_battery:
                action = "charge_low"
                # print("charge_low")
            elif (current_state.c - current_state.p) >= mdp.discharge_low and current_state.battery - mdp.step_high_discharge >= 0:
                action = "discharge_high"
                # print("discharge_high")
            elif (current_state.c - current_state.p) > 0 and current_state.battery - mdp.step_low_discharge >= 0:
                action = "discharge_low"
                # print("discharge_low")
            else:
                action = "do nothing"
                # print("Do nothing")
            # print(current_state.battery)

            rewards.append(State.get_cost(current_state,action, mdp))
            states.append((current_state.consumption, current_state.production, current_state.battery,current_state.time))
            actions.append(action)
            
            battery.append(current_state.battery)
            diffs.append((current_state.c - current_state.p) >= mdp.max_discharge and current_state.battery - mdp.step_high_discharge >= 0)

            current_state = State.get_next_state(current_state, action, data["Consumption"][i], data["Production"][i],data["Production"][(i+1)%len(data["Production"])], data["Time"][i], mdp)
            # print(current_state.consumption, current_state.production, current_state.battery,current_state.time)

        return rewards, states, actions, battery, diffs

