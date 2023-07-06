import numpy as np
from MA2_environment_diff import State
from MA2_environment import Reward
from MA2_environment_diff import StateTransition


class MA_QLearning:
# define parameters for q-learning
    def iterate(data, n_episodes, mdp_A, mdp_B, lr, gamma):
        print("####GO#####")

        #initialize the exploration probability to 1
        exploration_proba = 1

        #exploartion decreasing decay for exponential decreasing
        exploration_decreasing_decay = 4/n_episodes #4 / n_episodes

        # minimum of exploration proba
        min_exploration_proba = 0.01

        rewards_per_episode = []

        # initialize Q-table
        Q_A = np.zeros((mdp_A.n_states, mdp_A.n_actions))
        Q_B = np.zeros((mdp_B.n_states, mdp_B.n_actions))
        
        # initialize the first state of the episode
       
        state_A = State(data[0,0], data[0,2], 2000, data[0,4], mdp_A)
        state_B = State(data[0,1], data[0,3], 2000 ,data[0,4], mdp_A)
        l = data.shape[0]
            
        for e in range(n_episodes):
            
            #sum the rewards that the agent gets from the environment
            total_reward = 0
            
            for i in range(1,l): 
                state_A_id = int(State.get_id(state_A, mdp_A))
                state_B_id = int(State.get_id(state_B, mdp_B))
                # exploration 
                if np.random.uniform(0,1) < exploration_proba:
                    action_A = mdp_A.action_space[np.random.randint(0,mdp_A.n_actions)]
                    action_B = mdp_B.action_space[np.random.randint(0,mdp_B.n_actions)]
                # exploitation
                else:
                    # choose best action for given state from Q-table
                    a_A = Q_A[state_A_id,:]
                    action_A = mdp_A.action_space[mdp_A.get_best_action(a_A)]
                    a_B = Q_B[state_B_id,:]
                    action_B = mdp_B.action_space[mdp_B.get_best_action(a_B)]
                
                action_A_id = int(mdp_A.get_action_id(action_A))
                action_B_id = int(mdp_B.get_action_id(action_B))
                
                # run the chosen action and return the next state and the reward for the action in the current state.
                reward_A = Reward.get_reward(state_A, state_B, action_A, action_B, mdp_A, mdp_B)
                reward_B = Reward.get_reward(state_B, state_A, action_B, action_A, mdp_B, mdp_A)
                
                next_state_A = StateTransition.get_next_state(state_A, action_A, data[i,0], data[i,2], data[i,4], mdp_A)
                next_state_B = StateTransition.get_next_state(state_B, action_B, data[i,1], data[i,3], data[i,4], mdp_B)

                # get max expected future reward (only already explored states are included)
                max_next_A = mdp_A.get_best_next(Q_A[int(State.get_id(next_state_A, mdp_A)),:])
                max_next_B = mdp_B.get_best_next(Q_B[int(State.get_id(next_state_B, mdp_B)),:])
              
                # update Q-tables with Bellman equation
                Q_A[state_A_id, action_A_id] = (1-lr) * Q_A[state_A_id, action_A_id] + lr*(reward_A + gamma*max_next_A - Q_A[state_A_id, action_A_id])
                Q_B[state_B_id, action_B_id] = (1-lr) * Q_B[state_B_id, action_B_id] + lr*(reward_B + gamma*max_next_B - Q_B[state_B_id, action_B_id])

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
            print(e) if e% 10 == 0 else None
        
        return Q_A, Q_B, rewards_per_episode #, all_rewards, chosen_actions, states_id, states, battery



