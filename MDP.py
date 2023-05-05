# file to create MDP for reinforcement learning problem


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import time


class MDP:
    # construct state space

    # getters for data discretization
    def get_consumption(c):
        return "low" if c < 250  else  ("average" if (c > 250) and (c < 360) else ("high"))

    def get_production(p):
        return "none" if p == 0  else  ("low" if (p > 0) and (p < 1200) else ("high"))


    # state space

    # steps
    consumption = ["low", "average", "high"]
    production = ["none", "low", "high"]

    max_battery = 4
    battery = [*range(0,max_battery+1,1)]
    time = [*range(0,24,1)]


    # dimensions
    n_consumption = len(consumption)
    n_production = len(production)
    n_battery = len(battery)

    n_time = len(time)
    n_states = n_consumption * n_production * n_battery * n_time




    # action space  
    action_space = ["discharge", "do nothing", "charge"]
    n_actions = len(action_space)
    max_discharge = 1000
    max_charge = 1000

    def get_action_id(action):
        return MDP.action_space.index(action)

    # reward function
    max_loss = -10000


    # total cost function after applying learned policy
    def get_total_costs(rewards):
            rewards = np.array(rewards)
            rewards = [0 if x > 0 else x for x in rewards]
            
            return np.sum(rewards)


    def apply_q_table(Q_table, dat):
        rewards = []
        actions = []
        current_state = State(dat["Consumption"][0], dat["Production"][0], 2, dat["Time"][0])
        
        for i in range(len(dat["Consumption"])):
            if i == len(dat["Consumption"])-1:
                break
            # print("iteration: " + str(i) + "time: " + str(dat["Time"][i]))
            action = MDP.action_space[np.argmax(Q_table[State.get_id(current_state),:])]
            # print("State: " + str(State.get_id(current_state)))
            # print("Action ID: " + str(MDP.get_action_id(action)))
            # print("Q table: " + str(Q_table[State.get_id(current_state),]))
            rewards.append(State.get_cost(action, current_state))
            actions.append(MDP.get_action_id(action))
            current_state = State.get_next_state(current_state, action, dat["Consumption"][i+1], dat["Production"][i+1], dat["Time"][i+1])
        
        return rewards, actions

    def iterate_q(Q_table):
        actions = []
        for i in range(Q_table[1]):
            action = MDP.action_space[np.argmax(Q_table[i,:])]
            actions.append(MDP.get_action_id(action))


# class which constructs state
class State:
    def __init__(self, c, p, battery, time):
        self.consumption = MDP.get_consumption(c)
        self.production = MDP.get_production(p)
        self.battery = battery
        self.time = time
        self.c = c
        self.p = p
    

    def get_next_state(self, action, new_c, new_p, new_time):
        
        # update battery state based on chosen action
        
        if action == "discharge" and self.battery > 0:
            next_battery = self.battery - 1
        elif action == "charge" and self.battery < MDP.max_battery:
            next_battery = self.battery + 1
        else:
            next_battery = self.battery

        next_state = State(new_c, new_p, next_battery, new_time)

        return next_state
    
    def get_id(state):
        # consumption
        c = 0 if (state.consumption == "low") else (1 if (state.consumption == "average") else 2)
        p = 0 if (state.production == "none") else (1 if (state.consumption == "low") else 2)
     
        return c * (MDP.n_production*MDP.n_battery*24) + p *(MDP.n_battery*24) + state.battery * 24 + state.time

        

    def get_reward(action, state):
    
        # max_loss if action not possible, else calculate reward as difference between production and consumption
        # based on chosen action 
        if(action == "charge"):
            
            if state.battery == MDP.max_battery or (state.p - state.c) < 1000 :
                return MDP.max_loss 
            else:
                return 0 # state.p - (max_charge + state.c)
                    
        if(action == "discharge"):
            if state.p > state.c or state.battery == 0:
                return MDP.max_loss
            else:
                return ((state.p + MDP.max_discharge)  - state.c) 
        if(action == "do nothing"):
            # punish not doing something, if possible and reasonable
            if state.p - state.c > 1000 and state.battery < MDP.max_battery or state.c - state.p and state.battery > 0:
                return MDP.max_loss
            else:
                return state.p - state.c if state.p < state.c else 0
        
    def get_cost(action, state):
        if action == "charge":
            return 0
        if action == "discharge":
            return ((state.p + MDP.max_discharge)  - state.c) if ((state.p + MDP.max_discharge)  < state.c) else 0
        if action == "do nothing":
            return state.p - state.c if state.p < state.c else 0

        

    
## HELPER FUNCTIONS FOR TESTING
def get_last_x(states, x):
    lst = [0]*x
    for i in range(x):
        lst[-i] = states[-i]
    return lst

def get_battery_from_lst(lst):
    bat = []
    bat.append([x[2] for x in lst])
    return bat




