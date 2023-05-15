# file to create MDP for reinforcement learning problem


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import random
import time

from data import RealData
from collections import Counter

class MDP:
    # construct state space

    # use this, for dynamic setting of bins
    # dat = RealData.get_real_data()
    # bins_cons = RealData.get_bin_boundaries(dat['Consumption'], 5)
    # bins_prod = RealData.get_bin_boundaries(RealData.get_prod_nonzeros(dat['Production']))
    
    # # getters for data discretization
    # bins: (0-250, 250-360, >360)
    def get_consumption_three(c):
        return "low" if c < 250  else  ("average" if (c > 250) and (c < 360) else ("high"))

    # bins: [0.  217.  266.  339.  424. 2817.]
    def get_consumption(p):
        prod = ["very low", "low", "average", "high", "very high"]
        intervals = [pd.Interval(left = 0, right = 215, closed = 'both'),pd.Interval(left = 215,right = 270, closed = 'right'), pd.Interval(left = 270,right =  340, closed = 'right'), pd.Interval(left = 340,right =  430, closed = 'right'), pd.Interval(left = 430,right =  2900, closed = 'right')]
        return prod[0] if p in intervals[0] else (prod[1] if p in intervals[1] else (prod[2] if p in intervals[2] else (prod[3] if p in intervals[3] else prod[4]))) 

    # def get_consumption

    # with 3 bins: (0, 0-1200, >1200)
    def get_production_three(p):
        return "none" if p == 0  else  ("low" if (p > 0) and (p < 1200) else ("high"))
    # with 5 bins: (0, 0-330, 330-1200, 1200 - 3200, >3200), each bin with equal frequency
    def get_production(p):
        prod = ["none", "low", "average", "high", "very high"]
        intervals = [pd.Interval(left = 0, right = 0, closed = 'both'),pd.Interval(left = 0,right = 330, closed = 'right'), pd.Interval(left = 330,right =  1200, closed = 'right'), pd.Interval(left = 1200,right =  3200, closed = 'right'), pd.Interval(left = 3200,right =  7000, closed = 'right')]
        return prod[0] if p in intervals[0] else (prod[1] if p in intervals[1] else (prod[2] if p in intervals[2] else (prod[3] if p in intervals[3] else prod[4]))) 


    # function to add noise for prediction of production
    def get_predict_prod(p):
        mu, sigma = 1, 0.2
        rand = np.random.normal(mu, sigma, 1)
        return int(rand * p) 
    
    # function to return action_id with largest Q-value (!=0)
    # returns 1 (do nothing), if all q-values are the same
    
    def get_best_action(q_values):
        q_values[q_values == 0] = min(q_values)
        return np.argmax(q_values) if max(q_values) != 0 else 2 
    
    # state space

    # steps
    consumption = ["very low", "low", "average", "high", "very high"]
    production = ["none", "low","average", "high", "very high"]

    # battery
    max_battery = 6
    # battery = [*range(0,max_battery+1,0.5)]
    step_size = 0.5 
    battery = list(np.arange(0.0, float(max_battery) + step_size, step_size))

    # time
    time = [*range(0,24,1)]


    # dimensions
    n_consumption = len(consumption)
    n_production = len(production)
    n_pred_production = len(production)
    n_battery = len(battery)

    n_time = len(time)
    n_states = n_consumption * n_production * n_battery * n_time * n_pred_production
    
    # action space  
    action_space = ["discharge_high", "discharge_low", "do nothing","charge_low", "charge"]
    n_actions = len(action_space)
    max_discharge = 1000
    discharge_low = 500
    max_charge = 1000
    charge_low = 500

    def get_action_id(action):
        return MDP.action_space.index(action)

    def get_battery_id(battery):
        return int(battery * 2) 

    # reward function
    max_loss = -999999999999999999


    # total cost function after applying learned policy
    def get_total_costs(rewards):
            rewards = np.array(rewards)
            rewards = [0 if x > 0 else x for x in rewards]
            
            return np.sum(rewards)

                    

    def apply_q_table(Q_table, dat):
        rewards = []
        actions = []
        battery = []
        current_state = State(dat["Consumption"][0], dat["Production"][0], 2.0, dat["Production"][1], dat["Time"][0])
        
        for i in range(len(dat["Consumption"])):
            if i == len(dat["Consumption"])-1:
                break
            # print("iteration: " + str(i) + "time: " + str(dat["Time"][i]))
            action = MDP.action_space[MDP.get_best_action(Q_table[State.get_id(current_state),:])]
            # print("State: " + str(State.get_id(current_state))
            # print("Action ID: " + str(MDP.get_action_id(action)))
            # print("Q table: " + str(Q_table[State.get_id(current_state),]))
            rewards.append(State.get_cost(action, current_state))
            actions.append(MDP.get_action_id(action))
            battery.append(current_state.battery)
            current_state = State.get_next_state(current_state, action, dat["Consumption"][(i+1)%len(dat["Consumption"])], dat["Production"][(i+1)%len(dat["Consumption"])], dat["Production"][(i+2)%len(dat["Consumption"])], dat["Time"][(i+1)%len(dat["Consumption"])])
        
        return rewards, actions, battery

    def iterate_q(Q_table):
        actions = []
        for i in range(Q_table[1]):
            action = MDP.action_space[np.argmax(Q_table[i,:])]
            actions.append(MDP.get_action_id(action))


# class which constructs state
# Consumption, Production, Battery state, time, prediction of production in text timestep
class State:
    def __init__(self, c, p, battery, p_next, time):
        self.consumption = MDP.get_consumption(c)
        self.production = MDP.get_production(p)
        self.battery = battery
        self.time = time
        self.c = c
        self.p = p
        self.pred = MDP.get_predict_prod(p_next)
        self.predicted_prod = MDP.get_production(self.pred)
    

    def get_next_state(self, action, new_c, new_p,  new_pred, new_time):
        
        # update battery state based on chosen action
        
        if action == "discharge_high" and self.battery > 0.5:
            next_battery = self.battery - 1.0
        elif action == "discharge_low" and self.battery > 0:
            next_battery = self.battery - MDP.step_size
        elif action == "charge" and self.battery <= MDP.max_battery - 1:
            next_battery = self.battery + 1.0
        elif action == "charge_low" and self.battery < MDP.max_battery:
            next_battery = self.battery + MDP.step_size
        else:
            next_battery = self.battery

        next_state = State(new_c, new_p, next_battery, new_pred, new_time)

        return next_state
    
    def get_id(state):
        # consumption
        c = 0 if (state.consumption == "very low") else (1 if (state.consumption == "low") else (2 if (state.consumption == "average") else (3 if (state.consumption == "high") else 4)))
        # production
        p = 0 if (state.production == "none") else (1 if (state.production == "low") else (2 if (state.production == "average") else (3 if (state.production == "high") else 4)))
        # predicted production
        pred = 0 if (state.predicted_prod == "none") else (1 if (state.predicted_prod == "low") else (2 if (state.predicted_prod == "average") else (3 if (state.predicted_prod == "high") else 4)))
     
        return c * (MDP.n_production*MDP.n_battery*MDP.n_pred_production*24) + p *(MDP.n_battery*MDP.n_pred_production*24) + MDP.get_battery_id(state.battery) * (MDP.n_pred_production*24) + pred * 24 + state.time

        

    def get_reward(action, state):
        
        f_char = (state.p / state.pred) if state.p != 0 and state.pred != 0 else 1
        f_char = min(f_char, 2)
        f_dischar = (state.pred / state.p) if state.p != 0 and state.pred != 0 else 1
        f_dischar = min(f_dischar, 2)

        f_nothing = min(max((state.p / state.c),(state.c / state.p)),100) if (state.p != 0 and state.c != 0) else 1

        # max_loss if action not possible, else calculate reward as squared difference between production and consumption
        # based on chosen action 
        if action == 'charge':
            
            if state.battery >= MDP.max_battery - MDP.step_size or (state.p - state.c) < MDP.max_charge :
                return MDP.max_loss 
            else:
                return  -((MDP.max_battery+state.battery) / MDP.max_battery)*(state.p - (MDP.max_charge + state.c))**2
        if action == 'charge_low':
            if state.battery == MDP.max_battery or state.p - state.c < MDP.charge_low:
                return MDP.max_loss
            else:
                return -((MDP.max_battery+state.battery) / MDP.max_battery)*(state.p - (MDP.charge_low + state.c))**2
        if action == "discharge_high" :
            #if state.p > state.c or state.battery == 0:
            if state.battery < 1.0:
                return MDP.max_loss
            else:
                # discharging is also good, if next production is larger than current
                return -(MDP.max_battery / (state.battery + MDP.max_battery))*((state.p + MDP.max_discharge)  - state.c)**2 #if (state.p + MDP.max_discharge) < state.c else 0
        if action == 'discharge_low':
            if state.battery < 0.5:
                return MDP.max_loss
            else:
                # discharging is also good, if next production is larger than current
                return -(MDP.max_battery / (state.battery + MDP.max_battery))*((state.p + MDP.discharge_low)  - state.c)**2
        if action == "do nothing":
            # punish not doing something, if possible and reasonable
            # if state.p - state.c > 1000 and state.battery < MDP.max_battery or state.c - state.p >= 1000 and state.battery > 0:
            #     return MDP.max_loss
            #else:
            return -max((MDP.max_battery / (state.battery + MDP.max_battery)), (state.battery + MDP.max_battery / (MDP.max_battery)))\
                *(state.p - state.c)**2 

     
        
    def get_cost(action, state):
        if action == "charge" and state.battery == MDP.max_battery or action == "discharge" and state.battery < 1.0 or action == "discharge_low" and state.battery < 0.5:
            # return state.p - state.c
            return -10000
        if action == "charge":
            #return (state.p - (state.c + MDP.max_charge)) if (state.p < state.c + MDP.max_charge) else 0
            return (state.p - (state.c + MDP.max_charge)) if (state.p < state.c + MDP.max_charge) else 0
        
        if action == "charge_low":
            return (state.p - (state.c + MDP.charge_low)) if (state.p < state.c + MDP.charge_low) else 0
        if action == "discharge_high":
            return ((state.p + MDP.max_discharge)  - state.c) if ((state.p + MDP.max_discharge)  < state.c) else 0
        if action == "discharge_low":
            return ((state.p + MDP.discharge_low)  - state.c) if ((state.p + MDP.max_discharge)  < state.c) else 0
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


#### Data analysis #####
##cons, prod, battery, time
# state_1 = State(200,0,4,23)
# print(State.get_reward("discharge",state_1))
realdata = RealData.get_real_data()

def data_to_states(data):
    states = []
    for i in range(len(data)-1):
        #print(MDP.get_production(data["Production"][i]))
        # data["Consumption"][i], data["Production"][i], , data["Time"][i]
        states.append((MDP.get_consumption(data["Consumption"][i]), MDP.get_production(data["Production"][i]),MDP.get_production(data["Production"][i+1])))
        # print(data["Consumption"][i])
        # print(State(data["Consumption"][i], data["Production"][i], 2, data["Production"][i+1],data["Time"][i]).consumption)
    # return states   
    return Counter(states)

def count_occurences(data):
    cons = []
    prod = []
    c = list(data["Consumption"])
    p = list(data["Production"])
    [cons.append(MDP.get_consumption(x)) for x in c]
    [prod.append(MDP.get_production(x)) for x in p]
    

    cons_occ = [0]*5
    prod_occ = [0]*5
    for i in range(len(cons_occ)):    
        cons_occ[i] = cons.count(MDP.consumption[i])
        prod_occ[i] = prod.count(MDP.production[i])
        
    return cons_occ, prod_occ     
# data_to_states(realdata)
# print(data_to_states(realdata))
occurences = data_to_states(realdata).values()
#print(list(occurences))
# plt.plot(list(occurences))
# plt.show()
# check getid
#high very high 3.5 very high 14

# s = State(440, 3300, 3.5, 3300, 0)
# print(State.get_id(s))