# file to create MDP for reinforcement learning problem


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import random
import time

from collections import Counter
from decimal import Decimal

class MDP:
    # initialize MDP
    def __init__(self, charge_high, discharge_high, charge_low,discharge_low, max_battery):
        

        
        # consumption_discr = [["low", "average", "high"],
        #                     ["very low", "low", "average", "high", "very high"],
        #                     ["low", "very low", "low", "moderately low" "average", "moderalety high", "high", "very high"],
        #                     ["extremely low", "very low", "low", "moderately low","average", "moderalety high", "high", "very high", "extremely high", "exceptionally high"]]
        # idx_c = int(np.floor(bins_cons/2) - 1) if self.bins_cons != 10 else 3
        # self.consumption = consumption_discr[idx_c]
        
        # self.bins_prod = bins_prod
        # production_discr = [["low","average", "high"],
        #                     ["none", "low","average", "high", "very high"],
        #                     ["none", "very low","low", "average_low", "average_high", "high", "very high"],
        #                     ["none", "very low","low", "moderately low", "average", "moderately high", "high", "very high", "extremely high", "exceptionally high"]]
        # idx_p = int(np.floor(self.bins_prod/2) - 1) if self.bins_prod != 10 else 3
        # self.production = production_discr[idx_p]
        # self.difference = ["-2000", "-1500", "-1000", "-500"," 0", "500", "1000", "1500", "2000","2500", "3000","3500", "4000", "4500", "5000"]

        self.difference = ["-2500","-2000", "-1500", "-1000", "-500"," 0", "500", "1000", "1500", "2000","2500", "3000","3500", "4000", "4500", "5000"]
        # self.difference = ["-2000", "-1500", "-1000", "-500"," 0", "500", "1000", "1500", "2000","2500", "3000","3500", "4000", "4500", "5000"]
        
        # time
        self.time = [*range(0,24,1)]

        # action space  
        self.action_space = ["discharge_high", "discharge_low", "do nothing","charge_low", "charge_high"]
        self.n_actions = len(self.action_space)
        self.discharge_high = discharge_high
        self.discharge_low = discharge_low #500

        self.charge_high = charge_high
        self.charge_low = charge_low # 500
        
        self.max_battery = max_battery
        self.battery_steps = [- self.discharge_high, - self.discharge_low, 0, self.charge_low, self.charge_high]
      
        # dimensions
        # self.n_consumption = len(self.consumption)
        # self.n_production = len(self.production)
        self.n_diff = len(self.difference)
        # self.n_pred_production = len(self.production)
        self.n_battery = self.get_battery_id(self.max_battery) + 1

        self.n_time = len(self.time)
        self.n_states = self.n_diff * self.n_battery * self.n_time
        


    def get_action_id(self,action):
        return self.action_space.index(action)

    def get_battery_id(self,battery):
        return int(battery * (1/min(self.discharge_low, self.charge_low))) 

    # reward function
    max_loss = -999999999999999999


    ## getters for discretization of difference
    def get_difference(self,d):
        # bins = [-1947.0, -1268.0, -589.0, 90.0, 769.0, 1448.0, 2127.0, 2806.0, 3485.0, 4164.0, 4850.0]
        bins = [-2500,-2000, -1500, -1000, -500, 0, 500, 1000, 1500, 2000,2500, 3000,3500, 4000, 4500, 5000]
        # bins = [-1947.0, -404.0, -323.0, -245.0, -221.0, -166.0, -102.0, 364.0, 1400.0, 3156.0, 4844.0]
        diff = self.difference
        
        intervals = [pd.Interval(left = bins[0], right = bins[1], closed = 'right'),pd.Interval(left = bins[1],right = bins[2], closed = 'right'), pd.Interval(left = bins[2],right =  bins[3], closed = 'right'), pd.Interval(left = bins[3],right =  bins[4], closed = 'right'), pd.Interval(left = bins[4],right =  bins[5], closed = 'right'), pd.Interval(left = bins[5],right =  bins[6], closed = 'right'), pd.Interval(left = bins[6],right =  bins[7], closed = 'right'), pd.Interval(left = bins[7],right =  bins[8], closed = 'right'),pd.Interval(left = bins[8],right =  bins[9], closed = 'right'), pd.Interval(left = bins[9],right =  bins[10], closed = 'right'), pd.Interval(left = bins[10],right =  bins[11], closed = 'right'), pd.Interval(left = bins[11],right =  bins[12], closed = 'right'), pd.Interval(left = bins[12],right =  bins[13], closed = 'right'), pd.Interval(left = bins[13],right =  bins[14], closed = 'right'), pd.Interval(left = bins[14],right =  bins[15], closed = 'right')]
        # print("intervals: "  + str(intervals))
        return self.get_label_for_value(intervals, diff, d)
    
    def get_label_for_value(self, intervals, labels, value):
        for interval, label in zip(intervals, labels):
            if value in interval:
                return label

    
    # function to return action_id with largest Q-value (!=0)
    # returns 2 (do nothing), if all q-values are the same
    
    def get_best_action(self,q_values):
        min_value = min(q_values)
        q_values = [value if value != 0 else (min_value -1) for value in q_values]
        max_value = max(q_values)
        if all(q == q_values[0] for q in q_values) or len(q_values) == 0:
            return random.choice([0,1,2,3,4])#, q_values
        indices = [index for index, value in enumerate(q_values) if value == max_value]
        return random.choice(indices)#, q_values

    def get_best_next(self,q_values):
        min_value = min(q_values)
        # print(q_values)
        q_values = [value if value != 0 else min_value for value in q_values]
        return max(q_values)
     
    # total cost function after applying learned policy
    def get_total_costs(self,rewards):
            rewards = np.array(rewards)
            rewards = [0 if x > 0 else x for x in rewards]
            
            return np.sum(rewards)
        
    # function to find policy after training the RL agent
    def find_policy(self, Q_table, data):
        costs = []
        actions = []
        battery = []
        
        current_state = State(data[0,0], data[0,1], 2000,data[0,2], self)
        
        l = data.shape[0]
        for i in range(l):
            
            action = self.action_space[self.get_best_action(Q_table[int(State.get_id(current_state, self)),:])]
            costs.append(State.get_cost(current_state,action, self))
            actions.append(action)
            battery.append(current_state.battery)
            current_state = State.get_next_state(current_state, action, data[i,0], data[i,1] ,data[i,2], self)
            
        return costs, actions, battery

    def iterate_q(Q_table, self):
        actions = []
        for i in range(len(Q_table)):
            a = Q_table[i,:]
                    # print("row of Q-values: " +str(a))
            action = self.action_space[self.get_best_action(a)]
            
            actions.append(action)
        return actions


# class which constructs state
# Consumption, Production, Battery state, time, prediction of production in text timestep
class State:

    def __init__(self, c, p, battery, time, mdp):
        self.d = p - c
        self.difference = mdp.get_difference(self.d)
        self.battery = battery
        self.time = time
        self.c = c
        self.p = p
        
    
    def get_battery_value(action, mdp):
        return mdp.battery_steps[mdp.action_space.index(action)]


    # move to next state based on chosen action
    # only battery has to be updated
    # only battery has to be updated
    def get_next_state(self, action, new_c, new_p, new_time, mdp):
        delta = State.get_battery_value(action, mdp)
        next_battery = self.battery
        if 0 <= self.battery + delta <= mdp.max_battery:
            next_battery = self.battery + delta
        next_state = State(new_c, new_p, int(next_battery), new_time, mdp)

        return next_state
    

    def get_id(state, mdp):
        # print(state.d)
        #diff = {"-2000":0, "-1500":1, "-1000":2, "-500":3," 0":4, "500":5, "1000":6, "1500":7, "2000":8,"2500":9, "3000":10,"3500":11, "4000":12, "4500":13, "5000":14}
        diff = {"-2500":0, "-2000":1, "-1500":2, "-1000":3, "-500":4," 0":5, "500":6, "1000":7, "1500":8, "2000":9,"2500":10, "3000":11,"3500":12, "4000":13, "4500":14, "5000":15}
        
        d = diff.get(state.difference)
        # print(state.difference)
        return d * (mdp.n_battery*24) + mdp.get_battery_id(state.battery)  * 24 + state.time
    
    def check_action(state,action, mdp):
        illegal = False
        irrational = False
        prod = state.p
        cons = state.c
        delta = State.get_battery_value(action, mdp)
        if prod - cons < delta and action == "charge_high" or action == "charge_low":
            irrational = True
        if state.battery + delta < 0 or state.battery + delta > mdp.max_battery:
            illegal = True
        else:
            # charge
            if delta > 0:
                cons += delta
            if delta < 0:
                prod -= delta
        # print(state.battery, action, delta, state.c, cons, state.p, prod, illegal, irrational)
        return illegal, irrational, prod, cons
    
    def get_reward(state, action, mdp):
        action_illegal, action_irrational, p, c = State.check_action(state, action, mdp)
        if action_illegal or action_irrational:
            return mdp.max_loss
        else:
            return - np.abs(p - c)

     
    # function to calculate sum of purchased energy after using battery    
    def get_cost(state,action, mdp):
        action_illegal, action_irrational, p, c = State.check_action(state, action, mdp)
        if action_irrational and not action_illegal:
            if action == "charge_high":
                c += mdp.charge_high
            if action == "charge_low":
                c += mdp.charge_low
        return min(p - c, 0)
      

# print(mdp.get_difference(4516.0))
# data_to_states(mdp_7, realdata)
# # print(data_to_states(realdata))
# occurences = data_to_states(mdp_7, realdata)
# print(occurences)
# print(len(occurences))
# print(mdp_7.n_states / 13)
# print(mdp_7.n_battery)
# plt.plot(list(occurences))
# plt.show()
# check getid
#high very high 3.5 very high 14
# very high very high 6.0 very high 10
# s = State(440, 3300, 1.0, 3300, 23)
# print(State.get_id(s))

