# create MDP with difference

import numpy as np
import pandas as pd
import random



class MDP:
    # initialize MDP
    def __init__(self, charge_high, discharge_high, charge_low,discharge_low, max_battery):
        self.difference = ["-2500","-2000", "-1500", "-1000", "-500"," 0", "500", "1000", "1500", "2000","2500", "3000","3500", "4000", "4500", "5000"]

        # action space  
        self.action_space = ["discharge_high", "discharge_low", "do nothing","charge_low", "charge_high"]
        self.n_actions = len(self.action_space)
        self.discharge_high = discharge_high
        self.discharge_low = discharge_low
        self.charge_high = charge_high
        self.charge_low = charge_low 
        
        self.max_battery = max_battery
        self.battery_steps = [- self.discharge_high, - self.discharge_low, 0, self.charge_low, self.charge_high]
        self.max_loss = -999999999999999999
        # dimensions
        
        self.n_diff = len(self.difference)
        self.n_battery = self.get_battery_id(self.max_battery) + 1
        self.n_time = 24
        self.n_states = self.n_diff * self.n_battery * self.n_time
        


    def get_action_id(self,action):
        return self.action_space.index(action)

    def get_battery_id(self,battery):
        return int(battery * (1/min(self.discharge_low, self.charge_low))) 

    # for discretization of difference
    def get_difference(self,d):
        bins = [-2500,-2000, -1500, -1000, -500, 0, 500, 1000, 1500, 2000,2500, 3000,3500, 4000, 4500, 5000]
        diff = self.difference
        
        intervals = [pd.Interval(left = bins[0], right = bins[1], closed = 'right'),pd.Interval(left = bins[1],right = bins[2], closed = 'right'), pd.Interval(left = bins[2],right =  bins[3], closed = 'right'), pd.Interval(left = bins[3],right =  bins[4], closed = 'right'), pd.Interval(left = bins[4],right =  bins[5], closed = 'right'), pd.Interval(left = bins[5],right =  bins[6], closed = 'right'), pd.Interval(left = bins[6],right =  bins[7], closed = 'right'), pd.Interval(left = bins[7],right =  bins[8], closed = 'right'),pd.Interval(left = bins[8],right =  bins[9], closed = 'right'), pd.Interval(left = bins[9],right =  bins[10], closed = 'right'), pd.Interval(left = bins[10],right =  bins[11], closed = 'right'), pd.Interval(left = bins[11],right =  bins[12], closed = 'right'), pd.Interval(left = bins[12],right =  bins[13], closed = 'right'), pd.Interval(left = bins[13],right =  bins[14], closed = 'right'), pd.Interval(left = bins[14],right =  bins[15], closed = 'right')]
        return self.get_label_for_value(intervals, diff, d)
    
    def get_label_for_value(self, intervals, labels, value):
        for interval, label in zip(intervals, labels):
            if value in interval:
                return label
        
    def get_best_action(self,q_values):
        min_value = min(q_values)
        q_values = [value if value != 0 else (min_value -1) for value in q_values]
        max_value = max(q_values)
        if all(q == q_values[0] for q in q_values) or len(q_values) == 0:
            return random.choice([0,1,2,3,4])
        indices = [index for index, value in enumerate(q_values) if value == max_value]
        return random.choice(indices)

    def get_best_next(self,q_values):
        min_value = min(q_values)
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
            costs.append(Reward.get_cost(current_state,action, self))
            actions.append(action)
            battery.append(current_state.battery)
            current_state = State.get_next_state(current_state, action, data[i,0], data[i,1] ,data[i,2], self)
            
        return costs, actions, battery


# class which constructs state
# continuous consumption, continuous production, ESS state, time
class State:

    def __init__(self, c, p, battery, time, mdp):
        self.d = p - c
        self.difference = mdp.get_difference(self.d)
        self.battery = battery
        self.time = time
        self.c = c
        self.p = p
        
    def get_step(action, mdp):
        return mdp.battery_steps[mdp.action_space.index(action)]

    # state transition function
    def get_next_state(self, action, new_c, new_p, new_time, mdp):
        delta = State.get_step(action, mdp)
        next_battery = self.battery
        if 0 <= self.battery + delta <= mdp.max_battery:
            next_battery = self.battery + delta
        next_state = State(new_c, new_p, int(next_battery), new_time, mdp)

        return next_state
    

    def get_id(state, mdp):
        diff = {"-2500":0, "-2000":1, "-1500":2, "-1000":3, "-500":4," 0":5, "500":6, "1000":7, "1500":8, "2000":9,"2500":10, "3000":11,"3500":12, "4000":13, "4500":14, "5000":15}
        d = diff.get(state.difference)
        return d * (mdp.n_battery*24) + mdp.get_battery_id(state.battery)  * 24 + state.time

class Reward:
    def check_action(state,action, mdp):
        illegal = False
        irrational = False
        prod = state.p
        cons = state.c
        delta = State.get_step(action, mdp)
        if prod - cons < delta and action == "charge_high" or action == "charge_low":
            irrational = True
        if state.battery + delta < 0 or state.battery + delta > mdp.max_battery:
            illegal = True
        else:
            # charge
            if delta > 0:
                cons += delta
            # discharge
            if delta < 0:
                prod -= delta
        return illegal, irrational, prod, cons
    
    def get_reward(state, action, mdp):
        action_illegal, action_irrational, p, c = Reward.check_action(state, action, mdp)
        if action_illegal or action_irrational:
            return mdp.max_loss
        else:
            return - np.abs(p - c)

     
    # function to calculate sum of purchased energy after using battery    
    def get_cost(state,action, mdp):
        action_illegal, action_irrational, p, c = Reward.check_action(state, action, mdp)
        if action_irrational and not action_illegal:
            if action == "charge_high":
                c += mdp.charge_high
            if action == "charge_low":
                c += mdp.charge_low
        return min(p - c, 0)
      