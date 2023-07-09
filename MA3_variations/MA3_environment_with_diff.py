from MA2_environment import MDP
from MA2_environment import State as State2
from MA3_environment import Reward
import numpy as np
import pandas as pd
import random

class MDP:
    # initialize MDP
    def __init__(self, charge_high, discharge_high, charge_low,discharge_low, max_battery):
        
        self.difference = ["-2500","-2000", "-1500", "-1000", "-500"," 0", "500", "1000", "1500", "2000","2500", "3000","3500", "4000", "4500", "5000"]
    
        # action space  
        self.action_space = ["discharge_high", "discharge_low", "do nothing","charge_low", "charge_high"]
        self.action_space_c = ["discharge_high", "discharge_low", "do nothing"]
        self.n_actions = len(self.action_space)
        self.n_actions_c = len(self.action_space_c)
        self.discharge_high = discharge_high
        self.discharge_low = discharge_low #500
        self.charge_high = charge_high
        self.charge_low = charge_low # 500
        
        self.max_battery = max_battery
        self.battery_steps = [- self.discharge_high, - self.discharge_low, 0, self.charge_low, self.charge_high, self.charge_high, self.charge_low]
        self.n_diff = len(self.difference)        
        self.n_battery = self.get_battery_id(self.max_battery) + 1
        self.n_time = 24
        self.n_states = self.n_diff * self.n_battery * 24
        


    def get_action_id(self,action):
        return self.action_space.index(action)

    def get_battery_id(self,battery):
        return int(battery * (1/min(self.discharge_low, self.charge_low))) 

    # reward function
    max_loss = -999999999999999999


    ## getters for discretization of difference
    def get_difference(self,d):
        
        bins = [-2500,-2000, -1500, -1000, -500, 0, 500, 1000, 1500, 2000,2500, 3000,3500, 4000, 4500, 5000]
        
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
            return random.choice([0,1,2,3,4]) if len(q_values) == 5 else random.choice([0,1,2])
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
        
        
class Policy:                
    # function to find policy after training the RL agent
    def find_policies(mdp, Q_A, Q_B, Q_C, data):
        costs_A = []
        costs_B = []
        costs_C = []
        policy_A = []
        policy_B = []
        policy_C = []
        battery_A = []
        battery_B = []
        states = []
        discharged = 0
        loss = 0
        
        state_A = State(data[0,0], data[0,3], 2000, data[0,7], mdp)
        state_B = State(data[0,1], data[0,4], 2000, data[0,7], mdp)
        state_C = State(data[0,2], data[0,5], 2000, data[0,7], mdp)
        l = data.shape[0]

        for i in range(l):
            
            action_A = mdp.action_space[mdp.get_best_action(Q_A[int(State.get_id(state_A, mdp)),:])]
            action_B = mdp.action_space[mdp.get_best_action(Q_B[int(State.get_id(state_B, mdp)),:])]
            action_C = mdp.action_space[mdp.get_best_action(Q_C[int(State.get_id(state_C, mdp)),:])]
            
            cost_A, cost_B, cost_C = Reward.get_cost(state_A, state_B, state_C, action_A, action_B, action_C, mdp)
            costs_A.append(cost_A)
            costs_B.append(cost_B)
            costs_C.append(cost_C)
            
            
            policy_A.append(action_A)
            policy_B.append(action_B)
            policy_C.append(action_C)
            battery_A.append(state_A.battery)
            battery_B.append(state_B.battery)
            
            state_A = State.get_next_state(state_A,data[i,0], data[i,3] ,data[i,7], mdp, action_A, action_B, action_C)
            state_B = State.get_next_state(state_B,data[i,1], data[i,4] ,data[i,7], mdp, action_A, action_B, action_C)
            state_C = State.get_next_state(state_C,data[i,2], data[i,5] ,data[i,7], mdp, action_A, action_B, action_C)


        return costs_A, costs_B, costs_C, policy_A, policy_B, policy_C, battery_A

    def iterate_q(Q_table, self):
        actions = []
        for i in range(len(Q_table)):
            a = Q_table[i,:]
                    # print("row of Q-values: " +str(a))
            action = self.action_space[self.get_best_action(a)]
            
            actions.append(action)
        return actions

class State:
    def __init__(self, c, p, battery, time, mdp):
        self.d = p - c
        self.difference = mdp.get_difference(self.d)
        self.battery = battery
        self.time = time
        self.c = c
        self.p = p
    
    def get_id(state, mdp):
        diff = {"-2500":0, "-2000":1, "-1500":2, "-1000":3, "-500":4," 0":5, "500":6, "1000":7, "1500":8, "2000":9,"2500":10, "3000":11,"3500":12, "4000":13, "4500":14, "5000":15}
        d = diff.get(state.difference)
        
        return d * (mdp.n_battery*24) + mdp.get_battery_id(state.battery)  * 24 + state.time
    

    def get_next_state(self, new_c, new_p, new_time, mdp, action_A, action_B, action_C):
        next_battery = self.get_next_battery(action_A, action_B, action_C, mdp)
        # print("current: " + str(self.battery))
        # print("next: "+ str(next_battery))
        return State(new_c, new_p, int(next_battery), new_time, mdp)


    def get_battery_value(action, mdp):
        return mdp.battery_steps[mdp.action_space.index(action)]
        
    def get_action_for_delta(self,delta, mdp):
        return mdp.action_space[mdp.battery_steps.index(delta)]
    
    def check_actions(state, action_A, action_B, action_C, mdp):
        
        deltaA = State.get_battery_value(action_A, mdp)
        deltaB = State.get_battery_value(action_B, mdp)
        deltaC = State.get_battery_value(action_C, mdp)
        deltaA, deltaB, deltaC = State.check_deltas(state,deltaA, deltaB, deltaC, mdp)

        return deltaA, deltaB, deltaC #, action_A, action_B, action_C
    
    def check_deltas(state, deltaA, deltaB, deltaC, mdp):
        
        minimum = 0
        maximum = mdp.max_battery
        if deltaC > 0:
            raise ValueError
        deltas = [deltaA, deltaB, deltaC]
        sum_deltas = np.sum(deltas)
        # print("in check deltas:" + str(deltas))
        # print("battery: " + str(state.battery))
        # print(sum_deltas)
        if minimum <= state.battery + sum_deltas <= maximum:
            
            return deltas[0], deltas[1], deltas[2]
        
        # deltas = [deltaA, deltaB] if state.battery + sum_deltas > maximum else deltas
        # print(deltas)
        while state.battery + sum_deltas > maximum or state.battery + sum_deltas < minimum:
            #print("in while")
            
            for counter in range(len(deltas)):
                # CHARGING
                if state.battery + sum_deltas > maximum:
                    
                    # get_mins returns index of minimum value
                    i = random.choice(State.get_max(deltas))
                    
                    delta = deltas[i]
                    
                    if delta > 0:
                        reduction = mdp.charge_low
                        # print(state.battery + sum_deltas - maximum)
                        if delta >= reduction:
                            
                            deltas[i] -= reduction
                            sum_deltas -= reduction

                else:
                    # print("second")
                    # chooses minimum values from deltas
                    i = random.choice(State.get_mins(deltas))
                    delta = deltas[i]
                    
                    if delta < 0:
                        # print("second increase")
                        increase = mdp.discharge_low
                        # print(deltas)
                        # print(increase)
                        if delta <= -increase:
                            deltas[i] += increase
                            
                            sum_deltas += increase
                
                if minimum <= state.battery + sum_deltas <= maximum:
                    break

        deltaC = deltas[2] 
        deltaA = deltas[0]
        deltaB = deltas[1]
        return deltaA, deltaB, deltaC

    # checks if any element is equal to the minimum
    def get_mins(deltas):
        minimum = min(deltas)
        indexes = [i for i, value in enumerate(deltas) if value == minimum]
        return indexes
    def get_max(deltas):
        maximum = max(deltas)
        indexes = [i for i, value in enumerate(deltas) if value == maximum]
        return indexes
    
    def get_next_battery(self, action_A, action_B, action_C, mdp):
        deltaA, deltaB, deltaC = self.check_actions(action_A, action_B, action_C, mdp)
        return self.battery + (deltaA + deltaB + deltaC)
    