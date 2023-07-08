from environment import State
import numpy as np
import pandas as pd
import random
from MA2_environment import Reward

class StateTransition:
      # only battery has to be updated
    def get_next_state(state, action, new_c, new_p, new_time, mdp):
        delta = Reward.get_battery_value(action, mdp)
        next_battery = state.battery
        if 0 <= state.battery + delta <= mdp.max_battery:
            next_battery = state.battery + delta
        next_state = State(new_c, new_p, int(next_battery), new_time, mdp)
        return next_state

class Policy:                
    # function to find policy after training the RL agent
    def find_policies(Q_A, Q_B, data, mdp_A, mdp_B):
        costs_A = []
        costs_B = []
        policy_A = []
        policy_B = []
        battery_A = []
        battery_B = []
        
        state_A = State(data[0,0], data[0,2], 2000, data[0,4], mdp_A)
        
        state_B = State(data[0,1], data[0,3], 2000, data[0,4], mdp_A)
        l = data.shape[0]
        
        for i in range(l):
            
            # print("iteration: " + str(i) + "time: " + str(dat["Time"][i]))
            # print(state_A.consumption, state_A.production, state_A.predicted_prod, state_A.battery)
            action_A = mdp_A.action_space[mdp_A.get_best_action(Q_A[int(State.get_id(state_A, mdp_A)),:])]
            action_B = mdp_B.action_space[mdp_B.get_best_action(Q_B[int(State.get_id(state_B, mdp_B)),:])]
            
            # print("State: " + str(State.get_id(current_state))
            # print("Action ID: " + str(MDP.get_action_id(action)))
            # print("Q table: " + str(Q_table[State.get_id(current_state),]))
            cost_A = Reward.get_cost(state_A, state_B, action_A, action_B, mdp_A, mdp_B)
            cost_B = Reward.get_cost(state_B, state_A, action_B, action_A, mdp_B, mdp_A)
            
            costs_A.append(- cost_A)
            costs_B.append(- cost_B)
            
            policy_A.append(action_A)
            policy_B.append(action_B)
            
            battery_A.append(state_A.battery)
            battery_B.append(state_B.battery)
            
            state_A = StateTransition.get_next_state(state_A, action_A, data[i,0], data[i,2], data[i,4], mdp_A)
            state_B = StateTransition.get_next_state(state_B, action_B, data[i,1], data[i,3],data[i,4], mdp_B)
        return costs_A, costs_B,  policy_A, policy_B, battery_A, battery_B
 
class State:

    def __init__(self, c, p, battery, time, mdp):
        self.d = p - c
        self.difference = mdp.get_difference(self.d)
        self.battery = battery
        self.time = time
        self.c = c
        self.p = p
        
    def get_id(state, mdp):
        # print(state.d)
        # print(state.difference)
        diff = {"-2500":0, "-2000":1, "-1500":2, "-1000":3, "-500":4," 0":5, "500":6, "1000":7, "1500":8, "2000":9,"2500":10, "3000":11,"3500":12, "4000":13, "4500":14, "5000":15}
        d = diff.get(state.difference)
        # print(d)

        return d * (mdp.n_battery*24) + mdp.get_battery_id(state.battery)  * 24 + state.time
    
class MDP:
    # initialize MDP
    def __init__(self, charge_high, discharge_high, charge_low,discharge_low, max_battery):
        
        self.difference = ["-2500","-2000", "-1500", "-1000", "-500"," 0", "500", "1000", "1500", "2000","2500", "3000","3500", "4000", "4500", "5000"]
    
        # action space  
        self.action_space = ["discharge_high", "discharge_low", "do nothing","charge_low", "charge_high", "charge_high_import", "charge_low_import"]
        self.n_actions = len(self.action_space)

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

class Policy:                
    # function to find policy after training the RL agent
    def find_policies(Q_A, Q_B, data, mdp_A, mdp_B):
        costs_A = []
        costs_B = []
        policy_A = []
        policy_B = []
        battery_A = []
        battery_B = []
        states = []
        discharged = 0
        loss = 0
        
        state_A = State(data[0,0], data[0,2], 2000, data[0,4], mdp_A)
        
        state_B = State(data[0,1], data[0,3], 2000, data[0,4], mdp_A)
        l = data.shape[0]
        
        for i in range(l):
            
            # print("iteration: " + str(i) + "time: " + str(dat["Time"][i]))
            # print(state_A.consumption, state_A.production, state_A.predicted_prod, state_A.battery)
            action_A = mdp_A.action_space[mdp_A.get_best_action(Q_A[int(State.get_id(state_A, mdp_A)),:])]
            action_B = mdp_B.action_space[mdp_B.get_best_action(Q_B[int(State.get_id(state_B, mdp_B)),:])]
            
            # print("State: " + str(State.get_id(current_state))
            # print("Action ID: " + str(MDP.get_action_id(action)))
            # print("Q table: " + str(Q_table[State.get_id(current_state),]))
            cost_A = Reward.get_cost(state_A, state_B, action_A, action_B, mdp_A, mdp_B)
            cost_B = Reward.get_cost(state_B, state_A, action_B, action_A, mdp_B, mdp_A)
            
            costs_A.append(- cost_A)
            costs_B.append(- cost_B)
            
            policy_A.append(action_A)
            policy_B.append(action_B)
            
            battery_A.append(state_A.battery)
            battery_B.append(state_B.battery)
            
            state_A = StateTransition.get_next_state(state_A, action_A, data[i,0], data[i,2] ,data[i,4], mdp_A)
            state_B = StateTransition.get_next_state(state_B, action_B, data[i,1], data[i,3] ,data[i,4], mdp_B)
        return costs_A, costs_B,  policy_A, policy_B, battery_A, battery_B
