from MA2_environment import MDP
from MA2_environment import State as State2

import numpy as np
import pandas as pd
import random

class MDP:
    # initialize MDP
    def __init__(self, max_charge, max_discharge, charge_low, discharge_low, max_battery, bins_cons, bins_prod):
        

        self.bins_cons = bins_cons
        consumption_discr = [["low", "average", "high"],
                            ["very low", "low", "average", "high", "very high"],
                            ["very low", "low", "moderately low", "average", "moderately high", "high", "very high"],
                            ["extremely low", "very low", "low", "moderately low","average", "moderately high", "high", "very high", "extremely high", "exceptionally high"]]
        idx_c = int(np.floor(bins_cons/2) - 1) if self.bins_cons != 10 else 3
        self.consumption = consumption_discr[idx_c]
        
        self.bins_prod = bins_prod
        production_discr = [["low","average", "high"],
                            ["none", "low","average", "high", "very high"],
                            ["none", "very low","low", "average_low", "average_high", "high", "very high"],
                            ["none", "very low","low", "moderately low", "average", "moderately high", "high", "very high", "extremely high", "exceptionally high"]]
        idx_p = int(np.floor(self.bins_prod/2) - 1) if self.bins_prod != 10 else 3
        self.production = production_discr[idx_p]
        
        # time
        self.time = [*range(0,24,1)]

        ### only the action space is changed for the MARL
        # action space  
        self.action_space = ["discharge_high", "discharge_low", "do nothing","charge_low", "charge_high"]
        self.action_space_c = ["discharge_high", "discharge_low", "do nothing"]
        # self.action_space = ["discharge_high", "discharge_low", "do nothing","charge_low", "charge_high"]
        
        self.n_actions = len(self.action_space)
        self.n_actions_c = self.n_actions - 2
        self.discharge_high = max_discharge
        self.discharge_low = discharge_low #500

        self.charge_high = max_charge
        self.charge_low = charge_low # 500
        
        self.max_battery = max_battery
        # self.step_high_charge = int(max_charge / 100)
        # self.step_high_discharge = int(max_discharge/100)
        # self._low_charge = int(charge_low / 100)
        # self.step_low_discharge = int(discharge_low / 100)
        self.battery_steps = [- self.discharge_high, - self.discharge_low, 0, self.charge_low, self.charge_high]
        # dimensions
        self.n_consumption = len(self.consumption)
        self.n_production = len(self.production)
        self.n_pred_production = len(self.production)
        self.n_battery = self.get_battery_id(self.max_battery) + 1

        self.n_time = len(self.time)
        self.n_states = self.n_consumption * self.n_production * self.n_battery * self.n_time * self.n_pred_production
        
        


    def get_action_id(self,action):
        return self.action_space.index(action)

    def get_battery_id(self,battery):
        return int(battery * (1/min(self.discharge_low, self.charge_low))) 

    def get_best_next(self,q_values):
        min_value = min(q_values)
        # print(q_values)
        q_values = [value if value != 0 else min_value for value in q_values]
        return max(q_values)

    # reward function
    max_loss = -999999999999999999

    # # getters for data discretization

    def get_consumption(self,c):
        if self.bins_cons == 3: 
            return self.get_consumption_three(c)
        if self.bins_cons == 5: 
            return self.get_consumption_five(c)
        if self.bins_cons == 7: 
            return self.get_consumption_seven(c)
        if self.bins_cons == 10: 
            return self.get_consumption_ten(c)
        

    # bins: (0-250, 250-360, >360)
    def get_consumption_three(self,c):
        return "low" if c < 250  else  ("average" if (c > 250) and (c < 360) else ("high"))

    # bins: [0.  217.  266.  339.  424. 2817.]
    def get_consumption_five(self,c):
        cons = ["very low", "low", "average", "high", "very high"]
        intervals = [pd.Interval(left = 0, right = 215, closed = 'both'),pd.Interval(left = 215,right = 270, closed = 'right'), pd.Interval(left = 270,right =  340, closed = 'right'), pd.Interval(left = 340,right =  430, closed = 'right'), pd.Interval(left = 430,right =  2900, closed = 'right')]
        return self.get_label_for_value(intervals, cons, c)
    
    # bins : [0.0, 196.0, 231.0, 278.0, 329.0, 382.0, 478.0, 2817]
    def get_consumption_seven(self,c):
        bins = [0.0, 196.0, 231.0, 278.0, 329.0, 382.0, 478.0, 2817]
        # bins = [0, 402, 804, 1206, 1608, 2010, 2412, 2820]
        cons = ["very low", "low", "moderately low","average", "moderately high", "high", "very high"]
        intervals = [pd.Interval(left = bins[0], right = bins[1], closed = 'both'),pd.Interval(left = bins[1],right = bins[2], closed = 'right'), pd.Interval(left = bins[2],right =  bins[3], closed = 'right'), pd.Interval(left = bins[3],right =  bins[4], closed = 'right'), pd.Interval(left = bins[4],right =  bins[5], closed = 'right'), pd.Interval(left = bins[5],right =  bins[6], closed = 'right'), pd.Interval(left = bins[6],right =  bins[7], closed = 'right')]
        return self.get_label_for_value(intervals, cons, c)
    
    # bins: [0.0, 165.0, 217.0, 234.0, 266.0, 304.0, 339.0, 375.0, 424.0, 570.0]
    def get_consumption_ten(self,c):
        bins = [0.0, 165.0, 217.0, 234.0, 266.0, 304.0, 339.0, 375.0, 424.0, 570.0]
        cons = ["extremely low", "very low", "low", "moderately low","average", "moderately high", "high", "very high", "extremely high", "exceptionally high"]
        intervals = [pd.Interval(left = bins[0], right = bins[1], closed = 'both'),pd.Interval(left = bins[1],right = bins[2], closed = 'right'), pd.Interval(left = bins[2],right =  bins[3], closed = 'right'), pd.Interval(left = bins[3],right =  bins[4], closed = 'right'), pd.Interval(left = bins[4],right =  bins[5], closed = 'right'), pd.Interval(left = bins[5],right =  bins[6], closed = 'right'), pd.Interval(left = bins[6],right =  bins[7], closed = 'right'), pd.Interval(left = bins[7],right =  bins[8], closed = 'right'),pd.Interval(left = bins[8],right =  bins[9], closed = 'right'), pd.Interval(left = bins[9],right =  3000, closed = 'right')]
        return self.get_label_for_value(intervals, cons, c)

    # getters for production based on chosen discretization
    def get_production(self,p):
        if self.bins_prod == 3:
            return self.get_production_three(p)
        if self.bins_prod == 5:
            return self.get_production_five(p)
        if self.bins_prod == 7:
            return self.get_production_seven(p)
        if self.bins_prod == 10:
            return self.get_production_ten(p)

    # with 3 bins: (0, 0-1200, >1200)
    def get_production_three(self,p):
        return "none" if p == 0  else  ("low" if (p > 0) and (p < 1200) else ("high"))
    # with 5 bins: (0, 0-330, 330-1200, 1200 - 3200, >3200), each bin with equal frequency
    def get_production_five_copy(self,p):
        prod = ["none", "low", "average", "high", "very high"]
        intervals = [pd.Interval(left = 0, right = 0, closed = 'both'),pd.Interval(left = 0,right = 330, closed = 'right'), pd.Interval(left = 330,right =  1200, closed = 'right'), pd.Interval(left = 1200,right =  3200, closed = 'right'), pd.Interval(left = 3200,right =  7000, closed = 'right')]
        return self.get_label_for_value(intervals, prod, p)

    def get_production_five(self, p):
        prod = ["none", "low", "average", "high", "very high"]
        intervals = [pd.Interval(left = 0, right = 0, closed = 'both'),pd.Interval(left = 0,right = 330, closed = 'right'), pd.Interval(left = 330,right =  1200, closed = 'right'), pd.Interval(left = 1200,right =  3200, closed = 'right'), pd.Interval(left = 3200,right =  7000, closed = 'right')]
        return self.get_label_for_value(intervals, prod, p)

    # with 7 bins [0, 1.0, 171.0, 523.0, 1200.0, 2427.0, 4034.0]
    def get_production_seven(self,p):
        bins = [0, 1.0, 171.0, 523.0, 1200.0, 2427.0, 4034.0, 6070]
        # bins = [0, 1.0, 1012.0, 2023.0, 3034.0, 4045.0, 5056.0, 6070]
        prod = ["none", "very low","low", "average_low", "average_high", "high", "very high"]
        intervals = [pd.Interval(left = bins[0], right = bins[1], closed = 'both'),pd.Interval(left = bins[1],right = bins[2], closed = 'right'), pd.Interval(left = bins[2],right =  bins[3], closed = 'right'), pd.Interval(left = bins[3],right =  bins[4], closed = 'right'), pd.Interval(left = bins[4],right =  bins[5], closed = 'right'), pd.Interval(left = bins[5],right =  bins[6], closed = 'right'), pd.Interval(left = bins[6],right =  bins[7], closed = 'right')]
        return self.get_label_for_value(intervals, prod, p)
            
    # with 10 bins [0, 1.0, 95.0, 275.0, 523.0, 953.0, 1548.0, 2430.0, 3491.0, 4569.0]
    def get_production_ten(self,p):
        bins = [0, 1.0, 95.0, 275.0, 523.0, 953.0, 1548.0, 2430.0, 3491.0, 4569.0]
        prod = ["none", "very low","low", "moderately low", "average", "moderately high", "high", "very high", "extremely high", "exceptionally high"]
        intervals = [pd.Interval(left = bins[0], right = bins[0], closed = 'both'),pd.Interval(left = bins[1],right = bins[2], closed = 'right'), pd.Interval(left = bins[2],right =  bins[3], closed = 'right'), pd.Interval(left = bins[3],right =  bins[4], closed = 'right'), pd.Interval(left = bins[4],right =  bins[5], closed = 'right'), pd.Interval(left = bins[5],right =  bins[6], closed = 'right'), pd.Interval(left = bins[6],right =  bins[7], closed = 'right'), pd.Interval(left = bins[7],right =  bins[8], closed = 'right'),pd.Interval(left = bins[8],right =  bins[9], closed = 'right'), pd.Interval(left = bins[9],right =  7000, closed = 'right')]
        return self.get_label_for_value(intervals, prod, p)

    def get_label_for_value(self, intervals, labels, value):
        for interval, label in zip(intervals, labels):
            if value in interval:
                return label 
            
    # function to add noise for prediction of production (necessary ???)
    def get_predict_prod(self,p):
        mu, sigma = 1, 0.2
        rand = np.random.normal(mu, sigma, 1)
        return int(rand * p) 
    
    # function to return action_id with largest Q-value (!=0)
    # returns 2 (do nothing), if all q-values are the same
    
    def get_best_action(self,q_values):
        
        min_value = min(q_values)
        #print(q_values)
        q_values = [value if value != 0 else (min_value -1) for value in q_values]
        max_value = max(q_values)
        if all(q == q_values[0] for q in q_values) or len(q_values) == 0:
            return random.choice([0,1,2,3,4]) if len(q_values) == 5 else random.choice([0,1,2])#, q_values
        indices = [index for index, value in enumerate(q_values) if value == max_value]
        #print(indices)
        
        return random.choice(indices)#, q_values
    
    # total cost function after applying learned policy
    def get_total_costs(self,rewards):
            rewards = np.array(rewards)
            rewards = [0 if x > 0 else x for x in rewards]
            
            return np.sum(rewards)

class Policy:                
    # function to find policy after training the RL agent
    def find_policies(self, Q_A, Q_B, Q_C, dat):
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
        
        state_A = State(dat["Consumption_A"][0], dat["Production_A"][0], 6000, dat["Production_A"][1], dat["Time"][0], self)
        state_B = State(dat["Consumption_B"][0], dat["Production_B"][0], 6000, dat["Production_B"][1], dat["Time"][0], self)
        state_C = State(dat["Consumption_C"][0], dat["Production_C"][0], 6000, dat["Production_C"][1], dat["Time"][0], self)
                                      
        for i in range(len(dat["Consumption_A"])):
            
            # print("iteration: " + str(i) + "time: " + str(dat["Time"][i]))
            action_A = self.action_space[self.get_best_action(Q_A[State2.get_id(state_A, self),:])]
            action_B = self.action_space[self.get_best_action(Q_B[State2.get_id(state_B, self),:])]
            action_C = self.action_space[self.get_best_action(Q_C[State2.get_id(state_C, self),:])]
            
            # print("State: " + str(State.get_id(current_state))
            # print("Action ID: " + str(MDP.get_action_id(action)))
            # print("Q table: " + str(Q_table[State.get_id(current_state),]))
            cost_A, cost_B, cost_C = Reward.get_cost(state_A, state_B, state_C, action_A, action_B, action_C, self)
            costs_A.append(- cost_A)
            costs_B.append(- cost_B)
            costs_C.append(- cost_C)
            
            
            policy_A.append(action_A)
            policy_B.append(action_B)
            policy_C.append(action_C)
            battery_A.append(state_A.battery)
            battery_B.append(state_B.battery)
            l = len(dat["Consumption_A"])
            state_A = State.get_next_state(state_A, dat["Consumption_A"][i%l], dat["Production_A"][i%l], dat["Production_A"][(i+1)%l], dat["Time"][i%l], self, action_A, action_B, action_C)
            state_B = State.get_next_state(state_B, dat["Consumption_B"][i%l], dat["Production_B"][i%l], dat["Production_B"][(i+1)%l], dat["Time"][i%l], self, action_A, action_B, action_C)
            state_C = State.get_next_state(state_C, dat["Consumption_C"][i%l], dat["Production_C"][i%l], dat["Production_C"][(i+1)%l], dat["Time"][i%l], self, action_A, action_B, action_C)
           

        return costs_A, costs_B, costs_C, policy_A, policy_B, policy_C, battery_A, battery_B

    def iterate_q(Q_table, self):
        actions = []
        for i in range(len(Q_table)):
            a = Q_table[i,:]
                    # print("row of Q-values: " +str(a))
            action = self.action_space[self.get_best_action(a)]
            
            actions.append(action)
        return actions

class State:
    def __init__(self, c, p, battery, p_next, time, mdp):
        self.consumption = mdp.get_consumption(c)
        self.production = mdp.get_production(p)
        self.battery = battery
        self.time = time
        self.c = c
        self.p = p
        self.pred = p_next
        self.predicted_prod = mdp.get_production(self.pred)
        
    def get_next_state(self, new_c, new_p, new_pred, new_time, mdp, action_A, action_B, action_C):
        next_battery = self.get_next_battery(action_A, action_B, action_C, mdp)
        return State(new_c, new_p, int(next_battery), new_pred, new_time, mdp)

    def build_next_state(self, new_c, new_p, new_pred, new_time, mdp, next_battery):
        next_state = State(new_c, new_p, int(next_battery), new_pred, new_time, mdp)
        return next_state
    
    def get_battery_value(action, mdp):
        return mdp.battery_steps[mdp.action_space.index(action)]
        
    def get_action_for_delta(self,delta, mdp):
        return mdp.action_space[mdp.battery_steps.index(delta)]
    
    def check_actions(state, action_A, action_B, action_C, mdp):
        
        deltaA = State.get_battery_value(action_A, mdp)
        deltaB = State.get_battery_value(action_B, mdp)
        deltaC = State.get_battery_value(action_C, mdp)
        deltaA, deltaB, deltaC = State.check_deltas(state,deltaA, deltaB, deltaC, mdp)
        # # print(deltaA, deltaB, deltaC)
        # # check if deltas changed and get new actions
        # action_A = State.get_action_for_delta(state,deltaA, mdp)
        # action_B = State.get_action_for_delta(state,deltaB, mdp)
        # action_C = State.get_action_for_delta(state,deltaC, mdp)

        return deltaA, deltaB, deltaC #, action_A, action_B, action_C
    
    def check_deltas(state, deltaA, deltaB, deltaC, mdp):
        minimum = 0
        maximum = mdp.max_battery
        if deltaC > 0:
            raise ValueError
        deltas = [deltaA, deltaB, deltaC]
        sum_deltas = np.sum(deltas)
        
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
    

class Reward:
    # returns reward and new action if chosen action was not possible
    def get_reward(state_A, state_B, state_C, action_A, action_B, action_C, mdp):
        deltaA, deltaB, deltaC = State.check_actions(state_A, action_A, action_B, action_C, mdp)
        
        return Reward.calc_reward(state_A, deltaA, mdp), Reward.calc_reward(state_B, deltaB, mdp), Reward.calc_reward(state_C, deltaC, mdp)
    
    def calc_reward(state, delta, mdp):
        # max_loss only if charging and available does not cover production
        if delta > 0 and delta > state.p - state.c:
            return mdp.max_loss
        else:
            return - np.abs(state.p - delta - state.c)
    
    def get_cost(state_A, state_B, state_C, action_A, action_B, action_C, mdp):
        deltaA, deltaB, deltaC = State.check_actions(state_A, action_A, action_B, action_C, mdp)
        
        return Reward.calc_cost(state_A, deltaA), Reward.calc_cost(state_B, deltaB), Reward.calc_cost(state_C, deltaC)
    
    def calc_cost(state, delta):
        return min(state.p - delta - state.c, 0)

mdp = MDP(1000, 500, 500, 250,6000, 5,5)
# 1148.0,1148.0,388.0,5393.0,4314.0,0
s_A = State(1148,1000, 250, 1000, 13, mdp)
print(s_A.check_deltas(-250,-500,-250, mdp))
# s_B = State(1148, 4314, 5, 4000, 13, mdp)
# s_C = State(388, 0, 5, 0, 13, mdp)
# print(s_A.check_actions("discharge_high", "charge_high", "do nothing", mdp))
# print(Reward.get_reward(s_A, s_B, s_C, "charge_high", "charge_high", "do nothing", mdp))
# print(mdp.get_battery_id(500))

