# file to create MDP for reinforcement learning problem


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import random
import time


from data import RealData
from collections import Counter
from decimal import Decimal

class MDP:
    # initialize MDP
    def __init__(self, max_charge, max_discharge, charge_low, discharge_low, max_battery, bins_cons, bins_prod):
        

        self.bins_cons = bins_cons
        consumption_discr = [["low", "average", "high"],
                            ["very low", "low", "average", "high", "very high"],
                            ["very low", "low", "moderately low", "average", "moderately high", "high", "very high"],
                            ["extremely low", "very low", "low", "moderately low","average", "moderalety high", "high", "very high", "extremely high", "exceptionally high"]]
        
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

        # action space  
        self.action_space = ["discharge_high", "discharge_low", "do nothing","charge_low", "charge_high"]

        self.n_actions = len(self.action_space)
        self.discharge_high = max_discharge
        self.discharge_low = discharge_low #500

        self.charge_high = max_charge
        self.charge_low = charge_low # 500
        
        self.max_battery = max_battery
        # self.step_high_charge = int(max_charge / 100)
        # self.step_high_discharge = int(max_discharge/100)
        # self.step_low_charge = int(charge_low / 100)
        # self.step_low_discharge = int(discharge_low / 100)
        self.battery_steps = [- self.discharge_high, - self.discharge_low, 0, self.charge_low, self.charge_high]
        # self.charging_steps = [0, self.step_low_charge, self.step_low_charge]
        # self.discharging_steps = [0, - self.step_low_discharge, - self.step_high_discharge]
        # dimensions
        self.n_consumption = len(self.consumption)
        self.n_production = len(self.production)
        self.n_pred_production = len(self.production)
        self.n_battery = self.get_battery_id(self.max_battery) + 1

        self.n_time = len(self.time)
        self.n_states = self.n_consumption * self.n_production * self.n_battery * self.n_time * self.n_pred_production
        # print("states " + str(self.n_states))
        # print("battery: " + str(self.n_battery))
        # print("consumption: " + str(self.n_consumption))
        # print("production: " + str(self.n_production))
        


    def get_action_id(self,action):
        return self.action_space.index(action)

    def get_battery_id(self,battery):
        return int(battery * (1/min(self.discharge_low, self.charge_low))) 

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
        cons = ["extremely low", "very low", "low", "moderately low","average", "moderalety high", "high", "very high", "extremely high", "exceptionally high"]
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
        intervals = [pd.Interval(left = bins[0], right = bins[0], closed = 'both'),pd.Interval(left = bins[1],right = bins[2], closed = 'both'), pd.Interval(left = bins[2],right =  bins[3], closed = 'right'), pd.Interval(left = bins[3],right =  bins[4], closed = 'right'), pd.Interval(left = bins[4],right =  bins[5], closed = 'right'), pd.Interval(left = bins[5],right =  bins[6], closed = 'right'), pd.Interval(left = bins[6],right =  bins[7], closed = 'right'), pd.Interval(left = bins[7],right =  bins[8], closed = 'right'),pd.Interval(left = bins[8],right =  bins[9], closed = 'right'), pd.Interval(left = bins[9],right =  7000, closed = 'right')]
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
        # print(min(q_values))
        # print(q_values[q_values == 0])
        # q_values[q_values == 0] = min(q_values)
        # print(q_values)
        # print(np.argmax(q_values))
        min_value = min(q_values)
        #print(q_values)
        q_values = [value if value != 0 else (min_value -1) for value in q_values]
        max_value = max(q_values)
        
        # print("max value:" + str(max_value))
        if all(q == q_values[0] for q in q_values) or len(q_values) == 0:
            return random.choice([0,1,2,3,4])#, q_values
        indices = [index for index, value in enumerate(q_values) if value == max_value]
        #print(indices)
        
        return random.choice(indices)#, q_values
        #return np.argmax(q_values) if max(q_values) != 0 else 2, q_values

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
    def find_policy(self, Q_table, dat):
        costs = []
        actions = []
        battery = []
        states = []
        discharged = 0
        loss = 0
        current_state = State(dat["Consumption"][0], dat["Production"][0], 2000, dat["Production"][1], dat["Time"][0], self)
        
        for i in range(len(dat["Consumption"])):
            
            # print("iteration: " + str(i) + "time: " + str(dat["Time"][i]))
            action = self.action_space[self.get_best_action(Q_table[State.get_id(current_state, self),:])]
            # print(Q_table[State.get_id(current_state, self),:])
            # print(action)
            # print("State: " + str(State.get_id(current_state))
            # print("Action ID: " + str(MDP.get_action_id(action)))
            # print("Q table: " + str(Q_table[State.get_id(current_state),]))
            
            costs.append(State.get_cost(current_state,action, self))
            actions.append(action)
            battery.append(current_state.battery)
            states.append((current_state.consumption, current_state.production, current_state.battery,current_state.time))
            
            l = len(dat["Consumption"])
            current_state = State.get_next_state(current_state, action, dat["Consumption"][(i+1)%l], dat["Production"][(i+1)%l], dat["Production"][(i+2)%l], dat["Time"][(i+1)%l], self)
            
            # check amount of discharged energy
            if action == "discharge_high" and current_state.battery - self.discharge_high >= 0:
                    discharged += self.discharge_high
                    loss += max(((current_state.p + self.discharge_high) - current_state.c), 0)
            if action == "discharge_low" and current_state.battery - self.discharge_low >= 0:
                    discharged += self.discharge_low
                    loss += max(((current_state.p + self.discharge_high) - current_state.c), 0)
        return costs, actions, battery, discharged, loss, states

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

    def __init__(self, c, p, battery, p_next, time, mdp):
        self.consumption = mdp.get_consumption(c)
        self.production = mdp.get_production(p)
        self.battery = battery
        self.time = time
        self.c = c
        self.p = p
        self.pred = p_next # mdp.get_predicted_prod(p_next)
        self.predicted_prod = mdp.get_production(self.pred)
    
    # move to next state based on chosen action
    def get_battery_value(action, mdp):
        return mdp.battery_steps[mdp.action_space.index(action)]


    # only battery has to be updated
    def get_next_state(self, action, new_c, new_p,  new_pred, new_time, mdp):
        delta = State.get_battery_value(action, mdp)
        next_battery = self.battery
        if 0 <= self.battery + delta <= mdp.max_battery:
            next_battery = self.battery + delta
        next_state = State(new_c, new_p, int(next_battery), new_pred, new_time, mdp)

        return next_state
    

    # different getid for different discretization
    def get_id(state, mdp):
        id_functions = {
            3: State.get_id_three,
            5: State.get_id_five,
            7: State.get_id_seven,
            10: State.get_id_ten
        }
        bins = [3,5,7,10]
        return id_functions[mdp.bins_cons](state, mdp)
        

    # three 
    def get_id_three(state, mdp):
        c = {"low": 0, "average": 1, "high": 2}[state.consumption]
        p = {"none": 0, "low": 1, "average": 2}[state.production]
        pred = {"none": 0, "low": 1, "average": 2}[state.predicted_prod]
        
        return c * (mdp.n_production*mdp.n_battery*mdp.n_pred_production*24) + p *(mdp.n_battery*mdp.n_pred_production*24) + mdp.get_battery_id(state.battery) * (mdp.n_pred_production*24) + pred * 24 + state.time
        

    def get_id_five(state, mdp):
        c = {"very low": 0, "low": 1, "average": 2, "high": 3, "very high": 4}[state.consumption]
        p = {"none": 0, "low": 1, "average": 2, "high": 3, "very high": 4}[state.production]
        pred = {"none": 0, "low": 1, "average": 2, "high": 3, "very high": 4}[state.predicted_prod]

        return c * (mdp.n_production*mdp.n_battery*mdp.n_pred_production*24) + p *(mdp.n_battery*mdp.n_pred_production*24) + mdp.get_battery_id(state.battery) * (mdp.n_pred_production*24) + pred * 24 + state.time

    def get_id_seven(state, mdp):
        c = {"very low": 0, "low": 1, "moderately low": 2, "average": 3, "moderately high": 4, "high": 5, "very high": 6}[state.consumption]
        p = {"none": 0, "very low": 1, "low": 2, "average_low": 3, "average_high": 4, "high": 5, "very high":6}[state.production]
        pred = {"none": 0, "very low": 1, "low": 2, "average_low": 3, "average_high": 4, "high": 5, "very high": 6}[state.predicted_prod]
        # print(state.consumption)
        # print(state.production)
        # print(state.predicted_prod)
        # print("c " + str(c))
        # print("p " + str(p))
        # print("pred " +str(pred))
        return c * (mdp.n_production*mdp.n_battery*mdp.n_pred_production*24) + p *(mdp.n_battery*mdp.n_pred_production*24) + mdp.get_battery_id(state.battery) * (mdp.n_pred_production*24) + pred * 24 + state.time

    def get_id_ten(state, mdp):
        # consumption
        c = {"extremely low":0, "very low":1, "low":2, "moderately low":3,"average":4, "moderalety high":5, "high":6, "very high":7, "extremely high":8, "exceptionally high":9}[state.consumption]
        prod = {"none":0, "very low":1,"low":2, "moderately low":3, "average":4, "moderately high":5, "high":6, "very high":7, "extremely high":8, "exceptionally high":9}
        p = prod.get(state.production)
        pred = prod.get(state.predicted_prod)
        return c * (mdp.n_production*mdp.n_battery*mdp.n_pred_production*24) + p *(mdp.n_battery*mdp.n_pred_production*24) + mdp.get_battery_id(state.battery) * (mdp.n_pred_production*24) + pred * 24 + state.time
    
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
# realdata = RealData.get_real_data()

def data_to_states(mdp, data):
    states = []
    for i in range(len(data)-1):
        #print(MDP.get_production(data["Production"][i]))
        # data["Consumption"][i], data["Production"][i], , data["Time"][i]
        states.append((mdp.get_consumption(data["Consumption"][i]), mdp.get_production(data["Production"][i]),mdp.get_production(data["Production"][i+1]), data["Time"][i]))
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
# realdata = RealData.get_real_data()  
# mdp_7 = MDP(1000, 1000, 500, 500, 6, 7,7)
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

# # print(State.get_id(s))
mdp = MDP(1000, 500, 500, 200, 4000, 7,7)
# # # s = State(305.0, 119.0, 1000, 3300, 23, mdp)
# # # print(s.get_id(mdp))
# q = [-1, -2,0,-1,-2]
# print(MDP.get_best_next(mdp, q))
training, test = RealData.split_data(RealData.get_real_data(), 7)
s_training = data_to_states(mdp, training)
s_test = data_to_states(mdp, test)

unique_tuples = [t for t in s_training if (t not in s_test)]
# print(unique_tuples)
# print(len(unique_tuples))