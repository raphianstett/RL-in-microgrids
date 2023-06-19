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
    def __init__(self, max_charge, max_discharge, discharge_low, charge_low, max_battery, bins_d, bins_prod):
        

        self.bins_d = bins_d
        # consumption_discr = [["low", "average", "high"],
        #                     ["very low", "low", "average", "high", "very high"],
        #                     ["low", "very low", "low", "moderately low" "average", "moderalety high", "high", "very high"],
        #                     ["extremely low", "very low", "low", "moderately low","average", "moderalety high", "high", "very high", "extremely high", "exceptionally high"]]
        # idx_c = int(np.floor(bins_cons/2) - 1) if self.bins_cons != 10 else 3
        # self.consumption = consumption_discr[idx_c]
        
        self.bins_prod = bins_prod
        production_discr = [["low","average", "high"],
                            ["none", "low","average", "high", "very high"],
                            ["none", "very low","low", "average_low", "average_high", "high", "very high"],
                            ["none", "very low","low", "moderately low", "average", "moderately high", "high", "very high", "extremely high", "exceptionally high"]]
        idx_p = int(np.floor(self.bins_prod/2) - 1) if self.bins_prod != 10 else 3
        self.production = production_discr[idx_p]
        self.difference = ["-2000", "-1500", "-1000", "-500"," 0", "500", "1000", "1500", "2000","2500", "3000","3500", "4000", "4500"]


        # time
        self.time = [*range(0,24,1)]

        # action space  
        self.action_space = ["discharge_high", "discharge_low", "do nothing","charge_low", "charge_high"]
        self.n_actions = len(self.action_space)
        self.max_discharge = max_discharge
        self.discharge_low = discharge_low #500

        self.max_charge = max_charge
        self.charge_low = charge_low # 500
        
        self.max_battery = max_battery
        self.step_high_charge = int(max_charge / 100)
        self.step_high_discharge = int(max_discharge/100)
        self.step_low_charge = int(charge_low / 100)
        self.step_low_discharge = int(discharge_low / 100)

        # dimensions
        # self.n_consumption = len(self.consumption)
        # self.n_production = len(self.production)
        self.n_diff = len(self.difference)
        self.n_pred_production = len(self.production)
        self.n_battery = self.max_battery * 10 + 1

        self.n_time = len(self.time)
        self.n_states = self.n_diff * self.n_battery * self.n_time * self.n_pred_production
        


    def get_action_id(self,action):
        return self.action_space.index(action)

    def get_battery_id(self,battery):
        return int(battery * (1/min(self.step_low_charge, self.step_low_discharge))) 

    # reward function
    max_loss = -999999999999999999



    ## getters for discretization of difference
    def get_difference_ten(self,d):
        # bins = [-1947.0, -1268.0, -589.0, 90.0, 769.0, 1448.0, 2127.0, 2806.0, 3485.0, 4164.0, 4850.0]
        bins = [-2000, -1500, -1000, -500, 0, 500, 1000, 1500, 2000,2500, 3000,3500, 4000, 4500, 5000]
        # bins = [-1947.0, -404.0, -323.0, -245.0, -221.0, -166.0, -102.0, 364.0, 1400.0, 3156.0, 4844.0]
        diff = ["-2000", "-1500", "-1000", "-500"," 0", "500", "1000", "1500", "2000","2500", "3000","3500", "4000", "4500"]
        
        intervals = [pd.Interval(left = bins[0], right = bins[1], closed = 'right'),pd.Interval(left = bins[1],right = bins[2], closed = 'right'), pd.Interval(left = bins[2],right =  bins[3], closed = 'right'), pd.Interval(left = bins[3],right =  bins[4], closed = 'right'), pd.Interval(left = bins[4],right =  bins[5], closed = 'right'), pd.Interval(left = bins[5],right =  bins[6], closed = 'right'), pd.Interval(left = bins[6],right =  bins[7], closed = 'right'), pd.Interval(left = bins[7],right =  bins[8], closed = 'right'),pd.Interval(left = bins[8],right =  bins[9], closed = 'right'), pd.Interval(left = bins[9],right =  3000, closed = 'right'), pd.Interval(left = bins[9],right =  3000, closed = 'right'), pd.Interval(left = bins[9],right =  3000, closed = 'right'), pd.Interval(left = bins[9],right =  bins[10], closed = 'right'), pd.Interval(left = bins[10],right =  bins[11], closed = 'right'), pd.Interval(left = bins[11],right =  bins[12], closed = 'right'), pd.Interval(left = bins[12],right =  bins[13], closed = 'right')]
        return diff[0] if d in intervals[0] else (diff[1] if d in intervals[1] else (diff[2] if d in intervals[2] else (diff[3] if d in intervals[3] else (diff[4] if d in intervals[4] else (diff[5] if d in intervals[5] else (diff[6] if d in intervals[6] else (diff[7] if d in intervals[7] else (diff[8] if d in intervals[8] else (diff[9] if d in intervals[9] else (diff[10] if d in intervals[10] else (diff[11] if d in intervals[11] else (diff[12] if d in intervals[12] else diff[13])))))))))))) 

    # # getters for data discretization

    def get_difference(self,d):
        return self.get_difference_ten(d)
   
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
    def get_production_five(self,p):
        prod = ["none", "low", "average", "high", "very high"]
        intervals = [pd.Interval(left = 0, right = 0, closed = 'both'),pd.Interval(left = 0,right = 330, closed = 'right'), pd.Interval(left = 330,right =  1200, closed = 'right'), pd.Interval(left = 1200,right =  3200, closed = 'right'), pd.Interval(left = 3200,right =  7000, closed = 'right')]
        return prod[0] if p in intervals[0] else (prod[1] if p in intervals[1] else (prod[2] if p in intervals[2] else (prod[3] if p in intervals[3] else prod[4]))) 
    
    # with 7 bins [0, 1.0, 171.0, 523.0, 1200.0, 2427.0, 4034.0]
    def get_production_seven(self,p):
        bins = [0, 1.0, 171.0, 523.0, 1200.0, 2427.0, 4034.0, 6070]
        # bins = [0, 1.0, 1012.0, 2023.0, 3034.0, 4045.0, 5056.0, 6070]
        prod = ["none", "very low","low", "average_low", "average_high", "high", "very high"]
        intervals = [pd.Interval(left = bins[0], right = bins[1], closed = 'both'),pd.Interval(left = bins[1],right = bins[2], closed = 'right'), pd.Interval(left = bins[2],right =  bins[3], closed = 'right'), pd.Interval(left = bins[3],right =  bins[4], closed = 'right'), pd.Interval(left = bins[4],right =  bins[5], closed = 'right'), pd.Interval(left = bins[5],right =  bins[6], closed = 'right'), pd.Interval(left = bins[6],right =  bins[7], closed = 'right')]
        return prod[0] if p in intervals[0] else (prod[1] if p in intervals[1] else (prod[2] if p in intervals[2] else (prod[3] if p in intervals[3] else (prod[4] if p in intervals[4] else (prod[5] if p in intervals[5] else prod[6]))))) 
    
    # with 10 bins [0, 1.0, 95.0, 275.0, 523.0, 953.0, 1548.0, 2430.0, 3491.0, 4569.0]
    def get_production_ten(self,p):
        bins = [0, 1.0, 95.0, 275.0, 523.0, 953.0, 1548.0, 2430.0, 3491.0, 4569.0]
        prod = ["none", "very low","low", "moderately low", "average", "moderately high", "high", "very high", "extremely high", "exceptionally high"]
        intervals = [pd.Interval(left = bins[0], right = bins[0], closed = 'both'),pd.Interval(left = bins[1],right = bins[2], closed = 'right'), pd.Interval(left = bins[2],right =  bins[3], closed = 'right'), pd.Interval(left = bins[3],right =  bins[4], closed = 'right'), pd.Interval(left = bins[4],right =  bins[5], closed = 'right'), pd.Interval(left = bins[5],right =  bins[6], closed = 'right'), pd.Interval(left = bins[6],right =  bins[7], closed = 'right'), pd.Interval(left = bins[7],right =  bins[8], closed = 'right'),pd.Interval(left = bins[8],right =  bins[9], closed = 'right'), pd.Interval(left = bins[9],right =  7000, closed = 'right')]
        return prod[0] if p in intervals[0] else (prod[1] if p in intervals[1] else (prod[2] if p in intervals[2] else (prod[3] if p in intervals[3] else (prod[4] if p in intervals[4] else (prod[5] if p in intervals[5] else (prod[6] if p in intervals[6] else (prod[7] if p in intervals[7] else (prod[8] if p in intervals[8] else prod[9])))))))) 


    # function to add noise for prediction of production (necessary ???)
    def get_predict_prod(self,p):
        mu, sigma = 1, 0.2
        rand = np.random.normal(mu, sigma, 1)
        return int(rand * p) 
    
    # function to return action_id with largest Q-value (!=0)
    # returns 2 (do nothing), if all q-values are the same
    
    def get_best_action(self,q_values):
        q_values[q_values == 0] = min(q_values)
        return np.argmax(q_values) if max(q_values) != 0 else 2 

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
        current_state = State(dat["Consumption"][0], dat["Production"][0], 20, dat["Production"][1], dat["Time"][0], self)
        
        for i in range(len(dat["Consumption"])):
            
            # print("iteration: " + str(i) + "time: " + str(dat["Time"][i]))
            action = self.action_space[self.get_best_action(Q_table[State.get_id(current_state, self),:])]
            # print("State: " + str(State.get_id(current_state))
            # print("Action ID: " + str(MDP.get_action_id(action)))
            # print("Q table: " + str(Q_table[State.get_id(current_state),]))
            
            costs.append(State.get_cost(current_state,action, self))
            actions.append(action)
            battery.append(current_state.battery)
            states.append((current_state.difference, current_state.battery,current_state.time))
            
            l = len(dat["Consumption"])
            current_state = State.get_next_state(current_state, action, dat["Consumption"][(i+1)%l], dat["Production"][(i+1)%l], dat["Production"][(i+2)%l], dat["Time"][(i+1)%l], self)
            
            # check amount of discharged energy
            if action == "discharge_high":
                    discharged += self.max_discharge
                    loss += max(((current_state.p + self.max_discharge) - current_state.c), 0)
            if action == "discharge_low":
                    discharged += self.discharge_low
                    loss += max(((current_state.p + self.max_discharge) - current_state.c), 0)
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
        self.difference = p - c
        self.battery = battery
        self.time = time
        self.c = c
        self.p = p
        self.pred = p_next
        self.predicted_prod = mdp.get_production(self.pred)
    
    # move to next state based on chosen action
    # only battery has to be updated
    def get_next_state(self, action, new_c, new_p,  new_pred, new_time, mdp):
        
        if action == "discharge_high" and self.battery - mdp.step_high_discharge >= 0:
            next_battery = self.battery - mdp.step_high_discharge
        elif action == "discharge_low" and self.battery - mdp.step_low_discharge >= 0:
            next_battery = self.battery - mdp.step_low_discharge
        elif action == "charge_high" and self.battery + mdp.step_high_charge <= mdp.max_battery:
            next_battery = self.battery + mdp.step_high_charge
        elif action == "charge_low" and self.battery + mdp.step_low_charge <= mdp.max_battery:
            next_battery = self.battery + mdp.step_low_charge
        else:
            next_battery = self.battery

        next_state = State(new_c,new_p, int(next_battery), new_pred, new_time, mdp)

        return next_state
    

    # different getid for different discretization
    def get_id(state, mdp):
        return State.get_id_ten(state, mdp)
        if mdp.bins_cons == 3:
            return State.get_id_three(state, mdp)
        if mdp.bins_cons == 5:
            return State.get_id_five(state, mdp)
        if mdp.bins_cons == 7:
            return State.get_id_seven(state, mdp)
        if mdp.bins_cons == 10:
            return State.get_id_ten(state, mdp)
        

    # three 
    def get_id_three(state, mdp):
        # consumption
        c = 0 if (state.consumption == "low") else (1 if (state.consumption == "average") else 2)
        # production
        p = 0 if (state.production == "none") else (1 if (state.production == "low") else 2)
        # predicted production
        pred = 0 if (state.predicted_prod == "none") else (1 if (state.predicted_prod == "low") else 2)
        
        return c * (mdp.n_production*mdp.n_battery*mdp.n_pred_production*24) + p *(mdp.n_battery*mdp.n_pred_production*24) + mdp.get_battery_id(state.battery) * (mdp.n_pred_production*24) + pred * 24 + state.time
        

    def get_id_five(state, mdp):
        # consumption
        c = 0 if (state.consumption == "very low") else (1 if (state.consumption == "low") else (2 if (state.consumption == "average") else (3 if (state.consumption == "high") else 4)))
        # production
        p = 0 if (state.production == "none") else (1 if (state.production == "low") else (2 if (state.production == "average") else (3 if (state.production == "high") else 4)))
        # predicted production
        pred = 0 if (state.predicted_prod == "none") else (1 if (state.predicted_prod == "low") else (2 if (state.predicted_prod == "average") else (3 if (state.predicted_prod == "high") else 4)))
        
        return c * (mdp.n_production*mdp.n_battery*mdp.n_pred_production*24) + p *(mdp.n_battery*mdp.n_pred_production*24) + mdp.get_battery_id(state.battery) * (mdp.n_pred_production*24) + pred * 24 + state.time

    def get_id_seven(state, mdp):
        # consumption
        c = 0 if (state.consumption == "very low") else (1 if (state.consumption == "low") else (2 if (state.consumption == "moderately low") else (3 if (state.consumption == "average") else (4 if (state.consumption == "moderately high") else (5 if (state.consumption == "high") else 6)))))
        # production
        p = 0 if (state.production == "none") else (1 if (state.production == "very low") else (2 if (state.production == "low") else (3 if (state.production == "average_low") else (4 if (state.production == "average_high") else (5 if (state.production == "high") else 6))))) 
        # predicted production
        pred = 0 if (state.predicted_prod == "none") else (1 if (state.predicted_prod == "very low") else (2 if (state.predicted_prod == "low") else (3 if (state.predicted_prod == "average_low") else (4 if (state.predicted_prod == "average_high") else (5 if (state.predicted_prod == "high") else 6)))))
        return c * (mdp.n_production*mdp.n_battery*mdp.n_pred_production*24) + p *(mdp.n_battery*mdp.n_pred_production*24) + mdp.get_battery_id(state.battery) * (mdp.n_pred_production*24) + pred * 24 + state.time

    def get_id_ten(state, mdp):
        diff = ["-2000", "-1500", "-1000", "-500"," 0", "500", "1000", "1500", "2000","2500", "3000","3500", "4000", "4500"]
        
        d = 0 if (state.difference == diff[0]) else (1 if (state.difference == diff[1]) else (2 if (state.difference == diff[2]) else (3 if (state.difference == diff[3]) else (4 if (state.difference == diff[4]) else (5 if (state.difference == diff[5]) else (6 if (state.difference == diff[6]) else (7 if (state.difference == diff[7]) else (8 if (state.difference == diff[8]) else (9 if (state.difference == diff[9]) else (10 if (state.difference == diff[10]) else (11 if (state.difference == diff[11]) else (12 if (state.difference == diff[12]) else 13)))))))))))) 
        # predicted difference
        prod = ["none", "very low","low", "moderately low", "average", "moderately high", "high", "very high", "extremely high", "exceptionally high"]
        pred = 0 if (state.predicted_prod == prod[0]) else (1 if (state.predicted_prod == prod[1]) else (2 if (state.predicted_prod == prod[2]) else (3 if (state.predicted_prod == prod[3]) else (4 if (state.predicted_prod == prod[4]) else (5 if (state.predicted_prod == prod[5]) else (6 if (state.predicted_prod == prod[6]) else (7 if (state.predicted_prod == prod[7]) else (8 if (state.predicted_prod == prod[8]) else 9)))))))) 
        
        return d * (mdp.n_battery*mdp.n_pred_production*24) + mdp.get_battery_id(state.battery) * (mdp.n_pred_production*24) + pred * 24 + state.time
    
    def check_action(state,action, mdp):
        illegal = False
        irrational = False
        prod = state.p
        cons = state.c
        if action == 'charge_high':
            if (prod - cons) < mdp.max_charge:
                    irrational = True
                    if state.battery + mdp.step_high_charge > mdp.max_battery:
                        illegal = True
            else:
                cons += mdp.max_charge
        if action == 'charge_low':
            
            if state.p - state.c < mdp.charge_low:
                    irrational = True
                    if state.battery + mdp.step_low_charge > mdp.max_battery:
                        illegal  = True
            else:
                cons += mdp.charge_low
        if action == "discharge_high" :
            if state.battery - mdp.step_high_discharge < 0:
                illegal = True
            else:
                prod += mdp.max_discharge 
        if action == 'discharge_low':
            if state.battery - mdp.step_low_discharge < 0:
                illegal = True
            else:
                prod += mdp.discharge_low
        return illegal, irrational, prod, cons
    
    def get_reward(state, action, mdp):
        action_illegal, action_irrational, p, c = State.check_action(state, action, mdp)
        if action_illegal or action_irrational:
            return mdp.max_loss
        else:
            return - np.abs(p - c)
        
    def get_reward_old(action, state, mdp):
        
        # f_char = (state.p / state.pred) if state.p != 0 and state.pred != 0 else 1
        # f_char = min(f_char, 2)
        # f_dischar = (state.pred / state.p) if state.p != 0 and state.pred != 0 else 1
        # f_dischar = min(f_dischar, 2)

        # f_nothing = min(max((state.p / state.c),(state.c / state.p)),100) if (state.p != 0 and state.c != 0) else 1

        # max_loss if action not possible, else calculate reward as squared difference between production and consumption
        # based on chosen action 
        if action == 'charge_high':
            
            if state.battery + mdp.step_high > mdp.max_battery or (state.p - state.c) < mdp.max_charge :
                return mdp.max_loss 
            else:
                #((mdp.max_battery+state.battery) / mdp.max_battery)*
                #return  min(-(state.p - (mdp.max_charge + state.c)), 0) 
                return -(state.p - (mdp.max_charge + state.c))**2
        if action == 'charge_low':
            if state.battery + mdp.step_low_charge > mdp.max_battery or state.p - state.c < mdp.charge_low:
                return mdp.max_loss
            else:
                # ((mdp.max_battery+state.battery) / mdp.max_battery)*
                # return min(-(state.p - (mdp.charge_low + state.c)), 0) 
                return -(state.p - (mdp.charge_low + state.c))**2
            
        if action == "discharge_high" :
            if state.battery - mdp.step_high < 0.0 :
                return mdp.max_loss
            else:
                # return -min(((state.p + mdp.max_discharge)  - state.c),0)
                return ((state.p + mdp.max_discharge)  - state.c)**2 
        if action == 'discharge_low':
            if state.battery - mdp.step_low_discharge < 0.0:
                return mdp.max_loss
            else:
                
                #(mdp.max_battery / (state.battery + mdp.max_battery))
                # return min(-((state.p + mdp.discharge_low)  - state.c), 0) 
                return -((state.p + mdp.discharge_low)  - state.c)**2
        if action == "do nothing":
           
            # max((mdp.max_battery / (state.battery + mdp.max_battery)), (state.battery + mdp.max_battery / (mdp.max_battery)))\
            # return min(-(state.p - state.c), 0) 
            return -(state.p - state.c)**2 

     
    # function to calculate sum of purchased energy after using battery    
    def get_cost(state,action, mdp):
        action_illegal, action_irrational, p, c = State.check_action(state, action, mdp)
        if action_irrational and not action_illegal:
            if action == "charge_high":
                c += mdp.max_charge
            if action == "charge_low":
                c += mdp.charge_low
        return min(p - c, 0)
        if (action == "charge_high" and state.battery + mdp.step_high > mdp.max_battery) or (action == "discharge_high" and state.battery - mdp.step_high < 0.0) or (action == "discharge_low" and state.battery - mdp.step_low_discharge < 0.0) or (action == "charge_low" and state.battery + mdp.step_low_charge > mdp.max_charge):
            return min(state.p - state.c, 0)
        if action == "charge_high":
            #return (state.p - (state.c + mdp.max_charge)) if (state.p < state.c + mdp.max_charge) else 0
            return min((state.p - (state.c + mdp.max_charge)), 0)
        if action == "charge_low":
            return min((state.p - (state.c + mdp.charge_low)), 0)
        if action == "discharge_high":
            return min(((state.p + mdp.max_discharge)  - state.c), 0)
        if action == "discharge_low":
            return min(((state.p + mdp.discharge_low)  - state.c), 0)
        if action == "do nothing":
            return min(state.p - state.c, 0)
        
    
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
# s = State(440, 3300, 1.0, 3300, 23)
# print(State.get_id(s))

