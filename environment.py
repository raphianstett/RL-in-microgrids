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
    # initialize MDP
    def __init__(self, max_charge, max_discharge, discharge_low, charge_low, max_battery, bins_cons, bins_prod):
        

        self.bins_cons = bins_cons
        consumption_discr = [["low", "average", "high"],
                            ["very low", "low", "average", "high", "very high"],
                            ["low", "very low", "low", "moderately low" "average", "moderalety high", "high", "very high"],
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
        self.max_discharge = max_discharge
        self.discharge_low = discharge_low #500

        self.max_charge = max_charge
        self.charge_low = charge_low # 500

        self.max_battery = max_battery
        self.step_high = max_charge / 1000
        self.step_low_charge = charge_low / 1000
        self.step_low_discharge = discharge_low / 1000

        # dimensions
        self.n_consumption = len(self.consumption)
        self.n_production = len(self.production)
        self.n_pred_production = len(self.production)
        self.n_battery = int(self.max_battery * (1/min(self.step_low_charge, self.step_low_discharge))) + 1

        self.n_time = len(self.time)
        self.n_states = self.n_consumption * self.n_production * self.n_battery * self.n_time * self.n_pred_production
        


    def get_action_id(self,action):
        return self.action_space.index(action)

    def get_battery_id(self,battery):
        return int(battery * (1/min(self.step_low_charge, self.step_low_discharge))) 

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
        return cons[0] if c in intervals[0] else (cons[1] if c in intervals[1] else (cons[2] if c in intervals[2] else (cons[3] if c in intervals[3] else cons[4]))) 

    # bins : [0.0, 196.0, 231.0, 278.0, 329.0, 382.0, 478.0, 2817]
    def get_consumption_seven(self,c):
        cons = ["very low", "low", "moderately low","average", "moderalety high", "high", "very high"]
        intervals = [pd.Interval(left = 0, right = 196, closed = 'both'),pd.Interval(left = 196,right = 231, closed = 'right'), pd.Interval(left = 231,right =  278, closed = 'right'), pd.Interval(left = 278,right =  329, closed = 'right'), pd.Interval(left = 329,right =  382, closed = 'right'), pd.Interval(left = 382,right =  478, closed = 'right'), pd.Interval(left = 478,right =  3000, closed = 'right')]

        return cons[0] if c in intervals[0] else (cons[1] if c in intervals[1] else (cons[2] if c in intervals[2] else (cons[3] if c in intervals[3] else (cons[4] if c in intervals[4] else (cons[5] if c in intervals[5] else cons[6]))))) 

    # bins: [0.0, 165.0, 217.0, 234.0, 266.0, 304.0, 339.0, 375.0, 424.0, 570.0]
    def get_consumption_ten(self,c):
        bins = [0.0, 165.0, 217.0, 234.0, 266.0, 304.0, 339.0, 375.0, 424.0, 570.0]
        cons = ["extremely low", "very low", "low", "moderately low","average", "moderalety high", "high", "very high", "extremely high", "exceptionally high"]
        intervals = [pd.Interval(left = bins[0], right = bins[1], closed = 'both'),pd.Interval(left = bins[1],right = bins[2], closed = 'right'), pd.Interval(left = bins[2],right =  bins[3], closed = 'right'), pd.Interval(left = bins[3],right =  bins[4], closed = 'right'), pd.Interval(left = bins[4],right =  bins[5], closed = 'right'), pd.Interval(left = bins[5],right =  bins[6], closed = 'right'), pd.Interval(left = bins[6],right =  bins[7], closed = 'right'), pd.Interval(left = bins[7],right =  bins[8], closed = 'right'),pd.Interval(left = bins[8],right =  bins[9], closed = 'right'), pd.Interval(left = bins[9],right =  3000, closed = 'right')]

        return cons[0] if c in intervals[0] else (cons[1] if c in intervals[1] else (cons[2] if c in intervals[2] else (cons[3] if c in intervals[3] else (cons[4] if c in intervals[4] else (cons[5] if c in intervals[5] else (cons[6] if c in intervals[6] else (cons[7] if c in intervals[7] else (cons[8] if c in intervals[8] else cons[9])))))))) 

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
        prod = ["none", "very low","low", "average_low", "average_high", "high", "very high"]
        intervals = [pd.Interval(left = 0, right = 0, closed = 'both'),pd.Interval(left = 0,right = 171, closed = 'right'), pd.Interval(left = 171,right =  523, closed = 'right'), pd.Interval(left = 523,right =  1200, closed = 'right'), pd.Interval(left = 1200,right =  2427, closed = 'right'), pd.Interval(left = 2427,right =  4034, closed = 'right'), pd.Interval(left = 4034,right =  7000, closed = 'right')]
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
        discharged = 0
        current_state = State(dat["Consumption"][0], dat["Production"][0], 2.0, dat["Production"][1], dat["Time"][0], self)
        
        for i in range(len(dat["Consumption"])):
            
            # print("iteration: " + str(i) + "time: " + str(dat["Time"][i]))
            action = self.action_space[self.get_best_action(Q_table[State.get_id(current_state, self),:])]
            # print("State: " + str(State.get_id(current_state))
            # print("Action ID: " + str(MDP.get_action_id(action)))
            # print("Q table: " + str(Q_table[State.get_id(current_state),]))
            
            costs.append(State.get_cost(current_state,action, self))
            actions.append(action)
            battery.append(current_state.battery)
            l = len(dat["Consumption"])
            current_state = State.get_next_state(current_state, action, dat["Consumption"][(i+1)%l], dat["Production"][(i+1)%l], dat["Production"][(i+2)%l], dat["Time"][(i+1)%l], self)
            
            # check amount of discharged energy
            if action == "discharge_high":
                    discharged += self.max_discharge
            if action == "discharge_low":
                    discharged += self.discharge_low
        return costs, actions, battery, discharged

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
        self.pred = mdp.get_predict_prod(p_next)
        self.predicted_prod = mdp.get_production(self.pred)
    
    # move to next state based on chosen action
    # only battery has to be updated
    def get_next_state(self, action, new_c, new_p,  new_pred, new_time, mdp):
        
        if action == "discharge_high" and self.battery - mdp.step_high >= 0.0:
            next_battery = self.battery - mdp.step_high
        elif action == "discharge_low" and self.battery - mdp.step_low_discharge >= 0.0:
            next_battery = self.battery - mdp.step_low_discharge
        elif action == "charge_high" and self.battery + mdp.step_high <= mdp.max_battery:
            next_battery = self.battery + mdp.step_high
        elif action == "charge_low" and self.battery + mdp.step_low_charge <= mdp.max_battery:
            next_battery = self.battery + mdp.step_low_charge
        else:
            next_battery = self.battery

        next_state = State(new_c, new_p, next_battery, new_pred, new_time, mdp)

        return next_state
    

    # different getid for different discretization
    def get_id(state, mdp):
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
        # consumption
        cons= ["extremely low", "very low", "low", "moderately low","average", "moderalety high", "high", "very high", "extremely high", "exceptionally high"]
    
        c = 0 if (state.consumption == cons[0]) else (1 if (state.consumption == cons[1]) else (2 if (state.consumption == cons[2]) else (3 if (state.consumption == cons[3]) else (4 if (state.consumption == cons[4]) else (5 if (state.consumption == cons[5]) else (6 if (state.consumption == cons[6]) else (7 if (state.consumption == cons[7]) else (8 if (state.consumption == cons[8]) else 9)))))))) 
        # consumption
        prod = ["none", "very low","low", "moderately low", "average", "moderately high", "high", "very high", "extremely high", "exceptionally high"]

        p = 0 if (state.production == prod[0]) else (1 if (state.production == prod[1]) else (2 if (state.production == prod[2]) else (3 if (state.production == prod[3]) else (4 if (state.production == prod[4]) else (5 if (state.production == prod[5]) else (6 if (state.production == prod[6]) else (7 if (state.production == prod[7]) else (8 if (state.production == prod[8]) else 9)))))))) 
        # predicted production
        pred = 0 if (state.predicted_prod == prod[0]) else (1 if (state.predicted_prod == prod[1]) else (2 if (state.predicted_prod == prod[2]) else (3 if (state.predicted_prod == prod[3]) else (4 if (state.predicted_prod == prod[4]) else (5 if (state.predicted_prod == prod[5]) else (6 if (state.predicted_prod == prod[6]) else (7 if (state.predicted_prod == prod[7]) else (8 if (state.predicted_prod == prod[8]) else 9)))))))) 
        
        return c * (mdp.n_production*mdp.n_battery*mdp.n_pred_production*24) + p *(mdp.n_battery*mdp.n_pred_production*24) + mdp.get_battery_id(state.battery) * (mdp.n_pred_production*24) + pred * 24 + state.time
    
    def check_action(state,action, mdp):
        illegal = False
        irrational = False
        prod = state.p
        cons = state.c
        if action == 'charge_high':
            if (prod - cons) < mdp.max_charge:
                    irrational = True
                    if state.battery + mdp.step_high > mdp.max_battery:
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
            if state.battery - mdp.step_high < 0.0 :
                illegal = True
            else:
                prod += mdp.max_discharge 
        if action == 'discharge_low':
            if state.battery - mdp.step_low_discharge < 0.0:
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

# def data_to_states(data):
#     states = []
#     for i in range(len(data)-1):
#         #print(MDP.get_production(data["Production"][i]))
#         # data["Consumption"][i], data["Production"][i], , data["Time"][i]
#         states.append((MDP.get_consumption(data["Consumption"][i]), MDP.get_production(data["Production"][i]),MDP.get_production(data["Production"][i+1])))
#         # print(data["Consumption"][i])
#         # print(State(data["Consumption"][i], data["Production"][i], 2, data["Production"][i+1],data["Time"][i]).consumption)
#     # return states   
#     return Counter(states)

# # def count_occurences(data):
#     cons = []
#     prod = []
#     c = list(data["Consumption"])
#     p = list(data["Production"])
#     [cons.append(MDP.get_consumption(x)) for x in c]
#     [prod.append(MDP.get_production(x)) for x in p]
    

#     cons_occ = [0]*5
#     prod_occ = [0]*5
#     for i in range(len(cons_occ)):    
#         cons_occ[i] = cons.count(MDP.consumption[i])
#         prod_occ[i] = prod.count(MDP.production[i])
        
#     return cons_occ, prod_occ     
# data_to_states(realdata)
# print(data_to_states(realdata))
# occurences = data_to_states(realdata).values()
#print(list(occurences))
# plt.plot(list(occurences))
# plt.show()
# check getid
#high very high 3.5 very high 14
# very high very high 6.0 very high 10
# s = State(440, 3300, 1.0, 3300, 23)
# print(State.get_id(s))

