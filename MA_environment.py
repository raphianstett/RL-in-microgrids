from data import RealData
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import random

class Environment:
    def __init__(self, max_battery_1, max_battery_2, production_factor, max_charge, max_discharge, charge_low, discharge_low, battery_step_size):
        # Battery
        self.max_battery_1 = max_battery_1
        self.max_battery_2 = max_battery_2
        self.max_charge = max_charge
        self.max_discharge = max_discharge
        self.step_size = battery_step_size
        self.charge_low = charge_low
        self.discharge_low = discharge_low

        self.battery_1 = list(np.arange(0.0, float(self.max_battery_1) + self.step_size, self.step_size))
        self.battery_2 = list(np.arange(0.0, float(self.max_battery_2) + self.step_size, self.step_size))
        
        # State space 
        self.production_factor = production_factor
        self.consumption = ["very low", "low", "average", "high", "very high"]
        self.production = ["none", "low","average", "high", "very high"]
        self.time = [*range(0,24,1)]


        # dimensions
        self.n_consumption = len(self.consumption)*2
        self.n_production = len(self.production)*2
        self.n_pred_production = len(self.production)*2
        self.n_battery = len(self.battery_1) * len(self.battery_2)

        self.n_time = len(self.time)
        self.n_states = self.n_consumption * self.n_production * self.n_battery * self.n_time * self.n_pred_production
        
        # action space  
        self.action_space = ["discharge_high", "discharge_low", "do nothing","charge_low", "charge_high", "take", "give"]
        self.n_actions = len(self.action_space)

        # max loss
        self.max_loss = -999999999

    # bins: [0.  217.  266.  339.  424. 2817.]
    def get_consumption(p):
        cons = Environment.consumption
        intervals = [pd.Interval(left = 0, right = 215, closed = 'both'),pd.Interval(left = 215,right = 270, closed = 'right'), pd.Interval(left = 270,right =  340, closed = 'right'), pd.Interval(left = 340,right =  430, closed = 'right'), pd.Interval(left = 430,right =  2900, closed = 'right')]
        return cons[0] if p in intervals[0] else (cons[1] if p in intervals[1] else (cons[2] if p in intervals[2] else (cons[3] if p in intervals[3] else cons[4]))) 

    # discretize production
    # with 5 bins: (0, 0-330, 330-1200, 1200 - 3200, >3200), each bin with equal frequency
    def get_production(p):
        prod = Environment.production
        intervals = [pd.Interval(left = 0, right = 0, closed = 'both'),pd.Interval(left = 0,right = 330, closed = 'right'), pd.Interval(left = 330,right =  1200, closed = 'right'), pd.Interval(left = 1200,right =  3200, closed = 'right'), pd.Interval(left = 3200,right =  7000, closed = 'right')]
        return prod[0] if p in intervals[0] else (prod[1] if p in intervals[1] else (prod[2] if p in intervals[2] else (prod[3] if p in intervals[3] else prod[4]))) 


    # function to add gaussian noise 
    def add_noise(p):
        mu, sigma = 1, 0.2
        rand = np.random.normal(mu, sigma, 1)
        return int(rand * p) 
    
    # returns action_id with largest Q-value (!=0)
    # returns 2 (do nothing), if all q-values are the same
    
    def get_best_action(q_values):
        q_values[q_values == 0] = min(q_values)
        return np.argmax(q_values) if max(q_values) != 0 else 2 
    
    # returns index for string
    def get_action_id(action):
        return Environment.action_space.index(action)

    def get_battery_id(battery):
        return int(battery * 2) 
    
    ### TO IMPLEMENT ###
    def apply_q_table(q_table, dat):
        return 0


class State:
    def __init__(self, c1, c2, p1, p2, battery_1, battery_2, p_next_1, p_next_2, time):
        self.consumption_1 = Environment.get_consumption(c1)
        self.consumption_2 = Environment.get_consumption(c2)
        self.production_1 = Environment.get_production(p1)
        self.production_2 = Environment.get_production(p2)
        self.battery_1 = battery_1
        self.battery_2 = battery_2

        self.time = time
        self.c1 = c1
        self.c2 = c2
        self.p1 = p1
        self.p2 = p2
        self.pred_1 = Environment.get_predict_prod(p_next_1)
        self.pred_2 = Environment.get_predict_prod(p_next_2)
        
        self.predicted_prod_1 = Environment.get_production(self.pred_1)
        self.predicted_prod_2 = Environment.get_production(self.pred_2)

    def get_next_state(self, action1, action2, new_c1, new_p1, new_pred1,new_c2, new_p2, new_pred2, new_time):
        next_battery_1 = State.get_next_battery(self, action1, self.battery_1, Environment.max_battery_1)
        next_battery_2 = State.get_next_battery(self, action2, self.battery_2, Environment.max_battery_2)

        next_state = State(new_c1, new_c2,new_p1, new_p2, next_battery_1, next_battery_2, new_pred1, new_pred2, new_time)

        return next_state
    
    def get_next_battery(self, action, battery, max_battery):
        # update battery state based on chosen action
        
        if action == "discharge_high" and battery > 0.5:
            next_battery = battery - 1.0
        elif action == "discharge_low" and battery > 0:
            next_battery = battery - Environment.step_size
        elif action == "charge_high" and battery <= Environment.max_battery - 1:
            next_battery = battery + 1.0
        elif action == "charge_low" and battery < Environment.max_battery:
            next_battery = battery + Environment.step_size
        else:
            next_battery = battery
        return next_battery

    def get_id(state):
        # consumption_1
        c1 = 0 if (state.consumption_1 == "very low") else (1 if (state.consumption_1 == "low") else (2 if (state.consumption_1 == "average") else (3 if (state.consumption_1 == "high") else 4)))
        # consumption_2
        c2 = 0 if (state.consumption_2 == "very low") else (1 if (state.consumption_2 == "low") else (2 if (state.consumption_2 == "average") else (3 if (state.consumption_2 == "high") else 4)))
        
        idx_c = c1 * ((Environment.n_consumption/2) * Environment.n_production*Environment.n_battery*Environment.n_pred_production*24) + c2 * (Environment.n_production*Environment.n_battery*Environment.n_pred_production*24)
        # production_1
        p1 = 0 if (state.production_1 == "none") else (1 if (state.production_1 == "low") else (2 if (state.production_1 == "average") else (3 if (state.production_1 == "high") else 4)))
         # production_2
        p2 = 0 if (state.production_2 == "none") else (1 if (state.production_2 == "low") else (2 if (state.production_2 == "average") else (3 if (state.production_2 == "high") else 4)))
        idx_p = p1 *((Environment.n_production /2) * Environment.n_battery*Environment.n_pred_production*24) + p2 *(Environment.n_battery*Environment.n_pred_production*24)
        # battery
        idx_b = Environment.get_battery_id(state.battery_1) * (Environment.max_battery_2 * Environment.n_pred_production*24) + Environment.get_battery_id(state.battery_2) * (Environment.n_pred_production*24)
        # predicted production
        pred1 = 0 if (state.predicted_prod_1 == "none") else (1 if (state.predicted_prod_1 == "low") else (2 if (state.predicted_prod_1 == "average") else (3 if (state.predicted_prod_1 == "high") else 4)))
        pred2 = 0 if (state.predicted_prod_2 == "none") else (1 if (state.predicted_prod_2 == "low") else (2 if (state.predicted_prod_2 == "average") else (3 if (state.predicted_prod_2 == "high") else 4)))
        idx_pred = pred1 * ((Environment.n_pred_production/2) * 24) + pred2 * 24

        return idx_c + idx_p + idx_b + idx_pred + state.time
    
    # returns false if action not allowed
    def check_actions(action1, action2, state):
        bool_a1 = True
        if action1 == 'charge_high' and state.battery_1 >= Environment.max_battery_1 - Environment.step_size or (state.p1 - state.c1) < Environment.max_charge:
            bool_a1 = False
        if action2 == 'charge_high' and state.battery_2 >= Environment.max_battery_2 - Environment.step_size or (state.p2 - state.c2) < Environment.max_charge:
            bool_a2 = False
        if action1 == 'charge_low' and state.battery_1 == Environment.max_battery_1 or state.p1 - state.c1 < Environment.charge_low:
            bool_a1 = False
        if action2 == 'charge_low' and state.battery_2 == Environment.max_battery_2 or state.p2 - state.c2 < Environment.charge_low:
            bool_a2 = False
        if action1 == 'discharge_high' and state.battery_1 < 1.0:
            bool_a1 = False
        if action2 == 'discharge_high' and state.battery_2 < 1.0:
            bool_a2 = False
        if action1 == 'discharge_low' and state.battery_1 < 0.5:
            bool_a1 = False
        if action2 == 'discharge_low' and state.battery_2 < 0.5:
            bool_a2 = False
        if action1 == 'give' and state.p1 < state.c1:
            bool_a1 = False
        if action2 == 'give' and state.p2 < state.c2:
            bool_a2 = False
        if action1 == 'take' and state.p1 > state.c1:
            bool_a1 = False
        if action2 == 'take' and state.p2 > state.c2:
            bool_a2 = False
        return (bool_a1 and bool_a2)
        
    def apply_actions(action1, action2, state):
        if action1 == 'charge_high':
            demand_1 = state.c1 + Environment.max_charge
        if action2 == 'charge_high':
            demand_2 = state.c2 + Environment.max_charge
        if action1 == 'charge_low':
            demand_1 = state.c1 + Environment.charge_low
        if action2 == 'charge_low':
            demand_2 = state.c2 + Environment.charge_low
        if action1 == 'discharge_high':
            production_1 = state.p1 + Environment.max_discharge
        if action2 == 'discharge_high':
            production_2 = state.p2 + Environment.max_discharge
        if action1 == 'discharge_low':
            production_1 = state.p1 + Environment.discharge_low
        if action2 == 'discharge_low':
            production_2 = state.p2 + Environment.discharge_low
        if action1 == 'give':
            production_2 = state.p2 + Environment.max_charge
            demand_1 = state.c1 + Environment.max_discharge
        if action2 == 'give':
            production_1 = state.p1 + Environment.max_charge
            demand_2 = state.c2 + Environment.max_discharge
        if action1 == 'take':
            production_1 = state.p1 + Environment.max_charge
            demand_2 = state.c2 + Environment.max_discharge
        if action2 == 'take':
            production_2 = state.p2 + Environment.max_charge
            demand_1 = state.c1 + Environment.max_discharge
        return production_1, production_2, demand_1, demand_2
       

    def get_reward(action1, action2, state):
        if State.check_actions(action1, action2, state):
            return Environment.max_loss
        production_1, production_2, demand_1, demand_2 = State.apply_actions(action1, action2, state)
        
        return -((production_1 + production_2) - (demand_1 + demand_2))**2
    
    def get_costs(action1, action2, state):
        if State.check_actions(action1, action2, state):
            return -10000
        
        production_1, production_2, demand_1, demand_2 = State.apply_actions(action1, action2, state)
        return min((production_1 + production_2 - (demand_1 + demand_2)), 0)
        


