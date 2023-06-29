from environment import State
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
        self.action_space = ["discharge_high", "discharge_low", "do nothing","charge_low", "charge_high", "charge_high_import", "charge_low_import"]

        self.n_actions = len(self.action_space)
        self.discharge_high = max_discharge
        self.discharge_low = discharge_low #500

        self.charge_high = max_charge
        self.charge_low = charge_low # 500
        
        self.max_battery = max_battery
        self.battery_steps = [- self.discharge_high, - self.discharge_low, 0, self.charge_low, self.charge_high, self.charge_high, self.charge_low]
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
        
        min_value = min(q_values)
        #print(q_values)
        q_values = [value if value != 0 else (min_value -1) for value in q_values]
        max_value = max(q_values)
        
        if all(q == q_values[0] for q in q_values) or len(q_values) == 0:
            return random.choice([0,1,2,3,4,5,6])
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

# addition to State class from Single Agent
class StateTransition:
      # only battery has to be updated
    def get_next_state(state, action, new_c, new_p,  new_pred, new_time, mdp):
        delta = Reward.get_battery_value(action, mdp)
        next_battery = state.battery
        if 0 <= state.battery + delta <= mdp.max_battery:
            next_battery = state.battery + delta
        next_state = State(new_c, new_p, int(next_battery), new_pred, new_time, mdp)

        return next_state

class Policy:                
    # function to find policy after training the RL agent
    def find_policies(Q_A, Q_B, dat, mdp_A, mdp_B):
        costs_A = []
        costs_B = []
        policy_A = []
        policy_B = []
        battery_A = []
        battery_B = []
        states = []
        discharged = 0
        loss = 0
        
        state_A = State(dat["Consumption_A"][0], dat["Production_A"][0], 2000, dat["Production_A"][1], dat["Time"][0], mdp_A)
        state_B = State(dat["Consumption_B"][0], dat["Production_B"][0], 2000, dat["Production_B"][1], dat["Time"][0], mdp_B)
                                      
        for i in range(len(dat["Consumption_A"])):
            
            # print("iteration: " + str(i) + "time: " + str(dat["Time"][i]))
            # print(state_A.consumption, state_A.production, state_A.predicted_prod, state_A.battery)
            action_A = mdp_A.action_space[mdp_A.get_best_action(Q_A[State.get_id(state_A, mdp_A),:])]
            action_B = mdp_B.action_space[mdp_B.get_best_action(Q_B[State.get_id(state_B, mdp_B),:])]
            
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
            l = len(dat["Consumption_A"])
            state_A = StateTransition.get_next_state(state_A, action_A, dat["Consumption_A"][i%l], dat["Production_A"][i%l], dat["Production_A"][(i+1)%l], dat["Time"][i%l], mdp_A)
            state_B = StateTransition.get_next_state(state_B, action_B, dat["Consumption_B"][i%l], dat["Production_B"][i%l], dat["Production_B"][(i+1)%l], dat["Time"][i%l], mdp_B)

        return costs_A, costs_B,  policy_A, policy_B, battery_A, battery_B
    
class Reward:
    
    # returns charging value 
    def get_battery_value(action, mdp):
        return mdp.battery_steps[mdp.action_space.index(action)]
    
    def get_reward(state, other_state, action, other_action, mdp, other_mdp):
        action_illegal, sharing_required, sharing_demanded, p, c = Reward.check_action(state, action, mdp)
        
        if action_illegal:
            return mdp.max_loss
        if sharing_required or sharing_demanded:
            possible, available_energy, p2,c2 = Reward.demand_charging(action, other_action, p,c, other_state, mdp, other_mdp)
            c = c2
            p = p2
            # compute if other available is demanded
            if not possible and sharing_demanded:
                return mdp.max_loss
            if possible:
                return - np.abs(min((p + available_energy) - c, 0))
            
        return - np.abs(p - c)
    

    # function gets only those states in which sharing is required (if own production doesn't cover other anymore)
    # gets only charge_high, charge_low, charge_high_import, charge_low_import as action input
    # only checks if other has enough rest available and returns rest

    # output: possible, amount available, own_p, own_c
    def demand_charging(action, other_action, own_p, own_c, other_state, mdp, other_mdp):
        delta = Reward.get_battery_value(action, mdp)
        if delta < 0:
            raise Exception("Wrong Action passed")
        other_c = other_state.c
        other_p = other_state.p
        # change action if importing is necessary
        action = "charge_high_import" if action == 'charge_high' else action
        action = "charge_low_import" if action == 'charge_low' else action

        # update consumption with amount to charge 
        own_c += delta 
    
        # define how much own production doesn't cover
        own_deficit = own_c - own_p
    
        # update other consumption if they charge
        other_delta = Reward.get_battery_value(other_action, other_mdp)
        other_c += other_delta if other_delta < 0 else other_delta
        
        # check if there is a required rest and if it can be covered by available rest from other
        if own_deficit <= other_p - other_c and own_deficit > 0:
            # possible: True, available, production, updated consumption
            return True, other_p - other_c, own_p, own_c
        else:
        # if no deficit to cover, or no sufficient rest available
            return False, 0, own_p, own_c


    # function to check if charging or discharging is illegal or irrational and if importing is required
    def check_action(state,action, mdp):
        illegal = False
        sharing_required = False
        sharing_demanded = False
        prod = state.p
        cons = state.c
        delta = Reward.get_battery_value(action, mdp)
        if action == "charge_high_import" or action == "charge_low_import":
            sharing_demanded = True
        # check charging actions
        if delta > 0:
            if (prod - cons) < delta:
                sharing_required = True
                if state.battery + delta > mdp.max_battery:
                    illegal = True
            else:
                cons += delta

        # check discharging actions
        if delta < 0:
            if state.battery + delta < 0:
                illegal = True
            else:
                prod -= delta 
        
        return illegal, sharing_required, sharing_demanded, prod, cons
    
    def get_cost(state, other_state, action, other_action,  mdp, other_mdp):
        action_illegal, sharing_required, sharing_demanded, p, c = Reward.check_action(state, action, mdp)
        if sharing_required or sharing_demanded:
            possible, available_energy, p2,c2 = Reward.demand_charging(action, other_action, p,c, other_state, mdp, other_mdp)
            c = c2
            p = p2
            # compute if other available is demanded
            if possible:
                return - min((min((p + available_energy) - c, 0)), 0)
        return - min(p - c, 0)

# mdp_B = MDP(1000, 500, 500, 200, 4000, 7,7)
# s_A = State(192.0, 0.0, 2000, 0.0, 2, mdp_B)
# # print(Reward.check_action(s_A, "charge_low", mdp_B))
# print(Reward.get_cost(s_A, s_A, "charge_low", "charge_high", mdp_B, mdp_B))
# # print(Reward.get_battery_value("charge_low", mdp_B))