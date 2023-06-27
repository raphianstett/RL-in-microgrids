from environment import State
import numpy as np
import pandas as pd

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

        ### only the action space is changed for the MARL
        # action space  
        self.action_space = ["discharge_high", "discharge_low", "do nothing","charge_low", "charge_high", "charge_low_import", "charge_high_import"]
        # self.action_space = ["discharge_high", "discharge_low", "do nothing","charge_low", "charge_high"]
        
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
        self.n_consumption = len(self.consumption)
        self.n_production = len(self.production)
        self.n_pred_production = len(self.production)
        self.n_battery = self.max_battery * 10 + 1

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
        bins = [0.0, 196.0, 231.0, 278.0, 329.0, 382.0, 478.0, 2817]
        # bins = [0, 402, 804, 1206, 1608, 2010, 2412, 2820]
        cons = ["very low", "low", "moderately low","average", "moderalety high", "high", "very high"]
        intervals = [pd.Interval(left = bins[0], right = bins[1], closed = 'both'),pd.Interval(left = bins[1],right = bins[2], closed = 'right'), pd.Interval(left = bins[2],right =  bins[3], closed = 'right'), pd.Interval(left = bins[3],right =  bins[4], closed = 'right'), pd.Interval(left = bins[4],right =  bins[5], closed = 'right'), pd.Interval(left = bins[5],right =  bins[6], closed = 'right'), pd.Interval(left = bins[6],right =  bins[7], closed = 'right')]

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
    def find_policy(self, Q, dat, agent):
        costs = []
        actions = []
        battery = []
        states = []
        discharged = 0
        loss = 0
        if agent == "A":
            current_state = State(dat["Consumption_A"][0], dat["Production_A"][0], 20, dat["Production_A"][1], dat["Time"][0], self)
        if agent == 'B':
            current_state = State(dat["Consumption_B"][0], dat["Production_B"][0], 20, dat["Production_B"][1], dat["Time"][0], self)
        else:
            current_state = State(dat["Consumption_C"][0], dat["Production_C"][0], 20, dat["Production_C"][1], dat["Time"][0], self)
                                      
        for i in range(len(dat["Consumption_A"])):
            
            # print("iteration: " + str(i) + "time: " + str(dat["Time"][i]))
            action = self.action_space[self.get_best_action(Q[State.get_id(current_state, self),:])]
            
            # print("State: " + str(State.get_id(current_state))
            # print("Action ID: " + str(MDP.get_action_id(action)))
            # print("Q table: " + str(Q_table[State.get_id(current_state),]))
            
            costs.append(State.get_cost(current_state,action, self))
            actions.append(action)
            battery.append(current_state.battery)
            states.append((current_state.consumption, current_state.production, current_state.battery,current_state.time))
            
            l = len(dat["Consumption_A"])
            if agent == "A":
                current_state = State.get_next_state(current_state, action, dat["Consumption_A"][(i+1)%l], dat["Production_A"][(i+1)%l], dat["Production_A"][(i+2)%l], dat["Time"][(i+1)%l], self)
            if agent == 'B':
                current_state = State.get_next_state(current_state, action, dat["Consumption_B"][(i+1)%l], dat["Production_B"][(i+1)%l], dat["Production_B"][(i+2)%l], dat["Time"][(i+1)%l], self)
            else:
                current_state = State.get_next_state(current_state, action, dat["Consumption_C"][(i+1)%l], dat["Production_C"][(i+1)%l], dat["Production_C"][(i+2)%l], dat["Time"][(i+1)%l], self)
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

class Reward:
    def get_reward(state, other_state, action, other_action, mdp, other_mdp):
        action_illegal, sharing_required, p, c = Reward.check_action(state, action, mdp)
        if action_illegal:
            return mdp.max_loss
        if sharing_required:
            possible, available_energy, p2,c2 = Reward.demand_charging(action, other_action, p,c, other_state, mdp, other_mdp)
            c = c2
            p = p2
            # compute if other available is demanded
            if not possible:
                return mdp.max_loss
            if possible:
                return - np.abs(min((p + available_energy) - c, 0))
        else:
            return - np.abs(p - c)
    

    # function gets only those states in which sharing is required (if own production doesn't cover other anymore)
    # gets only charge_high, charge_low, charge_high_import, charge_low_import as action input
    # only checks if other has enough rest available and returns rest

    # output: possible, amount available, own_p, own_c
    def demand_charging(action, other_action, own_p, own_c, other_state, mdp, other_mdp):
        if action == "discharge_high" or action == "discharge_low":
            raise Exception("Wrong Action passed")
        other_c = other_state.c
        other_p = other_state.p
        action = "charge_high_import" if action == 'charge_high' else action
        action = "charge_low_import" if action == 'charge_low' else action
        # update consumption with amount to charge 
        own_c += mdp.max_charge if action == 'charge_high_import' else mdp.charge_low
    
        # define how much own production doesn't cover
        own_deficit = own_c - own_p
    
        # update other consumption if they charge
        if other_action =="charge_high" or 'charge_high_import':
            other_c += other_mdp.max_charge
        if other_action =="charge_low" or "charge_low_import":
            other_c += other_mdp.charge_low

        # check if there is a required rest and if it can be covered by available rest from other
        if own_deficit <= other_p - other_c and own_deficit > 0 :
            return True, other_p - other_c, own_p, own_c
        else:
        # if no deficit to cover, or no sufficient rest available
            return False, 0, own_p, own_c


# function to check if charging or discharging is illegal or irrational
    def check_action(state,action, mdp):
        illegal = False
        sharing_required = False
        prod = state.p
        cons = state.c
        if action == 'charge_high' or action == 'charge_high_import':
            if (prod - cons) < mdp.max_charge:
                sharing_required = True
                if state.battery + mdp.step_high_charge > mdp.max_battery:
                    illegal = True
            else:
                cons += mdp.max_charge

        if action == 'charge_low' or action == 'charge_low_import':
            if state.p - state.c < mdp.charge_low:
                    sharing_required = True
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
        return illegal, sharing_required, prod, cons
    
    def get_cost(state, other_state, action, other_action,  mdp, other_mdp):
        action_illegal, sharing_required, p, c = Reward.check_action(state, action, mdp)
        if sharing_required:
            possible, available_energy, p2,c2 = Reward.demand_charging(action, other_action, p,c, other_state, mdp, other_mdp)
            c = c2
            p = p2
            # compute if other available is demanded
            if possible:
                return - min((min((p + available_energy) - c, 0)), 0)
        else:
            return - min(p - c, 0)
    # def get_cost(self, state, other_state, action, other_action, mdp, other_mdp):
    #     p = state.p
    #     c = state.c
    #     if action == "discharge_high" or action =="discharge_low":
    #         possible, cons = Reward.check_discharging(state, action)
    #         c = cons if possible else c
    #     if action == "charge_high" or action =="charge_low":
    #         possible, prod = Reward.check_charging(state, action)
    #         p = prod if possible else p
    #     if action == "charge_low_import" or action == "charge_high_import":
    #         # if demanding is possible, the available energy from the other is added to the own production
    #         possible, rest = Reward.demand_charging(action, other_action, state, other_state, mdp, other_mdp)
    #         p += rest if possible else p 
    #     return min(p - c, 0)
        