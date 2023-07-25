from environment import MDP as sMDP
from environment import State as sState

import numpy as np
import pandas as pd
import random

class MDP(sMDP):
    def __init__(self, charge_high, discharge_high, charge_low, discharge_low, max_battery, bins_cons, bins_prod):
        super().__init__(charge_high, discharge_high, charge_low, discharge_low, max_battery, bins_cons, bins_prod)
        self.action_space_c = ["discharge_high", "discharge_low", "do nothing"]
    
    def get_best_action(self,q_values):
        min_value = min(q_values)
        q_values = [value if value != 0 else (min_value -1) for value in q_values]
        max_value = max(q_values)
        if all(q == q_values[0] for q in q_values) or len(q_values) == 0:
            return random.choice([0,1,2,3,4]) if len(q_values) == 5 else random.choice([0,1,2])
        indices = [index for index, value in enumerate(q_values) if value == max_value]
        return random.choice(indices)

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
        conflicts = 0
        
        state_A = State(data[0,0], data[0,3], 2000, data[1,3] ,data[0,7], mdp)
        state_B = State(data[0,1], data[0,4], 2000, data[1,4] ,data[0,7], mdp)
        state_C = State(data[0,2], data[0,5], 2000, data[1,5] ,data[0,7], mdp)
        l = data.shape[0]

        for i in range(l):
            
            action_A = mdp.action_space[mdp.get_best_action(Q_A[int(State.get_id(state_A, mdp)),:])]
            action_B = mdp.action_space[mdp.get_best_action(Q_B[int(State.get_id(state_B, mdp)),:])]
            action_C = mdp.action_space_c[mdp.get_best_action(Q_C[int(State.get_id(state_C, mdp)),:])]
            
            cost_A, cost_B, cost_C = Reward.get_cost(state_A, state_B, state_C, action_A, action_B, action_C, mdp)
            conflicts += State.check_conflict(state_A, action_A, action_B, action_C, mdp)
            costs_A.append(cost_A)
            costs_B.append(cost_B)
            costs_C.append(cost_C)
            
            policy_A.append(action_A)
            policy_B.append(action_B)
            policy_C.append(action_C)
            battery_A.append(state_A.battery)
            battery_B.append(state_B.battery)
            
            state_A = State.get_next_state(state_A,data[i,0], data[i,3], data[(i+1)%l,3] ,data[i,7],mdp, action_A, action_B, action_C)
            state_B = State.get_next_state(state_B,data[i,1], data[i,4], data[(i+1)%l,4] ,data[i,7],mdp, action_A, action_B, action_C)
            state_C = State.get_next_state(state_C,data[i,2], data[i,5], data[(i+1)%l,5] ,data[i,7],mdp, action_A, action_B, action_C)


        return costs_A, costs_B, costs_C, policy_A, policy_B, policy_C, battery_A, conflicts



class State(sState):
    def __init__(self, c, p, battery, p_next, time, mdp):
        super().__init__(c, p, battery, p_next, time, mdp)

    def get_next_state(self, new_c, new_p, new_pred, new_time, mdp, action_A, action_B, action_C):
        next_battery = self.get_next_battery(action_A, action_B, action_C, mdp)
        return State(new_c, new_p, int(next_battery), new_pred, new_time, mdp)

    def get_action_for_delta(self,delta, mdp):
        return mdp.action_space[mdp.battery_steps.index(delta)]
    
    def check_actions(state, action_A, action_B, action_C, mdp):
        deltaA = State.get_step(action_A, mdp)
        deltaB = State.get_step(action_B, mdp)
        deltaC = State.get_step(action_C, mdp)
        deltaA, deltaB, deltaC = State.check_deltas(state,deltaA, deltaB, deltaC, mdp)
    
        # # check if deltas changed and get new actions
        action_A = State.get_action_for_delta(state,deltaA, mdp)
        action_B = State.get_action_for_delta(state,deltaB, mdp)
        action_C = State.get_action_for_delta(state,deltaC, mdp)

        return deltaA, deltaB, deltaC 
    
    def check_conflict(state, action_A, action_B, action_C, mdp):
        
        deltaA = State.get_step(action_A, mdp)
        deltaB = State.get_step(action_B, mdp)
        deltaC = State.get_step(action_C, mdp)
        deltaA, deltaB, deltaC = State.check_deltas(state,deltaA, deltaB, deltaC, mdp)

        # # check if deltas changed and get new actions
        new_action_A = State.get_action_for_delta(state,deltaA, mdp)
        new_action_B = State.get_action_for_delta(state,deltaB, mdp)
        new_action_C = State.get_action_for_delta(state,deltaC, mdp)
        return 1 if action_A != new_action_A or action_B != new_action_B or action_C != new_action_C else 0
        
    def check_deltas(state, deltaA, deltaB, deltaC, mdp):
        
        minimum = 0
        maximum = mdp.max_battery
        if deltaC > 0:
            raise ValueError
        deltas = [deltaA, deltaB, deltaC]
        sum_deltas = np.sum(deltas)
        if minimum <= state.battery + sum_deltas <= maximum:
            return deltas[0], deltas[1], deltas[2]
    
        while state.battery + sum_deltas > maximum or state.battery + sum_deltas < minimum:
            for counter in range(len(deltas)):
                # if global charging step is too large, deltas have to be reduced
                if state.battery + sum_deltas > maximum:
                    i = random.choice(State.get_max(deltas))
                    
                    delta = deltas[i]
                    # reduce deltas
                    if delta > 0:
                        reduction = mdp.charge_low
                        if delta >= reduction:
                            
                            deltas[i] -= reduction
                            sum_deltas -= reduction
                # if global discharging step is too large, deltas have to be increased
                else:
                    # chooses minimum values from deltas
                    i = random.choice(State.get_mins(deltas))
                    delta = deltas[i]
                    
                    if delta < 0:
                        # increase deltas by smaller step
                        increase = mdp.discharge_low
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

