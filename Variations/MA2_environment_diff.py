
import numpy as np
import pandas as pd
import random

from Variations.environment_with_diff import MDP as sMDP
from Variations.environment_with_diff import State

class MDP(sMDP):
    def __init__(self, charge_high, discharge_high, charge_low, discharge_low, max_battery):
        super().__init__(charge_high, discharge_high, charge_low, discharge_low, max_battery)
        # Modify the desired attribute in the subclass
        self.action_space = ["discharge_high", "discharge_low", "do nothing","charge_low", "charge_high", "charge_high_import", "charge_low_import"]
        self.battery_steps = [- self.discharge_high, - self.discharge_low, 0, self.charge_low, self.charge_high, self.charge_high, self.charge_low]
        
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
        
        state_B = State(data[0,1], data[0,3], 2000, data[0,4], mdp_B)
        l = data.shape[0]
        
        for i in range(l):
            action_A = mdp_A.action_space[mdp_A.get_best_action(Q_A[int(State.get_id(state_A, mdp_A)),:])]
            action_B = mdp_B.action_space[mdp_B.get_best_action(Q_B[int(State.get_id(state_B, mdp_B)),:])]
            
            cost_A = Reward.get_cost(state_A, state_B, action_A, action_B, mdp_A, mdp_B)
            cost_B = Reward.get_cost(state_B, state_A, action_B, action_A, mdp_B, mdp_A)
            
            costs_A.append(- cost_A)
            costs_B.append(- cost_B)
            
            policy_A.append(action_A)
            policy_B.append(action_B)
            
            battery_A.append(state_A.battery)
            battery_B.append(state_B.battery)
            
            state_A = State.get_next_state(state_A, action_A, data[i,0], data[i,2], data[i,4], mdp_A)
            state_B = State.get_next_state(state_B, action_B, data[i,1], data[i,3],data[i,4], mdp_B)
        return costs_A, costs_B,  policy_A, policy_B, battery_A, battery_B

class Reward:
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
    
    def demand_charging(action, other_action, own_p, own_c, other_state, mdp, other_mdp):
        delta = State.get_step(action, mdp)
        if delta < 0:
            raise Exception("Wrong Action passed")
        other_c = other_state.c
        other_p = other_state.p

        action = "charge_high_import" if action == 'charge_high' else action
        action = "charge_low_import" if action == 'charge_low' else action

        own_c += delta 
        own_deficit = own_c - own_p
        other_delta = State.get_step(other_action, other_mdp)
        other_c += other_delta if other_delta < 0 else other_delta
    
        if own_deficit <= other_p - other_c and own_deficit > 0:
            return True, other_p - other_c, own_p, own_c
        else:
            return False, 0, own_p, own_c

    def check_action(state,action, mdp):
        illegal = False
        sharing_required = False
        sharing_demanded = False
        prod = state.p
        cons = state.c
        delta = State.get_step(action, mdp)
        if action == "charge_high_import" or action == "charge_low_import":
            sharing_demanded = True
        if delta > 0:
            if (prod - cons) < delta:
                sharing_required = True
                if state.battery + delta > mdp.max_battery:
                    illegal = True
            else:
                cons += delta

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
            if possible:
                return - min((min((p + available_energy) - c, 0)), 0)
        return - min(p - c, 0)
