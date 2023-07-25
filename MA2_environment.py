from environment import State
from environment import MDP as sMDP
import numpy as np
import pandas as pd
import random


class MDP(sMDP):
    def __init__(self, charge_high, discharge_high, charge_low, discharge_low, max_battery, bins_cons, bins_prod):
        super().__init__(charge_high, discharge_high, charge_low, discharge_low, max_battery, bins_cons, bins_prod)
        
        self.action_space = ["discharge_high", "discharge_low", "do nothing","charge_low", "charge_high", "charge_high_import", "charge_low_import"]
        self.battery_steps = [- self.discharge_high, - self.discharge_low, 0, self.charge_low, self.charge_high, self.charge_high, self.charge_low]

class Policy:                
    # function to find policies of agents A and B after training 
    def find_policies(Q_A, Q_B, data, mdp_A, mdp_B):
        costs_A = []
        costs_B = []
        policy_A = []
        policy_B = []
        battery_A = []
        battery_B = []
        
        state_A = State(data[0,0], data[0,2], 2000, data[1,2] ,data[0,4], mdp_A)
        state_B = State(data[0,1], data[0,3], 2000, data[1,3] ,data[0,4], mdp_B)
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
            
            state_A = state_A.get_next_state(action_A, data[i,0], data[i,2], data[(i+1)%l,2] ,data[i,4], mdp_A)
            state_B = state_B.get_next_state(action_B, data[i,1], data[i,3], data[(i+1)%l,3] ,data[i,4], mdp_B)
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
    

    # function gets only those states in which sharing is required (if own production doesn't cover other anymore)
    # gets only charge_high, charge_low, charge_high_import, charge_low_import as action input
    # only checks if other has enough rest available and returns rest

    # returns possible, amount available, own_p, own_c
    def demand_charging(action, other_action, own_p, own_c, other_state, mdp, other_mdp):
        delta = State.get_step(action, mdp)
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
        other_delta = State.get_step(other_action, other_mdp)
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
        delta = State.get_step(action, mdp)
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
                # only as much imported as necessary
                return - min((min((p + available_energy) - c, 0)), 0)
        return - min(p - c, 0)
