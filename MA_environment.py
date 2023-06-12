
class Reward:
    
    def get_reward(action,other_action, state, other_state, mdp, other_mdp):
        
        if Reward.check_charging(action, state, mdp) or Reward.check_discharging(action, state, mdp):
            return mdp.max_loss
         
        if action == 'charge_high':
            # also charge from other if possible
            if Reward.demand_charging(action, other_action, state, other_state, mdp, other_mdp):
                return -(min(((state.p + other_state.p) - (mdp.max_charge + state.c)), 0))**2
            else:
                return -(state.p - (mdp.max_charge + state.c))**2
        if action == 'charge_low':
            # with min, simulate that only the remaining is imported for charging
            if Reward.demand_charging(action, other_action, state, other_state, mdp, other_mdp):    
                return -(min(((state.p + other_state.p) - (mdp.charge_low + state.c)),0))**2
            
        if action == "discharge_high" :
            return ((state.p + mdp.max_discharge)  - state.c)**2 
        if action == 'discharge_low':
            return -((state.p + mdp.discharge_low)  - state.c)**2
        if action == "do nothing": 
            return -(state.p - state.c)**2 

    # returns false if action not allowed
    def check_discharging(action, state, mdp):
       
        if action == "discharge_high" and state.battery - mdp.step_high < 0.0 :
            return False
        if action == 'discharge_low' and state.battery - mdp.step_low_discharge < 0.0:
            return False
        else:
            return True

    # returns false if action is not allowed
    def check_charging(action, state, mdp):
        if action == 'charge_high' and state.battery + mdp.step_high > mdp.max_battery or (state.p - state.c) < mdp.max_charge:
            return False
        if action == 'charge_low' and state.battery + mdp.step_low_charge > mdp.max_battery or state.p - state.c < mdp.charge_low:
            return False
        else:
            return True
    
    # function to check wether charging from other household is possible
    def demand_charging(action, other_action, state, other_state, mdp, other_mdp):
        if action == 'charge_high':
            if other_action == 'charge_high' and other_state.p - other_mdp.max_charge < mdp.max_charge - state.p:
                return False 
            if other_action == 'charge_low' and other_state.p - other_mdp.charge_low < mdp.max_charge - state.p:
                return False
            else:
                return True
            
        if action == 'charge_low':
            if other_action == 'charge_high' and other_state.p - other_mdp.max_charge < mdp.charge_low - state.p:
                return False 
            if other_action == 'charge_low' and other_state.p - other_mdp.charge_low < mdp.charge_low - state.p:
                return False
            else:
                return True
            
        