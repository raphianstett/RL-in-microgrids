from MA2_environment import MDP

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
    
    def get_next_state(self, c_next, p_next, time_next, pred_next):
        
