import numpy as np
import matplotlib.pyplot as plt
def test_exploration(i):
    exp = [0]*i
    min_exploration_proba = 0.01
    #exploration_decreasing_decay = 0.01
    exploration_decreasing_decay = 1 / i
    exploration_proba = 1
    for i in range(i):  
          
        exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*i))
        exp[i] = exploration_proba
    return exp
plt.plot(test_exploration(500))
plt.show()