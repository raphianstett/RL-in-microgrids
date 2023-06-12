import numpy as np
import environment 
from environment import State
import data_analysis
import pandas as pd

x = 3*3*5*24
list = [0]*x

c = environment.consumption
p = environment.production
b = environment.battery
t = environment.time

#print(c, p, b, t)

# for i in range(0,3):
#     for j in range(0,3):
#         for h in range(0,5):
#             for k in range(0,24):
#                 idx = i * (3*5*24) + j * (5*24) + h * 24 + k
#                 list[i*j*h*k] = (idx, c[i], p[j], b[h], t[k])
#                 #print(list[i*j*h*k])

#print(len(list))
# c * (n_production*n_battery*24) + p *(n_battery*24) + state.battery * 24 + state.time

d = {'Consumption': data_analysis.generate_step_consumption(), 'Production': data_analysis.generate_step_production(), 'Time': [*range(0,24,1)]}
dat = pd.DataFrame(d, columns = ['Consumption', 'Production', 'Time'])

data = pd.concat([dat]*10, ignore_index=True)
#print(data)

for e in range(10): 
    current_state = (data["Consumption"][e * 24], data["Production"][e*24], 2, data["Time"][e*24])
         
    for i in range(0,24):
        print(data.loc[e * 24 + i].iat[2])
