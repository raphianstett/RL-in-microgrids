from MA2_learning import MA_QLearning

from environment import State
from MA2_environment import Reward
from MA2_environment import MDP
from data import RealData
from MA_data import Data
import matplotlib.pyplot as plt
import numpy as np
from learning import Baseline
import pandas as pd
data = Data()
df = data.get_data()

pA = list(df["Production_A"])
pB = list(df["Production_B"])
cA = list(df["Consumption_A"])
cB = list(df["Consumption_B"])

diff_prod = [p - c for p,c in zip(pA, pB)]
diff_cons = [p - c for p,c in zip(cA, cB)]



diff_A = [p - c for p,c in zip(pA, cA)]
print(diff_A)

diff_B = [p - c for p,c in zip(pB, cB)]
print(diff_B)


## A: charge_high_import, B: charge_high
def check_import_high_A(diff_A, diff_B):
    charge_high_A = 0
    for a,b in zip(diff_A, diff_B):
        if 0 < a < 1000 and b - 1000 > 1000 - a:
            charge_high_A += 1
    return charge_high_A

## A: charge_high_import, B: charge_low
def check_import_low_A(diff_A, diff_B):
    charge_high_A = 0
    for a,b in zip(diff_A, diff_B):
        if 0 < a < 1000 and b > 1000 - a:
            charge_high_A += 1
    return charge_high_A

print(check_import_low_A(diff_A, diff_B))
print(max(diff_cons))
print(min(diff_cons))

print(max(diff_prod))
print(min(diff_prod))

dat_analysis = pd.DataFrame({"Difference_A:": diff_A, "Difference_B:": diff_B, "Difference Production": diff_prod, "Difference Consumption": diff_cons})
# print(dat_analysis[:200])