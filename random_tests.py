import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def test_exploration(i, e, color):
    exp = [0]*i
    min_exploration_proba = 0.01
    #exploration_decreasing_decay = 0.01
    exploration_decreasing_decay = e / i
    exploration_proba = 1
    for i in range(i):  
          
        exploration_proba = -max(min_exploration_proba, np.exp(-exploration_decreasing_decay*i))
        exp[i] = exploration_proba
    plt.plot(exp, color = color, label = "decay = " + str(e))
    return exp
    # plt.plot(test_exploration(500))
["lightcoral", "sandybrown", "yellowgreen", "lightslategrey", "royalblue"]
[1,3,5,7,9]
test_exploration(5000, 1, "lightcoral")
test_exploration(5000, 3, "sandybrown")
test_exploration(5000, 5, "yellowgreen")
test_exploration(5000, 7, "lightslategrey")
test_exploration(5000, 9, "royalblue")
plt.title('Exploration probability functions')
plt.xlabel("training episodes")
plt.legend()
plt.show()
import matplotlib.pyplot as plt

# x_values = [1, 5, 20, 50]
# y_values = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]  # Replace with your corresponding y-values
# x2 = np.arange(len(x_values))

# plt.plot(x2, y_values[0], label='Line 1')
# plt.plot(x2, y_values[1], label='Line 2')
# plt.plot(x2, y_values[2], label='Line 3')
# plt.plot(x2, y_values[3], label='Line 4')

# plt.xticks(np.arange(len(x_values)), x_values)  # Set custom x-axis tick positions and labels
# print(np.arange(len(x_values)))
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Non-linear X-axis Plot')
# plt.legend()

# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Data points
x_data = [100, 500, 750, 1000, 2000, 5000, 10000]
y_data = [1, 2, 3, 4, 5, 6, 7]

# Define the logarithmic function
def logarithmic_function(x, a, b):
    return a * np.log(x) + b

# Fit the logarithmic function to the data
params, _ = curve_fit(logarithmic_function, x_data, y_data)

# Generate values for plotting the fitted curve
x_plot = np.linspace(100, 10000, 100)
y_plot = logarithmic_function(x_plot, *params)
# print(y_plot)
# print(x_plot)
# Plot the original data and the fitted curve
# plt.scatter(x_data, y_data, label='Data')
# plt.plot(x_plot, y_plot, label='Fitted Curve')

# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Approximation with Logarithmic Function')
# plt.legend()

#plt.show()
# print(np.arange(1,11,1))
# print(len(np.arange(1,11,1)))

# print(len(np.arange(0.1,1.1,0.1)))
lrs = np.arange(0.1,1.1,0.1)
print(np.arange(12,200,24))
print(np.arange(0,8,1))
#print(str(lrs))