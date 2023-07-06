import matplotlib.pyplot as plt
import numpy as np

def test_exploration(i, e, color):
    exp = [0]*i
    min_exploration_proba = 0.01
    #exploration_decreasing_decay = 0.01
    exploration_decreasing_decay = e / i
    exploration_proba = 1
    for i in range(i):  
          
        exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*i))
        exp[i] = exploration_proba
    plt.plot(exp, color = color)
    return exp
    # plt.plot(test_exploration(500))

import matplotlib.pyplot as plt

# Data for the individual plots
x = [1, 2, 3]
y1 = [4, 5, 6]
y2 = [7, 8, 9]
y3 = [10, 11, 12]
y4 = [13, 14, 15]

# Create a figure and subplots
fig = plt.figure()

# Add the first subplot
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(x, y1)
ax1.set_title('Plot 1')

# Add the second subplot
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(x, y2)
ax2.set_title('Plot 2')

# Add the third subplot
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(x, y3)
ax3.set_title('Plot 3')

# Add the fourth subplot
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(x, y4)
ax4.set_title('Plot 4')

# Adjust spacing between subplots
plt.tight_layout()

# Display the combined plot
plt.show()
