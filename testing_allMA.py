import MA3_testing
import MA2_testing
import testing
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def plot_comparison():
    costs_5_1, costs_7_1, costs_d_1, baseline_costs_1, bs_1 = testing.get_performances([10000])
    costs_5_2, costs_7_2, costs_d_2, baseline_costs_2, bs_2 = MA2_testing.get_performances([10000])
    costs_5_3, costs_7_3, costs_d_3, baseline_costs_3, bs_3 = MA3_testing.get_performances_all([10000])
    perf1 = [1 - (costs_5_1/bs_1),1 - ( costs_7_1/bs_1), 1 - (costs_d_1/bs_1), 1 - (baseline_costs_1/bs_1)]
    perf2 = [1 - (costs_5_2/bs_2), 1 - (costs_7_2/bs_2), 1 - (costs_d_2/bs_2), 1 - (baseline_costs_2/bs_2)]
    perf3 = [1 - (costs_5_3/bs_3),1 - ( costs_7_3/bs_3), 1 - (costs_d_3/bs_3), 1 - (baseline_costs_3/bs_3)]
    print("baseline 3 costs:" + str(baseline_costs_3))
    print("2 difference costs" + str(costs_d_2))
    print("baseline 2: " + str(baseline_costs_2))
    print("3 difference costs: " + str(costs_d_3))
    print(bs_3)
    print(perf1)
    print(perf2)
    print(perf3)   
    # Set the number of bars
    num_bars = len(perf1)

    # Set the width of each bar
    bar_width = 0.2

    # Create an array of indices for the x-axis
    x_indices = np.arange(num_bars)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the bars for each list
    rects1 = ax.bar(x_indices, perf1, bar_width, label='Single Agent', color = "royalblue")
    rects2 = ax.bar(x_indices + bar_width, perf2, bar_width, label='2MARL', color = "sandybrown")
    rects3 = ax.bar(x_indices + 2 * bar_width, perf3, bar_width, label='3MARL', color = "yellowgreen")

    # Set the labels and title
    ax.set_xlabel('Models')
    ax.set_ylabel('Percentage of Improvement')
    # ax.set_title('Bar Chart with Multiple Lists')

    # Set the x-axis ticks and labels
    ax.set_xticks(x_indices + bar_width)
    ax.set_xticklabels(["5 bins", "7 bins", "difference", "baselines"])

    # Set the legend
    ax.legend(loc = 'lower left')
    plt.savefig('Comparison_all.png', dpi = 300)
    # Display the plot
    plt.show() 

def compare_three():
    iterations = [100,500,1000,2500,5000,10000]
    single_A = np.zeros((5,len(iterations)))
    multi_A = np.zeros((5,len(iterations)))
    
    single_B = np.zeros((5,len(iterations)))
    multi_B = np.zeros((5,len(iterations)))
    
    for i,n in enumerate(iterations):    
        costs_5_1, costs_7_1, costs_d_1, baseline_costs_1, bs_1 = testing.get_performances([n], 'A')

        single_A[:,i] = [costs_5_1, costs_7_1, costs_d_1, baseline_costs_1, bs_1]

        costs_5_1, costs_7_1, costs_d_1, baseline_costs_1, bs_1 = testing.get_performances([n], 'B')
        single_B[:,i] = [costs_5_1, costs_7_1, costs_d_1, baseline_costs_1, bs_1]


        costs_5_A3,costs_5_B3, costs_7_A3,costs_7_B3, costs_d_A3,costs_d_B3, baseline_costs_A3,baseline_costs_B3, bs_A3,bs_B3 = MA3_testing.get_performance([n])
        multi_A[:,i] = [costs_5_A3, costs_7_A3, costs_d_A3,baseline_costs_A3, bs_A3]
        multi_B[:,i] = [costs_5_B3, costs_7_B3, costs_d_B3, baseline_costs_B3, bs_B3]
    
    
    colors = ["lightcoral", "lightslategrey", "yellowgreen"]
    
    labels = ["MDP with 5 bins", "MDP with 7 bins","MDP with difference"]
    markers = ['^','s','x','o']
    ab = ['A', 'B']
    title = ['(a)', '(b)']
    fig = plt.figure(figsize=(12, 5))

    for j,x in enumerate(ab):
        multi = multi_A if x == 'A' else multi_B
        single = single_A if x == 'A' else single_B
        ax = fig.add_subplot(1,2,j+1)
        for r in range(3):
            ax.plot(single[r,], color = str(colors[r]), label = str(labels[r]), linestyle = "dashdot", marker = markers[r], markersize = 5)
            ax.plot(multi[r,], color = str(colors[r]), label = str(labels[r]), linestyle = "solid", marker = markers[r], markersize = 5)
            
    
        ax.plot(single[3,], label ="rule-based Baseline", color = "purple", linestyle = "dashed")
        ax.plot(multi[3,], label ="rule-based Baseline", color = "purple", linestyle = "dotted")
    
    
        ax.plot(multi[4,], label ="Baseline without ESS", color = "grey", linestyle = "dotted")
    
        ax.set_xticks(np.arange(0,len(iterations),1),labels = iterations)
        ax.set_xlabel('Number of training episodes')
        if x == 'B':
            plt.gca().yaxis.set_label_position('right')
            plt.gca().yaxis.set_ticks_position('right')
        ax.set_ylabel('Costs')
        ax.text(0.5, -0.15, title[j], transform=ax.transAxes, ha='center')

    
    # plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    custom_legend = [Line2D([0], [0], color='lightcoral', marker='^', linestyle='None'),
                     Line2D([0], [0], color='lightslategrey', marker='s', linestyle='None'),
                     Line2D([0], [0], color='yellowgreen', marker='x', linestyle='None'),
                     Line2D([0], [0], color='purple', linestyle='dotted'),
                     Line2D([0], [0],linestyle = 'dotted', color='grey'),
                     Line2D([0], [0], linestyle='dashdot', color = 'black'),
                     Line2D([0], [0], linestyle='solid', color = 'black'),
                     ]
    
    
    
    fig.legend(custom_legend, ['MDP with 5 bins', 'MDP with 7 bins','MDP with difference', 'rule-based baselines', 'baseline without ESS', 'Single Agent', '3MARL Agent'],bbox_to_anchor=(0.75, 1.0), fontsize = 'small', ncol = 3)

    
    plt.savefig("one_three_comparison.png", dpi = 300, bbox_inches = 'tight')


#plot_comparison()
# compare_three('A')
# plt.show()
compare_three()
plt.show()