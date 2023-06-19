import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

dat = pd.read_csv('household_with_pv.csv', delimiter=";")


dat = dat.fillna(0)

cons = dat["Consumption"] 
prod = dat["Production"]


# determine buckets for consumption
cons = np.sort(cons)

idx1 = round((len(cons) / 3))
idx2 = 2*idx1

# each bucket contains 2920 data points
# print(len(cons[idx1:idx2]))
# print(cons[idx1])
# print(cons[idx2])

# determine buckets for production
prod = np.sort(prod)

# print("production: ")
# print(prod)

idx1_p = round((len(prod) / 3))

# print("first third: ")
# print(idx1_p)
# print(prod[idx1_p])


idx2_p = 2*idx1

# print("second third: ")
# print(idx2_p)
# print(prod[idx2_p])

# print("production zero: ")

zeros = (prod == 0.0).sum()
non_zeros_low = np.logical_and(prod > 0.0, prod < 492)

# print(zeros)
# print(non_zeros_low.sum())

idx_p_half = round(zeros + ((len(prod) - zeros) / 2))
# print(idx_p_half)

# print("spaltespunkt: ")
# print(prod[idx_p_half])

prod_nonzeros = prod[4444:]

low = min(prod_nonzeros)
high = max(prod_nonzeros)
n = int((high- low)/6)
intervals = [0, low, low + n, low + 2*n, low + 3*n, low + 4*n, low + 5*n, low + 6*n]
print(intervals)
# print(dat)
# print(cons)

#plt.plot(cons)

# print("consumption mean: ")
# print(np.mean(cons))

# plt.hist(prod_nonzeros, bins = [0, prod[idx_p_half], prod_nonzeros.max()], rwidth= 0.7)
# plt.show()

# print(dat)

# convert Dates in dat to datetime format

dat["Dates"] = pd.to_datetime(dat["Dates"], dayfirst=True)
# print(dat)

dat["Date"] = dat["Dates"].dt.date
dat["Time"] = dat["Dates"].dt.hour

# print(dat)

# export to csv
#dat.to_csv('household_with_pv_new.csv', index=False)

# plot mean consumption and production for the day

# for production
# print(np.median(prod))

def get_prod_mean():
    # mean production for one day
    prod_mean = [0]*24

    time = dat["Time"]
    #print(time)
    print(len(prod_mean))

    for i in range(0, len(prod)-1):
        #print(time[i])
        prod_mean[time[i]] += dat["Production"][i]
    print(prod_mean)

    prod_mean[:] = [x / 365 for x in prod_mean]
    print(prod_mean)
    return prod_mean


# mean consumption for one day
def get_cons_mean():
    cons_mean = [0]*24

    time = dat["Time"]
    #print(time)
    print(len(cons_mean))

    for i in range(0, len(prod)-1):
        # print(time[i])
        cons_mean[time[i]] += dat["Consumption"][i]
    
    cons_mean[:] = [x / 365 for x in cons_mean]
    return cons_mean

def plot_cons_mean():
    plt.plot(get_cons_mean())
    

def plot_prod_mean():
    plt.plot(get_prod_mean())

def changes():
    return 0

# plot_cons_mean()
# plot_prod_mean()
# plt.show()
# generate step consumption
# 400 Wh 7 - 14, 300 Wh 14 - 19, 400 Wh 19 - 22

## Binning for finer discretization

# dat_prod = dat[['Production', 'Time']]

# labels = ['none', 'low', 'average', 'high', 'very high']
# dat['bin_qcut'] = pd.qcut(dat['Production'], q=5, precision=1, labels=labels)

cons = np.array(dat["Consumption"])
cons.sort()
split_points = np.array_split(cons, 10)
bins_cons = [[]]
bins_cons.extend(split_points)
#print(split_points)
# print(bins_cons)
bin_edges = [split_points[i][0] for i in range(10)]
#print(bin_edges)

# print(prod_nonzeros)
nonzeros = np.array(prod_nonzeros)
nonzeros.sort()
split_points = np.array_split(nonzeros, 9)


# Create 7 bins where the first bin contains only zeros
bins = [[]]
bins.extend(split_points)

# Add the zero values to the first bin
bins[0] = [0] * (4444 - 1)
bin_edges = [bins[i][0] for i in range(10)]
# print(bin_edges)
# print(bins)
# Output the bins
# for i, bin_values in enumerate(bins):
#     print(f"Bin {i + 1}: {bin_values}")

# hist, edges = np.histogram(nonzeros, bins = 6)
# print(edges)
# print(hist)
# plt.stairs(hist, edges)
# print(np.flip(hist))
#plt.hist(nonzeros, bins = hist)
# plt.show()

# prod = np.sort(dat['Consumption'])
# n = int(max(prod) / 7) 
# print(n)
# steps = [0, n, 2*n, 3*n, 4*n, 5*n, 6*n, 7*n]
# print(steps)
# intervals = [0, n, 2*n, 3*n, 4*n, 5*n, 6*n, 7*n]
# # intervals = [prod[0], prod[n], prod[2*n], prod[3*n], prod[4*n], prod[5*n], prod[6*n], prod[7*n]]
# print(intervals)

def check_prediction(prod):
    x = []
    for i in range(len(prod)-1):
        if(prod[i] == 0):
            x.append(0)
            
        else:
            x.append(prod[i+1]/prod[i])
        
    # x = np.array(x)
    # x[np.isnan(x)] = 0
    # x[np.isinf(x)] = 0
    return x

def check_difference(cons, prod):
    q = [0]*len(cons)
    # prod = list(prod)
    # cons = list(cons)
    # prod[prod == 0] = 1
    # cons[cons == 0] = 1
    #for i in range(len(cons)):
        # diff = max((prod[i] / cons[i]),(cons[i]/ prod[i])) if (prod[i] != 0 and cons[i] != 0) else 1
        # q[i] = quot
    
    diff = [prod[i] - cons[i] for i in range(len(prod))]
    diff_bool = [diff[i] < -200   for i in range(len(diff))]
    #print(len(diff_bool))
    
    return diff

def check_difference2(cons,prod):
    prod = list(prod)
    prod = [1 if x == 0 else x for x in prod]
    cons = list(cons)
    cons = [1 if x == 0 else x for x in cons]

    diff = [prod[i+1]/prod[i] for i in range(len(prod)-1)]
    #[(prod[i+1]/prod[i]) for i in range(len(prod)-1)]

    return diff

b = check_difference(dat["Consumption"], dat['Production'])
print("mean")
print(np.mean(b))
print(np.median(b))
print("max and min: ")
print(max(b))
print(min(b))

d = np.sort(b)
# n = int(len(b) / 10) 
n = int((max(b) - min(b)) / 10)
m = min(b)
print((max(b) - min(b)))
print(n)
steps = [m + 0*n ,m + n,m + 2*n,m + 3*n,m + 4*n,m + 5*n,m + 6*n,m + 7*n,m + 8*n, m + 9*n, m + 10*n]
print(steps)
intervals = [d[0], d[n], d[2*n], d[3*n], d[4*n], d[5*n], d[6*n], d[7*n], d[8*n], d[9*n], d[10*n - 1]]
print(intervals)
# print(b.count(True))
#print(check_difference(dat["Consumption"], dat['Production']))
# print(max(check_difference(dat["Consumption"], dat['Production'])))
# print(np.mean(check_difference(dat["Consumption"], dat['Production'])))
# print(np.median(check_difference(dat["Consumption"], dat['Production'])))
plt.plot(check_difference(dat["Consumption"], dat['Production'])[:240])
plt.show()

# plt.plot(check_difference(dat["Consumption"], dat['Production'])[3600:3624])
# plt.plot(check_difference2(dat["Consumption"], dat['Production'])[3600:3624])
# # print(check_difference2(dat["Consumption"], dat['Production'])[3600:3624])

# plt.show()
max_battery = 4
step_size = 0.5 
#battery = [*range(0,max_battery+1,0.5)]
battery = list(np.arange(0.0, float(max_battery) + step_size, step_size))
# print(battery)
# print(len(battery))

def get_battery_id(battery):
    return int(battery * 2) 

## Plotting data exemplary
# print(len(dat["Consumption"]))


hours = np.arange(500, 741)  # Assuming each value represents a day
days = (hours - 500) / 24  # Convert days to hours
x_ticks = days
x_ticklabels = hours

plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "Arial"

plt.figure(figsize=(10, 5))

plt.plot(dat["Consumption"][0:168], color="red", linestyle="-", label="Demand")
plt.plot(dat["Production"][0:168], color="green", linestyle="dashdot", label="Production")

plt.xlabel("Days")
plt.xticks([], [])
plt.ylabel("Value in Wh")
plt.title("Demand and Production")
plt.legend()

# plt.show()
## figure to compare cases in data and in baseline
# charge_high, charge_low, nothing, discharge_low, discharge_high
in_data = [2028, 447, 741, 5487, 60]
in_baseline = [1354, 461, 3793, 3138, 13]

num_bins = min(len(in_data), len(in_baseline))

# Set the bar width for each pairwise value
bar_width = 0.35

# Calculate the x-axis positions for the bars
x = range(num_bins)

# Plot the histogram
fig, ax = plt.subplots()
rects1 = ax.bar(x, in_data[:num_bins], bar_width, color='blue', label='in_data')
rects2 = ax.bar([i + bar_width for i in x], in_baseline[:num_bins], bar_width, color='orange', label='in_baseline')

# Add labels, title, and legend
ax.set_xlabel('Actions')
ax.set_ylabel('Value')
ax.set_title('Histogram of Pairwise Values')
ax.set_xticks([i + bar_width/2 for i in x])
ax.set_xticklabels(x)
ax.legend()
#plt.show()

print(np.median(dat["Production"]))
# print(dat["Date"][5880:6048])

