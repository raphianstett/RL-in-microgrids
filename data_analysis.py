import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



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
    

# plot_cons_mean()
# plot_prod_mean()
# plt.show()
# generate step consumption
# 400 Wh 7 - 14, 300 Wh 14 - 19, 400 Wh 19 - 22



