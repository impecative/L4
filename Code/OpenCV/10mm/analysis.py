import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import os
from scipy.stats import norm, chisquare, sem


# open the data in .npz file
# 10 images each, 30000 repeats : "uncertainty_data_30000_10_images.npz"
# 20 images each, 5000 repeats : "uncertainty_data_5000.npz"
# 10 images each, 500 repeats : "uncertainty_data_500_10_images.npz"

outfile = "uncertainty_data_500_10_images.npz"
npzfile = np.load(outfile)

folder = "graphs_500"

if not os.path.exists(folder):
    os.mkdir(folder)

# extract the arrays
cxs = npzfile["cxs"]
cys = npzfile["cys"]
newcxs = npzfile["newcxs"]
newcys = npzfile["newcys"]
fxs = npzfile["fxs"]
fys = npzfile["fys"]
newfxs = npzfile["newfxs"]
newfys = npzfile["newfys"]
k1s = npzfile["k1s"]
k2s = npzfile["k2s"]
k3s = npzfile["k3s"]
p1s = npzfile["p1s"]
p2s = npzfile["p2s"]

parameterDict = {
    "cxs"   :  cxs ,
    "cys"     : cys,
    "newcxs"  : newcxs,
    "newcys"  : newcys,
    "fxs"     : fxs,
    "fys"     : fys,
    "newfxs"  : newfxs,
    "newfys"  : newfys,
    "k1s"     : k1s,
    "k2s"     : k2s,
    "k3s"     : k3s,
    "p1s"     : p1s,
    "p2s"     : p2s,
}


def draw_graph(data, xlabel=""):
    mean = np.mean(data)
    std = np.std(data)

    mu, std = norm.fit(data)

    final_xs = [x for x in data if ((x > mean - 3 * std) and (x < mean + 3 * std))]

    mu, std = norm.fit(final_xs)
    stdError = sem(data)
    #print("Standard Error is {}".format(stdError))

    # draw data
    plt.figure()

    #n, bins, patches = plt.hist(data, 50, color="green", edgecolor="black", density=True)
    n2, bins2, patches2 = plt.hist(final_xs, 50, edgecolor="black", density=True, alpha=1)

    # Line of Best Fit
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000)
    # y = ((1 / (np.sqrt(2 * np.pi) * std)) *
    #  np.exp(-(x-mean)**2 /(2 * std**2)))

    # plt.plot(x, y, '--')
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)

    plt.xlabel(xlabel)
    plt.ylabel("Probability Density")
    plt.grid(True)

    # plot the mean
    #plt.vlines(mean, 0, 1, linestyles="--", alpha=0.3)

    plt.savefig("{}/{}.png".format(folder, xlabel+"_10mm_500_combinations"))
    # plt.show()

    return mean, std, stdError

for key, value in parameterDict.items():
    mean, std, stdError = draw_graph(value, xlabel=key)

    print("Data for parameter: {}\n".format(key))
    print("The mean is {:3f} +- {:.2g}".format(mean, stdError), "\n")
    print("Standard Deviation: {:.3f}".format(std), "\n")
    print("_________________________________________________________________")

#plt.show()