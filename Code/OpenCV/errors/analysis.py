import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import os
from scipy.stats import norm, chisquare, sem


# options are: "uncertainty_data_10_pics_each.npz"
# or "uncertainty_data_10000.npz"

# open the data in .npz file
outfile = "uncertainty_data_5000.npz"
npzfile = np.load(outfile)

folder = "graphs_5000_20_images"

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

    stdError = sem(data)

    # draw data
    plt.figure()

    n, bins, patches = plt.hist(data, 75, density=True)

    # Line of Best Fit
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
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

    plt.savefig("{}/{}.png".format(folder, xlabel+"_50mm_5000_combinations_20images"))
    # plt.show()

    return mean, std, stdError

for key, value in parameterDict.items():
    mean, std, stdError = draw_graph(value, xlabel=key)

    print("\nData for parameter {}".format(key))
    print("The mean is {:3f} +- {:.1g}".format(mean, stdError), "\n")
    print("Standard Deviation: {:.3f}".format(std), "\n")
    print("___________________________________________________________\n")

plt.show()