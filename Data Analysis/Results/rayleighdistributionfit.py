from scipy import stats
from scipy.optimize import curve_fit
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def rayleigh(r, t, D):
    return (r / (2 * D * t)) * (np.exp((-r**2)/(4 * D * t)))

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent #pulls the local repository path automatically from computer assuming this file is 2 sub-folders deep

    px = 0.12048
    dpx = 0.003
    step_sizes = np.array([])
    for i in range(1, 24):
        df = pd.read_table(PROJECT_ROOT / f"Raw_Data/Position_Data/data{i}.txt", header=1)
        df.columns = df.columns.str.strip()
        x = df['X (pixels)'].to_numpy()
        y = df['Y (pixels)'].to_numpy()
        x1 = x[1:]
        y1 = y[1:]
        x1 = np.append(x1, 0)
        y1 = np.append(y1, 0)
        x = x*px
        y = y*px
        x1 = x1*px
        y1 = y1*px
        distances = distance(x, y, x1, y1)
        step_sizes = np.append(step_sizes, distances[:-1])

    # std_dev = np.std(step_sizes)
    # max_value = max(step_sizes)
    # mean = np.mean(step_sizes)
    # while np.abs(max_value - mean) > 3 * std_dev:
    #     max_index = np.argmax(step_sizes)
    #     step_sizes = np.delete(step_sizes, max_index)
    #     std_dev = np.std(step_sizes)
    #     max_value = max(step_sizes)
        # print(max_value, max_index)
    
    # print(max(step_sizes))
    # print(step_sizes)
    # print(len(step_sizes))

    n = 100

    freq, bin_edges = np.histogram(step_sizes, bins=n, density=True)
    bins = (bin_edges[:-1] + bin_edges[1:])/2

    dt = 0.5
    D_guess = np.mean(step_sizes**2) / (4 * dt)
    y, cov = curve_fit(lambda r, D: rayleigh(r, dt, D), bins, freq, p0=[D_guess], bounds=(0, np.inf))

    D_fit = y[0]
    r_fit = np.linspace(0, max(step_sizes), n)
    rayleigh_fit = rayleigh(r_fit, dt, D_fit)
    print("D fit from curve:", D_fit)

    plt.hist(step_sizes, bins=n, density=True)
    plt.plot(r_fit, rayleigh_fit, label="Rayleigh Curve Fit")
    plt.title("Rayleigh Distribution")
    plt.xlabel("Step Sizes (um)")
    plt.ylabel("Frequency")
    plt.savefig("Rayleigh Distribution Fit")
    plt.show()