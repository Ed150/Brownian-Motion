"""
Thermal motion lab: mean squared displacement (r^2 vs t) analysis.

Put the tracker text files (one per bead, columns: x, y in pixels) in the
DATA_FOLDER below and run this file. Results and figures go in OUTPUT_FOLDER.

Since the tracker gives (x, y) the motion is 2-D, so
    <r^2> = 4 D t     and     k = D * gamma / T,   gamma = 6 pi eta r
"""
import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
print("Project root:", PROJECT_ROOT)
DATA_FOLDER = PROJECT_ROOT /"Raw_Data/Position_Data/"
OUTPUT_FOLDER = PROJECT_ROOT / "Data Analysis" / "Results" / "Output"
SKIP_ROWS = 2            # number of header lines at the top of each file

# Constants from the lab manual
PIXEL = 0.1204           # um per pixel
PIXEL_ERR = 0.003
DT = 0.5                 # s between frames (2 frames per second)
DT_ERR = 0.03            # s
POS_ERR = 0.1            # um, error on each x or y position
DIAMETER = 1.9           # um
DIAMETER_ERR = 0.1
TEMP = 296.5             # K
TEMP_ERR = 0.5
VISC_20C = 1.00e-3       # Pa s at 20 C
VISC_ERR = 0.05          # fractional error on the viscosity
K_ACCEPTED = 1.38e-23    # J/K
R_GAS = 8.314            # J/(mol K)

# ----------------------------------------------------------------------------
# Constants worked out from the above
# ----------------------------------------------------------------------------
radius = DIAMETER / 2 * 1e-6                       # m
viscosity = VISC_20C * (1 - 0.02 * (TEMP - 293.15))  # drops 2% per degree above 20 C
gamma = 6 * np.pi * viscosity * radius             # Stokes drag, kg/s


def k_from_D(D):
    """Boltzmann constant from D given in um^2/s."""
    return D * 1e-12 * gamma / TEMP


os.makedirs(OUTPUT_FOLDER, exist_ok=True)
lines = []


def say(text=""):
    print(text)
    lines.append(text)


# ----------------------------------------------------------------------------
# Load the data
# ----------------------------------------------------------------------------
files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.txt")))
if len(files) == 0:
    raise SystemExit("No .txt files found in the folder '" + DATA_FOLDER + "'")

names = []
times = []
r2_all = []
for f in files:
    data = np.loadtxt(f, skiprows=SKIP_ROWS)
    x = data[:, 0] * PIXEL
    y = data[:, 1] * PIXEL
    t = np.arange(len(x)) * DT
    r2 = (x - x[0]) ** 2 + (y - y[0]) ** 2         # squared distance from the start
    names.append(os.path.splitext(os.path.basename(f))[0])
    times.append(t)
    r2_all.append(r2)

n_beads = len(names)
say("Loaded %d beads" % n_beads)
say("viscosity at %.1f K = %.4f cP, gamma = %.3e kg/s" % (TEMP, viscosity * 1e3, gamma))
say()

# ----------------------------------------------------------------------------
# Straight line fit of r^2 vs t for each bead
# ----------------------------------------------------------------------------
slopes = []
intercepts = []
D_values = []
k_values = []
r2_errors = []

say("Fit of r^2 = a + 4*D*t for each bead")
say("(chi2_red uses the position error on each point as the uncertainty)")
say("%-10s %10s %10s %12s %9s %7s %9s" % ("bead", "D (um2/s)", "err D", "k (J/K)", "% diff", "R^2", "chi2_red"))

for i in range(n_beads):
    t = times[i]
    r2 = r2_all[i]
    fit = stats.linregress(t, r2)

    # error on each r^2 point from the error on x and y (they are differences of
    # two positions so each has an error of sqrt(2)*POS_ERR)
    err = 2 * np.sqrt(2) * POS_ERR * np.sqrt(r2 + 2 * POS_ERR ** 2)
    residuals = r2 - (fit.intercept + fit.slope * t)
    chi2_red = np.sum((residuals / err) ** 2) / (len(t) - 2)

    D = fit.slope / 4
    k = k_from_D(D)
    slopes.append(fit.slope)
    intercepts.append(fit.intercept)
    D_values.append(D)
    k_values.append(k)
    r2_errors.append(err)
    say("%-10s %10.4f %10.4f %12.3e %9.1f %7.3f %9.1f" % (
        names[i], D, fit.stderr / 4, k, 100 * (k - K_ACCEPTED) / K_ACCEPTED,
        fit.rvalue ** 2, chi2_red))

D_values = np.array(D_values)
k_values = np.array(k_values)

# ----------------------------------------------------------------------------
# Average over the beads
# ----------------------------------------------------------------------------
D_mean = np.mean(D_values)
D_std = np.std(D_values, ddof=1)
D_sem = D_std / np.sqrt(n_beads)

say()
say("Average over beads")
say("  D = %.4f +/- %.4f um^2/s (standard error), standard deviation = %.4f" % (D_mean, D_sem, D_std))

# Ensemble average: average r^2 over all beads at each time, then fit a line.
# (The points are not independent since they come from the same paths, so
# the chi2 and the error on the slope are only a rough guide.)
n_points = min(len(t) for t in times)
t_ens = times[0][:n_points]
r2_stack = np.array([r2[:n_points] for r2 in r2_all])
r2_mean = r2_stack.mean(axis=0)
r2_sem = r2_stack.std(axis=0, ddof=1) / np.sqrt(n_beads)


def line(t, a, b):
    return a + b * t


# skip the t = 0 point because its error bar is zero
popt, pcov = curve_fit(line, t_ens[1:], r2_mean[1:], sigma=r2_sem[1:], absolute_sigma=True)
res = r2_mean[1:] - line(t_ens[1:], *popt)
chi2_ens = np.sum((res / r2_sem[1:]) ** 2) / (n_points - 1 - 2)
slope_ens = popt[1]
slope_ens_err = np.sqrt(pcov[1, 1])
if chi2_ens > 1:
    slope_ens_err = slope_ens_err * np.sqrt(chi2_ens)   # scale error up if chi2_red > 1
D_ens = slope_ens / 4
D_ens_err = slope_ens_err / 4

say()
say("Fit to the average r^2 (all beads) vs t")
say("  slope = %.3f +/- %.3f um^2/s, intercept = %.3f +/- %.3f um^2" % (
    slope_ens, slope_ens_err, popt[0], np.sqrt(pcov[0, 0])))
say("  reduced chi^2 = %.2f" % chi2_ens)
say("  D = %.4f +/- %.4f um^2/s" % (D_ens, D_ens_err))

# ----------------------------------------------------------------------------
# Boltzmann constant and its uncertainty
# ----------------------------------------------------------------------------
# relative errors that add in quadrature (D has units of length^2 so the
# pixel size error counts twice)
rel_errors = {
    "fit (standard error of D)": D_sem / D_mean,
    "pixel size": 2 * PIXEL_ERR / PIXEL,
    "viscosity": np.sqrt(VISC_ERR ** 2 + (0.02 * TEMP_ERR / (1 - 0.02 * (TEMP - 293.15))) ** 2),
    "bead radius": DIAMETER_ERR / DIAMETER,
    "temperature": TEMP_ERR / TEMP,
    "time between frames": DT_ERR / DT,
}
total_rel = np.sqrt(sum(v ** 2 for v in rel_errors.values()))

k_mean = k_from_D(D_mean)
k_err = k_mean * total_rel
percent_diff = 100 * (k_mean - K_ACCEPTED) / K_ACCEPTED
n_sigma = (k_mean - K_ACCEPTED) / k_err
N_A = R_GAS / k_mean
N_A_err = N_A * total_rel

say()
say("Error budget for k (relative uncertainty)")
for name in rel_errors:
    say("  %-28s %5.1f %%" % (name, 100 * rel_errors[name]))
say("  %-28s %5.1f %%" % ("total", 100 * total_rel))

say()
say("Result from r^2 vs t (mean of the beads)")
say("  k   = (%.2f +/- %.2f) e-23 J/K" % (k_mean / 1e-23, k_err / 1e-23))
say("  accepted k = 1.38e-23 J/K, difference = %.1f %% (%.2f sigma)" % (percent_diff, n_sigma))
say("  N_A = (%.2f +/- %.2f) e23 per mol (accepted 6.022e23)" % (N_A / 1e23, N_A_err / 1e23))

k_ens = k_from_D(D_ens)
# same systematic errors as above, but with the error from the fit of the average
other_rel = np.sqrt(total_rel ** 2 - rel_errors["fit (standard error of D)"] ** 2)
k_ens_err = k_ens * np.sqrt(other_rel ** 2 + (D_ens_err / D_ens) ** 2)
say()
say("Result from the fit to the average r^2")
say("  k   = (%.2f +/- %.2f) e-23 J/K, difference = %.1f %%" % (
    k_ens / 1e-23, k_ens_err / 1e-23, 100 * (k_ens - K_ACCEPTED) / K_ACCEPTED))

with open(os.path.join(OUTPUT_FOLDER, "results.txt"), "w") as out:
    out.write("\n".join(lines) + "\n")

# save the per-bead numbers as a table
table = np.column_stack([D_values, k_values])
np.savetxt(os.path.join(OUTPUT_FOLDER, "per_bead_results.csv"), table, delimiter=",",
           header="D (um^2/s),k (J/K)", comments="")

# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
# 1. all beads on one plot
plt.figure(figsize=(7, 5))
for i in range(n_beads):
    plt.plot(times[i], r2_all[i], ".", markersize=3, color="C%d" % (i % 10))
    plt.plot(times[i], intercepts[i] + slopes[i] * times[i], "-", linewidth=1, color="C%d" % (i % 10))
plt.xlabel("Time (s)")
plt.ylabel(r"Squared distance from start, $r^2$ ($\mu$m$^2$)")
plt.title("Squared distance vs time for all beads\n(dots: data, lines: straight-line fits)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "fig1_r2_all_beads.png"), dpi=200)
plt.close()

# 2. one panel per bead, with error bars
ncols = 4
nrows = int(np.ceil(n_beads / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
for i in range(nrows * ncols):
    ax = axes[i // ncols, i % ncols]
    if i >= n_beads:
        ax.axis("off")
        continue
    ax.errorbar(times[i], r2_all[i], yerr=r2_errors[i], fmt="k.", markersize=3,
                elinewidth=0.5, label="data")
    ax.plot(times[i], intercepts[i] + slopes[i] * times[i], "r-", label="fit")
    ax.set_title("%s: D = %.3f $\\mu$m$^2$/s" % (names[i], D_values[i]), fontsize=9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$r^2$ ($\mu$m$^2$)")
    ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=8)
fig.suptitle("Squared distance vs time for each bead (error bars from the position uncertainty)")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_FOLDER, "fig2_r2_each_bead.png"), dpi=200)
plt.close(fig)

# 3. average over all beads
plt.figure(figsize=(7, 5))
plt.errorbar(t_ens, r2_mean, yerr=r2_sem, fmt="ko", markersize=3, elinewidth=0.8,
             label="average of %d beads (error bar = standard error)" % n_beads)
plt.plot(t_ens, line(t_ens, *popt), "r-",
         label="fit: slope = %.2f $\\mu$m$^2$/s, D = %.3f $\\mu$m$^2$/s" % (slope_ens, D_ens))
plt.xlabel("Time (s)")
plt.ylabel(r"Average squared distance, $\langle r^2 \rangle$ ($\mu$m$^2$)")
plt.title("Average squared distance vs time")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "fig3_average_r2.png"), dpi=200)
plt.close()

# 4. k for each bead
plt.figure(figsize=(7, 4.5))
plt.bar(np.arange(1, n_beads + 1), k_values, color="C0")
plt.axhline(K_ACCEPTED, color="r", linestyle="--", label="accepted k = 1.38e-23 J/K")
plt.axhline(k_mean, color="k", linestyle=":", label="mean = %.2e J/K" % k_mean)
plt.xticks(np.arange(1, n_beads + 1))
plt.xlabel("Bead number")
plt.ylabel("Boltzmann constant k (J/K)")
plt.title("k from the fit to each bead")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "fig4_k_each_bead.png"), dpi=200)
plt.close()

print()
print("Figures and results.txt saved in '%s'" % OUTPUT_FOLDER)