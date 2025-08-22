import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Your data ---
G = np.array([126.14, 117.6, 111.14, 96.11, 93.32, 86.29, 84.6,
                78, 63.25, 48.23, 48.47, 32.19, 34.96, 24.63,
                15.11, 11.88], dtype=float)

OD = np.array([15.44, 11.54, 7.41, 5.58, 4.43, 4.12, 3.77,
               3.00, 1.87, 0.90, 0.855, 0.391, 0.345, 0.197,
               0.090, 0.050], dtype=float)

# --- Model: OD = a*(G-Gblank)^b + c*(G-Gblank)^d + e*(G-Gblank)^f ---
def growth_model(G, a, b, c, d, e, f, Gblank):
    X = G - Gblank
    return a*X**b + c*X**d + e*X**f

# --- Bounds keep exponents reasonable and ensure X = G - Gblank > 0 ---
eps = 1e-6
lower = [0.0, 0.10, 0.0, 0.10, 0.0, 0.10, 0.0]
upper = [np.inf, 6.0, np.inf, 6.0, np.inf, 6.0, float(np.min(G) - eps)]

# --- Initial guesses (generic but sensible) ---
p0 = [0.05, 0.8, 0.003, 1.6, 1e-7, 3.2, float(np.min(G) - 0.5)]

# --- Fit ---
params, cov = curve_fit(growth_model, G, OD, p0=p0, bounds=(lower, upper), maxfev=200000)

# --- Print parameters ---
names = ["a","b","c","d","e","f","Gblank"]
for n, v in zip(names, params):
    print(f"{n} = {v:.12g}")

# --- Fit quality ---
pred = growth_model(G, *params)
ss_res = float(np.sum((OD - pred)**2))
ss_tot = float(np.sum((OD - np.mean(OD))**2))
r2 = 1.0 - ss_res/ss_tot
rmse = float(np.sqrt(ss_res/len(OD)))
print(f"R2 = {r2:.6f}, RMSE = {rmse:.6f}")

# --- Plot ---
G_fit = np.linspace(G.min(), G.max(), 500)
OD_fit = growth_model(G_fit, *params)
plt.scatter(G, OD, label="Data")
plt.plot(G_fit, OD_fit, label="Fitted curve")
plt.xlabel("G (Green value)")
plt.ylabel("OD")
plt.legend()
plt.show()
