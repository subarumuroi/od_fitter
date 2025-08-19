import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Your data
G_values = np.array([118.87, 103.33, 86.16, 67.80, 64.27, 59.95, 59.67,
                     53.08, 43.24, 30.54, 29.77, 20.27, 19.22, 11.15,
                     5.88, 4.98])
OD_values = np.array([15.44, 11.54, 7.41, 5.58, 4.43, 4.12, 3.77,
                      3, 1.87, 0.9, 0.855, 0.391, 0.345, 0.197,
                      0.090, 0.050])

# Define the model including Gblank as a parameter
def od_model(G, a, b, c, d, e, f, Gblank):
    Gnet = G - Gblank
    return a * Gnet**b + c * Gnet**d + e * Gnet**f

# Initial guesses: [a, b, c, d, e, f, Gblank]
initial_guess = [0.1, 1, 0.1, 1, 0.1, 1, min(G_values)/2]

# Fit the model
popt, pcov = curve_fit(od_model, G_values, OD_values, p0=initial_guess, maxfev=20000)

print("Fitted parameters [a,b,c,d,e,f,Gblank]:", popt)

# Generate smooth curve for plotting
G_fit = np.linspace(min(G_values), max(G_values), 200)
OD_fit = od_model(G_fit, *popt)

# Plot data vs fit
plt.scatter(G_values, OD_values, color='red', label='Data')
plt.plot(G_fit, OD_fit, color='blue', label='7-param fit')
plt.xlabel('GValue')
plt.ylabel('OD')
plt.legend()
plt.show()