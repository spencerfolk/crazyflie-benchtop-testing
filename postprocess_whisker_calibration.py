import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import linregress

alpha = 0.3
scatter_size = 5

# Path to your logged CSV
data_dir = os.path.join("data", "cf_whisker")
csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv") and "whisker" in f]
if not csv_files:
    raise FileNotFoundError("No CSV files found in data/cf_whisker")

filename = csv_files[0]
print(f"Loading {filename}")

# Load data
df = pd.read_csv(filename).dropna(subset=["U", "V"])
U = df["U"].values
V = df["V"].values
Bx = df["Bx"].values
By = df["By"].values

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# --- Top-left: Bx vs U (with linear fit) ---
slope, intercept, r, p, stderr = linregress(U, Bx)
Bx_pred = slope * U + intercept
axs[0, 0].scatter(U, Bx, s=scatter_size, c='k', alpha=alpha, label="Data")
axs[0, 0].plot(U, Bx_pred, color="red", lw=2, label="Fit")
axs[0, 0].set_xlabel("U (anemometer)")
axs[0, 0].set_ylabel("Bx (Crazyflie)")
axs[0, 0].set_title(f"Bx vs U (R²={r**2:.3f})")
# Add equation as text
eq_text = f"Bx = {slope:.3f}*U + {intercept:.3f}"
axs[0, 0].text(0.05, 0.95, eq_text, transform=axs[0, 0].transAxes,
               fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))
axs[0, 0].legend()

# --- Top-right: Bx vs V (scatter only) ---
axs[0, 1].scatter(V, Bx, s=scatter_size, c='k', alpha=alpha)
axs[0, 1].set_xlabel("V (anemometer)")
axs[0, 1].set_ylabel("Bx (Crazyflie)")
axs[0, 1].set_title("Bx vs V")

# --- Bottom-left: By vs U (scatter only) ---
axs[1, 0].scatter(U, By, s=scatter_size, c='k', alpha=alpha)
axs[1, 0].set_xlabel("U (anemometer)")
axs[1, 0].set_ylabel("By (Crazyflie)")
axs[1, 0].set_title("By vs U")

# --- Bottom-right: By vs V (with linear fit) ---
slope, intercept, r, p, stderr = linregress(V, By)
By_pred = slope * V + intercept
axs[1, 1].scatter(V, By, s=scatter_size, c='k', alpha=alpha, label="Data")
axs[1, 1].plot(V, By_pred, color="red", lw=2, label="Fit")
axs[1, 1].set_xlabel("V (anemometer)")
axs[1, 1].set_ylabel("By (Crazyflie)")
axs[1, 1].set_title(f"By vs V (R²={r**2:.3f})")
# Add equation as text
eq_text = f"By = {slope:.3f}*V + {intercept:.3f}"
axs[1, 1].text(0.05, 0.95, eq_text, transform=axs[1, 1].transAxes,
               fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))
axs[1, 1].legend()

plt.tight_layout()
plt.show()
