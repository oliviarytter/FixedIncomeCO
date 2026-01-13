"""
Problem Set Week 2 - Problem 2
ACTUAL SOLUTION - BOOTSTRAPS FROM DATA
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/mnt/project')

import fixed_income_derivatives_E2025 as fid
from scipy.optimize import minimize

# Data
libor_3m = 0.01570161
libor_6m = 0.01980204

data = [
    {"instrument": "libor", "maturity": 0.25, "rate": libor_3m},
    {"instrument": "libor", "maturity": 0.5, "rate": libor_6m},
    {"instrument": "fixed_rate_bond", "maturity": 1.0, "coupon": 0.04, "coupon_freq": "quarterly", "price": 102.33689177, "principal": 100},
    {"instrument": "fixed_rate_bond", "maturity": 1.0, "coupon": 0.05, "coupon_freq": "semiannual", "price": 104.80430234, "principal": 100},
    {"instrument": "fixed_rate_bond", "maturity": 1.5, "coupon": 0.05, "coupon_freq": "semiannual", "price": 105.1615306, "principal": 100},
    {"instrument": "fixed_rate_bond", "maturity": 1.5, "coupon": 0.06, "coupon_freq": "quarterly", "price": 105.6581905, "principal": 100},
    {"instrument": "fixed_rate_bond", "maturity": 2.0, "coupon": 0.05, "coupon_freq": "quarterly", "price": 104.028999992, "principal": 100},
    {"instrument": "fixed_rate_bond", "maturity": 2.0, "coupon": 0.03, "coupon_freq": "annual", "price": 101.82604116, "principal": 100}
]

# PART B: Bootstrap
T_fit, R_fit = fid.zcb_curve_fit(data, interpolation_options={"method": "linear"})
p_fit = fid.zcb_prices_from_spot_rates(T_fit, R_fit, method="continuous")

# PART C: Forward rates
f_fit = fid.forward_rates_from_zcb_prices(T_fit, p_fit, horizon=1, method="continuous")

# PART D: FRN
frn_price = 100.0

# PART E: Par swap rate
R_swap = fid.swap_rate_from_zcb_prices(0, 0, 2.0, "semiannual", T_fit, p_fit)
if isinstance(R_swap, tuple):
    R_swap = R_swap[0]

# PART F: Compare
avg_fwd = np.nanmean(f_fit)

# Plot
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0,0].plot(T_fit, p_fit, 'b-o')
ax[0,0].set_title('ZCB Prices')
ax[0,1].plot(T_fit, np.array(R_fit)*100, 'g-s')
ax[0,1].set_title('Spot Rates')
ax[1,0].plot(T_fit[~np.isnan(f_fit)], f_fit[~np.isnan(f_fit)]*100, 'r-^')
ax[1,0].set_title('Forward Rates')
ax[1,1].plot(T_fit, np.array(R_fit)*100, 'g-', label='Spot')
ax[1,1].plot(T_fit[~np.isnan(f_fit)], f_fit[~np.isnan(f_fit)]*100, 'r--', label='Forward')
ax[1,1].legend()
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/problem2.png', dpi=150)

print(f"FRN Price: {frn_price}")
print(f"Par Swap Rate: {R_swap*100:.4f}%")
print(f"Avg Forward: {avg_fwd*100:.4f}%")
