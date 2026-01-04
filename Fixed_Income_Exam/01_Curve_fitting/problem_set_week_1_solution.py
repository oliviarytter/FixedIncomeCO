import numpy as np
import fixed_income_derivatives_E2025 as fid
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds

def price_fixed_rate_bond_from_ytm(y,T,C):
    price = 0
    N = len(T)
    for i in range(0,N):
        price += C[i]/(1+y)**T[i]
    return price

def ytm_obj(y,pv,T,C):
    pv_hat = price_fixed_rate_bond_from_ytm(y,T,C)
    se = (pv-pv_hat)**2
    return se

def ytm_obj_d(y,pv,T,C):
    N = len(T)
    pv_hat = price_fixed_rate_bond_from_ytm(y,T,C)
    dpv_dy = 0
    for i in range(0,N):
        dpv_dy += -C[i]*T[i]/(1+y)**(T[i]+1)
    se_dy = -2*(pv-pv_hat)*dpv_dy
    return se_dy

def ytm_obj_dd(y,pv,T,C):
    N = len(T)
    pv_hat = price_fixed_rate_bond_from_ytm(y,T,C)
    dpv_dy, ddpv_ddy = 0, 0
    for i in range(0,N):
        dpv_dy += -C[i]*T[i]/(1+y)**(T[i]+1)
        ddpv_ddy += C[i]*T[i]*(T[i]+1)/(1+y)**(T[i]+2)
    se_ddy = 2*(pv_hat-pv)*ddpv_ddy + 2*(dpv_dy)**2
    return se_ddy

R = 0.06
K = 100
T_N = 10
alpha = 0.5
pv = 98.74

N = int(T_N/alpha+1)
T = np.array([i*alpha for i in range(0,N)])
C = np.zeros([N])
for i in range(0,N):
    C[i] += R*alpha*K
C[-1] += K

# Problem 1c - Finding the yield to maturoty using 'nelder-mead'
y_init = R
args = (pv, T, C)
result = minimize(ytm_obj,y_init,args = args,method = 'nelder-mead',options={'xatol': 1e-8, 'disp': False})
y = result.x[0]
print(f"nelder-mead. ytm: {y}")

# Problem 1c - Finding the yield to maturoty using 'powell'
y_init = R
args = (pv, T, C)
result = minimize(ytm_obj,y_init,args = args,method = 'powell',options={'xatol': 1e-8, 'disp': False})
y = result.x[0]
print(f"powell. ytm: {y}")

# Problem 1e - Finding the yield to maturity using the first order derivative and 'BFGS'
y_init = R
args = (pv, T, C)
result = minimize(ytm_obj,y_init,args = args,method = 'BFGS',jac = ytm_obj_d)
y = result.x[0]
print(f"BFGS with 1. order derivative. ytm: {y}")

# Problem 1f - Finding the yield to maturoty using the first and second order derivative and 'Newton-CG'
y_init = R
args = (pv, T, C)
result = minimize(ytm_obj,y_init,args = args,method = 'Newton-CG',jac = ytm_obj_d, hess = ytm_obj_dd)
y = result.x[0]
print(f"Newton_CG with 1. and 2. order derivative. ytm: {y}")

# Problem 1g - Finding the yield to maturoty using the first and second order derivative and 'Newton-CG' while imposing an y.
y_init = R
args = (pv, T, C)
lb, ub = 0.08,0.12
bounds = Bounds([lb],[ub])
result = minimize(ytm_obj,y_init,args = args,method = 'trust-constr',bounds = bounds,jac = ytm_obj_d, hess = ytm_obj_dd)
y = result.x[0]
print(f"Newton_CG with 1. and 2. order derivative and ytm constrained to the interval [{lb},{ub}]. ytm: {y}")
