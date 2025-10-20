import numpy as np
from scipy.stats import norm, ncx2
from scipy.optimize import minimize, Bounds
from scipy.special import ndtr, gammainc
from scipy.linalg import sqrtm
from numpy.polynomial.hermite import hermfit, hermval, hermder
import copy

# Conversions between ZCB prices, spot rates forward rates and libor rates
def zcb_prices_from_spot_rates(T,R,type = "continuous"):
    M = len(T)
    p = np.zeros([M])
    if type == "continuous":
        for i in range(0,M):
            if T[i] < 1e-8:
                p[i] = 1
            else:
                p[i] = np.exp(-R[i]*T[i])
    elif type == "simple":
        for i in range(0,M):
            if T[i] < 1e-8:
                p[i] = 1
            else:
                p[i] = 1/(1+R[i]*T[i])
    elif type == "discrete":
        for i in range(0,M):
            if T[i] < 1e-8:
                p[i] = 1
            else:
                p[i] = 1/(1+R[i])**T[i]
    return p

def spot_rates_from_zcb_prices(T,p,type = "continuous"):
    M = len(T)
    R = np.zeros([M])
    if type == "continuous":
        for i in range(0,M):
            if T[i] < 1e-12:
                R[i] = np.nan
            else:
                R[i] = -np.log(p[i])/T[i]
    elif type == "simple":
        for i in range(0,M):
            if T[i] < 1e-12:
                R[i] = np.nan
            else:
                R[i] = (1-p[i])/(T[i]*p[i])
    elif type == "discrete":
        for i in range(0,M):
            if T[i] < 1e-12:
                R[i] = np.nan
            else:
                R[i] = p[i]**(-1/T[i]) - 1
    return R

def forward_rates_from_zcb_prices(T,p,horizon = 1,type = "continuous"):
    # horizon = 0 corresponds to approximated instantaneous forward rates. Note that the first entry of T is assumed to be T[0] = 0
    M = len(T)
    f = np.nan*np.ones([M])
    if type == "continuous":
        if horizon == 0:
            f[0] = (np.log(p[0])-np.log(p[1]))/(T[1]-T[0])
            f[-1] = (np.log(p[-2])-np.log(p[-1]))/(T[-1]-T[-2])
            i = 1
            while i < M - 1.5:
                f[i] = (np.log(p[i-1])-np.log(p[i+1]))/(T[i+1]-T[i-1])
                i += 1
        elif 0 < horizon:
            i = horizon
            while i < M - 0.5:
                f[i] = (np.log(p[i-horizon])-np.log(p[i]))/(T[i]-T[i-horizon])
                i += 1
    elif type == "simple":
        if horizon == 0:
            f[0] = (p[0] - p[1])/(p[1]*(T[1]-T[0]))
            f[-1] = (p[-2] - p[-1])/(p[-1]*(T[-1]-T[-2]))
            i = 1
            while i < M - 1.5:
                f[i] = (p[i-1] - p[i+1])/(p[i+1]*(T[i+1]-T[i-1]))
                i += 1
        elif 0 < horizon:
            i = horizon
            while i < M - 0.5:
                f[i] = (p[i-horizon] - p[i])/(p[i]*(T[i]-T[i-horizon]))
                i += 1
    elif type == "discrete":
        if horizon == 0:
            f[0] = (p[0]/p[1])**(1/(T[1]-T[0])) - 1
            f[-1] = (p[-2]/p[-1])**(1/(T[-1]-T[-2])) - 1
            i = 1
            while i < M - 1.5:
                f[i] = (p[i-1]/p[i+1])**(1/(T[i+1]-T[i-1])) - 1
                i += 1
        elif 0 < horizon:
            i = horizon
            while i < M - 0.5:
                f[i] = (p[i-horizon]/p[i])**(1/(T[i]-T[i-horizon])) - 1
                i += 1
    return f

# Interest Rate Swap
def accrual_factor_from_zcb_prices(t,T_n,T_N,fixed_freq,T,p):
    T_fix = []
    if type(fixed_freq) == str:
        if fixed_freq == "quarterly":
            for i in range(1,int((T_N-T_n)*4) + 1):
                if T_n + i*0.25 > t:
                    T_fix.append(T_n + i*0.25)
        elif fixed_freq == "semiannual":
            for i in range(1,int((T_N-T_n)*2) + 1):
                if T_n + i*0.5 > t:
                    T_fix.append(T_n + i*0.5)
        elif fixed_freq == "annual":
            for i in range(1,int(T_N-T_n) + 1):
                if T_n + i > t:
                    T_fix.append(T_n + i)
    elif type(fixed_freq) == int or type(fixed_freq) == float or type(fixed_freq) == np.int32 or type(fixed_freq) == np.int64 or type(fixed_freq) == np.float64:
        for i in range(1,int((T_N-T_n)/fixed_freq) + 1):
            if T_n + i*fixed_freq > t:
                T_fix.append(T_n + i*fixed_freq)
    p_fix = np.array(for_values_in_list_find_value_return_value(T_fix,T,p))
    T_fix = np.array(T_fix)
    S = (T_fix[0] - T_n)*p_fix[0]
    for i in range(1,len(T_fix)):
        S += (T_fix[i] - T_fix[i-1])*p_fix[i]
    return S

def swap_rate_from_zcb_prices(t,T_n,T_N,fixed_freq,T,p,float_freq = 0,L = 0):
    S = accrual_factor_from_zcb_prices(t,T_n,T_N,fixed_freq,T,p)
    if t <= T_n:
        if T_n < 1e-6:
            p_n = 1
        else:
            Ind_n, output_n = find_value_return_value(T_n,T,p)
            if Ind_n == True:
                p_n = output_n[0][1]
        Ind_N, output_N = find_value_return_value(T_N,T,p)
        if Ind_N == True:
            p_N = output_N[0][1]
        R = (p_n-p_N)/S
    elif t > T_n:
        if float_freq == 0:
            print(f"WARNING! Since t is after inception, 'float_freq' must be given as an argument")
            R = np.nan
        else:
            if type(float_freq) == str:
                if float_freq == "quarterly":
                    float_freq = 0.25
                elif float_freq == "semiannual":
                    float_freq = 0.5
                elif fixed_freq == "annual":
                    float_freq = 1
            i, I_done = 0, False
            while I_done == False and i*float_freq < T_N:
                if i*float_freq >= t:
                    T_n = i*float_freq
                    I_done = True
                i += 1
            if I_done == True:
                [p_n,p_N] = for_values_in_list_find_value_return_value([T_n,T_N],T,p)
                R = (((T_n-t)*L+1)*p_n-p_N)/S
            else:
                print(f"WARNING! Not able to compute the par swap rate")
                R = np.nan
    return R, S

# Fixed rate bond
def macauley_duration(pv,T,C,ytm):
    D = 0
    N = len(T)
    for i in range(0,N):
        D += C[i]*T[i]/(1+ytm)**T[i]
    D = D/pv
    return D

def modified_duration(pv,T,C,ytm):
    D = 0
    N = len(T)
    for i in range(0,N):
        D += C[i]*T[i]/(1+ytm)**T[i]
    D = D/(pv*(1+ytm))
    return D

def convexity(pv,T,C,ytm):
    D = 0
    N = len(T)
    for i in range(0,N):
        D += C[i]*T[i]**2/(1+ytm)**T[i]
    D = D/pv
    return D

def price_fixed_rate_bond_from_ytm(ytm,T,C):
    price = 0
    N = len(T)
    for i in range(0,N):
        price += C[i]/(1+ytm)**T[i]
    return price

def ytm(pv,T,C,ytm_init = 0.05):
    args = (pv, T, C, 1)
    result = minimize(ytm_obj,ytm_init,args = args, options={'disp': False})
    ytm = result.x[0]
    return ytm

def ytm_obj(ytm,pv,T,C,scaling = 1):
    N = len(T)
    pv_new = 0
    for i in range(0,N):
        pv_new += C[i]/(1+ytm[0])**T[i]
    sse = scaling*(pv-pv_new)**2
    return sse

# Vasicek short rate model
def zcb_price_vasicek(r0,a,b,sigma,T):
    if type(T) == int or type(T) == float or type(T) == np.int32 or type(T) == np.float64:
        B = (1/a)*(1-np.exp(-a*T))
        A = (B-T)*(a*b-0.5*sigma**2)/(a**2)-(sigma**2*B**2)/(4*a)
        p = np.exp(A-r0*B)
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        M = len(T)
        p = np.zeros([M])
        for i in range(0,M):
            B = (1/a)*(1-np.exp(-a*T[i]))
            A = (B-T[i])*(a*b-0.5*sigma**2)/(a**2)-(sigma**2*B**2)/(4*a)
            p[i] = np.exp(A-r0*B)
    else:
        print(f"T not of a recognized type")
        p = False
    return p

def spot_rate_vasicek(r0,a,b,sigma,T):
    if type(T) == int or type(T) == float or type(T) == np.int32 or type(T) == np.int64 or type(T) == np.float64:
        B = (1/a)*(1-np.exp(-a*T))
        A = (B-T)*(a*b-0.5*sigma**2)/(a**2)-(sigma**2*B)/(4*a)
        if T < 1e-6:
            r = r0
        elif T >= 1e-6:
            r = (-A+r0*B)/T
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        M = len(T)
        r = np.zeros([M])
        for i in range(0,M):
            B = (1/a)*(1-np.exp(-a*T[i]))
            A = (B-T[i])*(a*b-0.5*sigma**2)/(a**2)-(sigma**2*B)/(4*a)
            if T[i] < 1e-6:
                r[i] = r0
            else:
                r[i] = (-A+r0*B)/T[i]
    else:
        print(f"T not of a recognized type")
        r = False
    return r

def forward_rate_vasicek(r0,a,b,sigma,T):
    if type(T) == int or type(T) == float or type(T) == np.int32 or type(T) == np.int64 or type(T) == np.float64:
        B = (1/a)*(1-np.exp(-a*T))
        B_T = np.exp(-a*T)
        if T < 1e-6:
            f = r0
        elif T >= 1e-6:
            f = (1-B_T)*(a*b-0.5*sigma**2)/(a**2) + (sigma**2*B*B_T)/(2*a) + r0*B_T
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        M = len(T)
        f = np.zeros([M])
        for i in range(0,M):
            B = (1/a)*(1-np.exp(-a*T[i]))
            B_T = np.exp(-a*T[i])
            if T[i] < 1e-6:
                f[i] = r0
            else:
                f[i] = (1-B_T)*(a*b-0.5*sigma**2)/(a**2) + (sigma**2*B*B_T)/(2*a) + r0*B_T
    else:
        print(f"T not of a recognized type")
        f = False
    return f

def mean_vasicek(r0,a,b,sigma,T):
    if type(T) == int or type(T) == float or type(T) == np.int32 or type(T) == np.int64 or type(T) == np.float64:
        if T == np.inf:
            mean = b/a
        else:
            mean = r0*np.exp(-a*T) + b/a*(1-np.exp(-a*T))
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        N = len(T)
        mean = np.zeros(N)
        for n in range(0,N):
            if T[n] == np.inf:
                mean[n] = b/a
            else:
                mean[n] = r0*np.exp(-a*T[n]) + b/a*(1-np.exp(-a*T[n]))
    return mean

def stdev_vasicek(r0,a,b,sigma,T):
    if T == np.inf:
        std = np.sqrt(sigma**2/(2*a))
    else:
        std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T)))
    return std

def ci_vasicek(r0,a,b,sigma,T,size_ci,type_ci = "two_sided"):
    if type(T) == int or type(T) == float or type(T) == np.int32 or type(T) == np.int64 or type(T) == np.float64:
        if type_ci == "lower":
            z = norm.ppf(size_ci,0,1)
            if T < 1e-6:
                lb, ub = r0, r0
            elif T == np.inf:
                mean = b/a
                std = np.sqrt(sigma**2/(2*a))
                lb, ub = mean - z*std, np.inf
            else:
                mean = r0*np.exp(-a*T) + b/a*(1-np.exp(-a*T))
                std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T)))
                lb, ub = mean - z*std, np.inf
        elif type_ci == "upper":
            z = norm.ppf(size_ci,0,1)
            if T < 1e-6:
                lb, ub = r0, r0
            elif T == np.inf:
                mean = b/a
                std = np.sqrt(sigma**2/(2*a))
                lb, ub = -np.inf, mean + z*std
            else:
                mean = r0*np.exp(-a*T) + b/a*(1-np.exp(-a*T))
                std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T)))
                lb, ub = -np.inf, mean + z*std
        elif type_ci == "two_sided":
            z = norm.ppf(size_ci + 0.5*(1-size_ci),0,1)
            if T < 1e-6:
                lb, ub = r0, r0
            elif T == np.inf:
                mean = b/a
                std = np.sqrt(sigma**2/(2*a))
                lb, ub = mean - z*std, mean + z*std
            else:
                mean = r0*np.exp(-a*T) + b/a*(1-np.exp(-a*T))
                std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T)))
                lb, ub = mean - z*std, mean + z*std
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        N = len(T)
        lb, ub = np.zeros([N]), np.zeros([N])
        if type_ci == "lower":
            z = norm.ppf(size_ci,0,1)
            for i in range(0,N):
                if T[i] < 1e-6:
                    lb[i], ub[i] = r0, r0
                else:
                    mean = r0*np.exp(-a*T[i]) + b/a*(1-np.exp(-a*T[i]))
                    std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T[i])))
                    lb[i], ub[i] = mean - z*std, np.inf
        elif type_ci == "upper":
            z = norm.ppf(size_ci,0,1)
            for i in range(0,N):
                if T[i] < 1e-6:
                    lb[i], ub[i] = r0, r0
                else:
                    mean = r0*np.exp(-a*T[i]) + b/a*(1-np.exp(-a*T[i]))
                    std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T[i])))
                    lb[i], ub[i] = -np.inf, mean + z*std
        elif type_ci == "two_sided":
            z = norm.ppf(size_ci + 0.5*(1-size_ci),0,1)
            for i in range(0,N):
                if T[i] < 1e-6:
                    lb[i], ub[i] = r0, r0
                else:
                    mean = r0*np.exp(-a*T[i]) + b/a*(1-np.exp(-a*T[i]))
                    std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T[i])))
                    lb[i], ub[i] = mean - z*std, mean + z*std
    else:
        print(f"T is not of recognized type")
        lb,ub = False, False
    return lb, ub

def simul_vasicek(r0,a,b,sigma,M,T,method = "exact",seed = None):
    if seed is not None:
        np.random.seed(seed)
    delta = T/M
    r = np.zeros([M+1])
    r[0] = r0
    Z = np.random.standard_normal(M)
    if method == "exact":
        for m in range(1,M+1):
            r[m] = r[m-1]*np.exp(-a*delta) + (b/a)*(1-np.exp(-a*delta)) + sigma*np.sqrt((1-np.exp(-2*a*delta))/(2*a))*Z[m-1]
    elif method == "euler" or method == "milstein":
        delta_sqrt = np.sqrt(delta)
        for m in range(1,M+1):
            r[m] = r[m-1] + (b-a*r[m-1])*delta + sigma*delta_sqrt*Z[m-1]
    return r

def euro_option_price_vasicek(K,T1,T2,p_T1,p_T2,a,sigma,type = "call"):
    sigma_p = (sigma/a)*(1-np.exp(-a*(T2-T1)))*np.sqrt((1-np.exp(-2*a*T1))/(2*a))
    d1 = (np.log(p_T2/(p_T1*K)))/sigma_p + 0.5*sigma_p
    d2 = d1 - sigma_p
    if type == "call":
        price = p_T2*ndtr(d1) - p_T1*K*ndtr(d2)
    elif type == "put":
        price = p_T1*K*ndtr(-d2) - p_T2*ndtr(-d1)
    return price

def caplet_prices_vasicek(sigma,strike,a,T,p):
    price_caplet = np.zeros([len(T)])
    if type(strike) == int or type(strike) == float or type(strike) == np.int32 or type(strike) == np.int64 or type(strike) == np.float64:
        for i in range(2,len(T)):
            price_caplet[i] = (1 + (T[i]-T[i-1])*strike)*euro_option_price_vasicek(1/(1 + (T[i]-T[i-1])*strike),T[i-1],T[i],p[i-1],p[i],a,sigma,type = "put")
    elif type(strike) == tuple or type(strike) == list or type(strike) == np.ndarray:
        for i in range(2,len(T)):
            price_caplet[i] = (1 + (T[i]-T[i-1])*strike[i])*euro_option_price_vasicek(1/(1 + (T[i]-T[i-1])*strike[i]),T[i-1],T[i],p[i-1],p[i],a,sigma,type = "put")
    return price_caplet

def fit_vasicek_obj(param,R_star,T,scaling = 1):
    r0, a, b, sigma = param
    M = len(T)
    R_fit = spot_rate_vasicek(r0,a,b,sigma,T)
    sse = 0
    for m in range(0,M):
        sse += scaling*(R_fit[m] - R_star[m])**2
    return sse

def fit_vasicek_no_sigma_obj(param,sigma,R_star,T,scaling = 1):
    r0, a, b = param
    M = len(T)
    R_fit = spot_rate_vasicek(r0,a,b,sigma,T)
    sse = 0
    for m in range(0,M):
        sse += scaling*(R_fit[m] - R_star[m])**2
    return sse




# List operations
def find_value_return_value(val,L1,L2,precision = 10e-8):
    # This function searches for 'val' in 'L1' and returns index 'idx' of 'val' in 'L1' and 'L2[idx]'.
    Ind, output = False, []
    for idx, item in enumerate(L1):
        if abs(val-item) < precision:
            Ind = True
            output.append((idx,L2[idx]))
    return Ind, output

def for_values_in_list_find_value_return_value(L1,L2,L3,precision = 10e-8):
    # For all 'item' in L1, this function searches for 'item' in L2 and returns the value corresponding to same index from 'L3'.
    if type(L1) == int or type(L1) == float or type(L1) == np.float64 or type(L1) == np.int32 or type(L1) == np.int64:
        output = None
        Ind, output_temp = find_value_return_value(L1,L2,L3,precision)
        if Ind == True:
            output = output_temp[0][1]
    elif type(L1) == tuple or type(L1) == list or type(L1) == np.ndarray:
        output = len(L1)*[None]
        for i, item in enumerate(L1):
            Ind, output_temp = find_value_return_value(item,L2,L3,precision)
            if Ind == True:
                output[i] = output_temp[0][1]
    return output

# ZCB curvefitting
def zcb_curve_fit(data_input,interpolation_options = {"method": "linear"},scaling = 1):
    data = copy.deepcopy(data_input)
    data_known = []
    libor_data, fra_data, swap_data = [], [], []
    # Separateing the data and constructing data_known from fixings
    for item in data:
        if item["instrument"] == "libor":
            libor_data.append(item)
            data_known.append({"maturity":item["maturity"],"rate":np.log(1+item["rate"]*item["maturity"])/item["maturity"]})
        elif item["instrument"] == "fra":
            fra_data.append(item)
        elif item["instrument"] == "swap":
            swap_data.append(item)
    # Adding elements to data_knwon based on FRAs
    I_done = False
    while I_done == False:
        for fra in fra_data:
            I_exer, known_exer = value_in_list_of_dict_returns_I_idx(fra["exercise"],data_known,"maturity")
            I_mat, known_mat = value_in_list_of_dict_returns_I_idx(fra["maturity"],data_known,"maturity")
            if I_exer == True and I_mat == False:
                data_known.append({"maturity":fra["maturity"],"rate":(known_exer["rate"]*known_exer["maturity"]+np.log(1+(fra["maturity"]-fra["exercise"])*fra["rate"]))/fra["maturity"]})
                I_done = False
                break
            if I_exer == False and I_mat == True:
                pass
            if I_exer == True and I_mat == True:
                pass
            else:
                I_done = True
    T_known, T_swap, T_knot = [], [], []
    R_known = []
    # Finding T's and corresponding R's where there is some known data
    for known in data_known:
        T_known.append(known["maturity"])
        R_known.append(known["rate"])
    # Finding T_swap - The times where there is a cashflow to at least one of the swaps.
    for swap in swap_data:
        T_knot.append(swap["maturity"])
        if swap["float_freq"] == "quarterly":
            if value_in_list_returns_I_idx(0.25,T_known)[0] == False and value_in_list_returns_I_idx(0.25,T_swap)[0] == False:
                T_swap.append(0.25)
        elif swap["float_freq"] == "semiannual":
            if value_in_list_returns_I_idx(0.5,T_known)[0] == False and value_in_list_returns_I_idx(0.5,T_swap)[0] == False:
                T_swap.append(0.5)
        elif swap["float_freq"] == "annual":
            if value_in_list_returns_I_idx(1,T_known)[0] == False and value_in_list_returns_I_idx(1,T_swap)[0] == False:
                T_swap.append(1)
        if swap["fixed_freq"] == "quarterly":
            for i in range(1,4*swap["maturity"]):
                if value_in_list_returns_I_idx(i*0.25,T_known)[0] == False and value_in_list_returns_I_idx(i*0.25,T_knot)[0] == False and value_in_list_returns_I_idx(i*0.25,T_swap)[0] == False:
                    T_swap.append(i*0.25)
        elif swap["fixed_freq"] == "semiannual":
            for i in range(1,2*swap["maturity"]):
                if value_in_list_returns_I_idx(i*0.5,T_known)[0] == False and value_in_list_returns_I_idx(i*0.5,T_knot)[0] == False and value_in_list_returns_I_idx(i*0.5,T_swap)[0] == False:
                    T_swap.append(i*0.5)
        elif swap["fixed_freq"] == "annual":
            for i in range(1,swap["maturity"]):
                if value_in_list_returns_I_idx(i,T_known)[0] == False and value_in_list_returns_I_idx(i*1,T_knot)[0] == False and value_in_list_returns_I_idx(i,T_swap)[0] == False:
                    T_swap.append(i)
    # Finding T_fra and T_endo
    T_endo, T_fra = [], []
    fra_data.reverse()
    for fra in fra_data:
        if value_in_list_returns_I_idx(fra["maturity"],T_known)[0] == False:
            I_fra_mat, idx_fra_mat = value_in_list_returns_I_idx(fra["maturity"],T_fra)
            I_endo_mat, idx_endo_mat = value_in_list_returns_I_idx(fra["maturity"],T_endo)
            if I_fra_mat is False and I_endo_mat is False:
                T_fra.append(fra["maturity"])
            elif I_fra_mat is True and I_endo_mat is False:
                pass
            elif I_fra_mat is False and I_endo_mat is True:
                pass
            elif I_fra_mat is True and I_endo_mat is True:
                T_fra.pop(idx_fra_mat)
        if value_in_list_returns_I_idx(fra["exercise"],T_known)[0] == False:
            I_fra_exer, idx_fra_exer = value_in_list_returns_I_idx(fra["exercise"],T_fra)
            I_endo_exer, idx_endo_exer = value_in_list_returns_I_idx(fra["exercise"],T_endo)
            if I_fra_exer is False and I_endo_exer is False:
                T_endo.append(fra["exercise"])
            elif I_fra_exer is True and I_endo_exer is False:
                T_fra.pop(idx_fra_exer)
                T_endo.append(fra["exercise"])
            elif I_fra_exer is False and I_endo_exer is True:
                pass
            elif I_fra_exer is True and I_endo_exer is True:
                T_fra.pop(idx_fra_exer)
    fra_data.reverse()
    # Fitting the swap portion of the curve
    T_swap_fit = T_known + T_swap + T_knot
    T_swap_fit.sort(), T_fra.sort(), T_endo.sort()
    R_knot_init = [None]*len(swap_data)
    for i, swap in enumerate(swap_data):
        indices = []
        if swap["fixed_freq"] == "quarterly":
            for j in range(1,4*swap["maturity"]+1):
                indices.append(value_in_list_returns_I_idx(j*0.25,T_swap_fit)[1])
        elif swap["fixed_freq"] == "semiannual":
            for j in range(1,2*swap["maturity"]+1):
                indices.append(value_in_list_returns_I_idx(j*0.5,T_swap_fit)[1])
        elif swap["fixed_freq"] == "annual":
            for j in range(1,swap["maturity"]+1):
                indices.append(value_in_list_returns_I_idx(j,T_swap_fit)[1])
        swap["indices"] = indices
        R_knot_init[i] = swap["rate"]
        i += 1
    args = (T_known,T_knot,T_swap_fit,R_known,swap_data,interpolation_options,1)
    result = minimize(zcb_curve_swap_fit_obj,R_knot_init,method = 'nelder-mead',args = args,options={'xatol': 1e-12,'disp': False})
    # print(f"Output from the swap portion fit")
    # print(result)
    T_swap_curve, R_swap_curve = T_known + T_knot, R_known + list(result.x)
    T_fra_fit = T_swap_curve + T_fra + T_endo
    T_fra_fit.sort()
    R_fra_fit, R_fra_fit_deriv = interpolate(T_fra_fit,T_swap_curve,R_swap_curve,interpolation_options)
    R_fra_init = [None]*len(T_fra)
    for i in range(0,len(T_fra)):
        R_fra_init[i] = R_fra_fit[value_in_list_returns_I_idx(T_fra[i],T_fra_fit)[1]]
    args = (T_fra,T_known,T_endo,T_fra_fit,R_fra_fit,fra_data,interpolation_options,scaling)
    result = minimize(zcb_curve_fra_fit_obj,R_fra_init,method = 'nelder-mead',args = args,options={'xatol': 1e-12,'disp': False})
    # print(f"Output from the FRA portion fit")
    # print(result)
    R_fra = list(result.x)
    R_endo = R_T_endo_from_R_T_fra(R_fra,T_fra,T_endo,fra_data)
    for i in range(0,len(T_fra_fit)):
        I_fra, idx_fra = value_in_list_returns_I_idx(T_fra_fit[i],T_fra)
        if I_fra is True:
            R_fra_fit[i] = R_fra[idx_fra]
        elif I_fra is False:
            I_endo, idx_endo = value_in_list_returns_I_idx(T_fra_fit[i],T_endo)
            if I_endo is True:
                R_fra_fit[i] = R_endo[idx_endo]
    return np.array(T_fra_fit), np.array(R_fra_fit)

def zcb_curve_interpolate(T_inter,T,R,interpolation_options = {"method":"linear"}):
    N = len(T_inter)
    p_inter = np.ones([N])
    R_inter = np.zeros([N])
    f_inter = np.zeros([N])
    R_inter, R_inter_deriv = interpolate(T_inter,T,R,interpolation_options = interpolation_options)
    for i in range(0,N):
        f_inter[i] = R_inter[i] + R_inter_deriv[i]*T_inter[i]
        p_inter[i] = np.exp(-R_inter[i]*T_inter[i])
    return p_inter, R_inter, f_inter, T_inter

def spot_rate_bump(T_bump,size_bump,T,R_input,p_input):
    R, p = R_input.copy(), p_input.copy()
    if type(T_bump) == int or type(T_bump) == float or type(T_bump) == np.float64 or type(T_bump) == np.int32 or type(T_bump) == np.int64:
        I_bump, idx_bump = value_in_list_returns_I_idx(T_bump,T)
        R[idx_bump] = R[idx_bump] + size_bump
        p[idx_bump] = np.exp(-R[idx_bump]*T_bump)
    elif type(T_bump) == tuple or type(T_bump) == list or type(T_bump) == np.ndarray:
        if type(size_bump) == int or type(size_bump) == float or type(size_bump) == np.float64:
            for i in range(0,len(T_bump)):
                I_bump, idx_bump = value_in_list_returns_I_idx(T_bump[i],T)
                R[idx_bump] = R[idx_bump] + size_bump
                p[idx_bump] = np.exp(-R[idx_bump]*T_bump[i])
        elif type(size_bump) == tuple or type(size_bump) == list or type(size_bump) == np.ndarray:
            for i in range(0,len(T_bump)):
                I_bump, idx_bump = value_in_list_returns_I_idx(T_bump[i],T)
                R[idx_bump] = R[idx_bump] + size_bump[i]
                p[idx_bump] = np.exp(-R[idx_bump]*T_bump[i])
    return R, p

def market_rate_bump(idx_bump,size_bump,T_inter,data,interpolation_options = {"method": "linear"}):
    data_bump = copy.deepcopy(data)
    if type(idx_bump) == int or type(idx_bump) == float or type(idx_bump) == np.float64 or type(idx_bump) == np.int32 or type(idx_bump) == np.int64:
        data_bump[idx_bump]["rate"] += size_bump
        T_fit_bump, R_fit_bump = zcb_curve_fit(data_bump,interpolation_options = interpolation_options)
        p_inter_bump, R_inter_bump, f_inter_bump, T_inter_bump = zcb_curve_interpolate(T_inter,T_fit_bump,R_fit_bump,interpolation_options = interpolation_options)
    elif type(idx_bump) == tuple or type(idx_bump) == list or type(idx_bump) == np.ndarray:
        if type(size_bump) == int or type(size_bump) == float or type(size_bump) == np.float64 or type(size_bump) == np.int32 or type(size_bump) == np.int64:
            for i in range(0,len(idx_bump)):
                data_bump[idx_bump[i]]["rate"] += size_bump
            T_fit_bump, R_fit_bump = zcb_curve_fit(data_bump,interpolation_options = interpolation_options)
            p_inter_bump, R_inter_bump, f_inter_bump, T_inter_bump = zcb_curve_interpolate(T_inter,T_fit_bump,R_fit_bump,interpolation_options = interpolation_options)
        elif type(size_bump) == tuple or type(size_bump) == list or type(size_bump) == np.ndarray:
            for i in range(0,len(idx_bump)):
                data_bump[idx_bump[i]]["rate"] += size_bump[i]
            T_fit_bump, R_fit_bump = zcb_curve_fit(data_bump,interpolation_options = interpolation_options)
            p_inter_bump, R_inter_bump, f_inter_bump, T_inter_bump = zcb_curve_interpolate(T_inter,T_fit_bump,R_fit_bump,interpolation_options = interpolation_options)
    return p_inter_bump, R_inter_bump, f_inter_bump, T_inter_bump, data_bump

def extrapolate(x_extra,x,y,extrapolation_options = {"method":"linear"}):
    # Extrapoltion of value corresponding to a choice of x_extra
    if extrapolation_options["method"] == "linear":
        if x_extra < x[0]:
            a = (y[1]-y[0])/(x[1]-x[0])
            b = y[0]-a*x[0]
            y_extra = a*x_extra + b
            y_extra_deriv = a
        elif x[-1] < x_extra:
            a = (y[-1]-y[-2])/(x[-1]-x[-2])
            b = y[-1]-a*x[-1]
            y_extra = a*x_extra + b
            y_extra_deriv = a
        else:
            print(f"WARNING! x_extra is inside the dataset")
    elif extrapolation_options["method"] == "hermite":
        if x_extra < x[0]:
            coefs = hermfit(x[0:extrapolation_options["degree"]+1],y[0:extrapolation_options["degree"]+1],extrapolation_options["degree"])
            y_extra, y_extra_deriv = hermval(x_extra,coefs), hermval(x_extra,hermder(coefs))
        elif x[-1] < x_extra:
            coefs = hermfit(x[-extrapolation_options["degree"]-1:],y[-extrapolation_options["degree"]-1:],extrapolation_options["degree"])
            y_extra, y_extra_deriv = hermval(x_extra,coefs), hermval(x_extra,hermder(coefs))
        else:
            print(f"WARNING! x_extra is inside the dataset")
    elif extrapolation_options["method"] == "nelson_siegel":
        if x_extra < x[0]:
            x1, x2 = x[1]-x[0], x[2]-x[0]
            coefs = nelson_siegel_coef(x1,x2,y[0],y[1],y[2])
            y_extra, y_extra_deriv = coefs[0]+coefs[1]*np.exp(-coefs[2]*(x_extra-x[0])), -coefs[1]*coefs[2]*np.exp(-coefs[2]*(x_extra-x[0]))
        elif x[-1] < x_extra:
            x1, x2 = x[-2]-x[-3], x[-1]-x[-3]
            coefs = nelson_siegel_coef(x1,x2,y[-3],y[-2],y[-1])
            y_extra, y_extra_deriv = coefs[0]+coefs[1]*np.exp(-coefs[2]*(x_extra-x[-3])), -coefs[1]*coefs[2]*np.exp(-coefs[2]*(x_extra-x[-3]))
        else:
            print(f"WARNING! x_extra is inside the dataset")
    return y_extra, y_extra_deriv

def interpolate(x_inter,x,y,interpolation_options = {"method":"linear", "transition": None}):
    N, M = len(x_inter), len(x)
    y_inter, y_inter_deriv = np.nan*np.ones([N]), np.nan*np.ones([N])
    if interpolation_options["method"] == "linear":
        coefs = np.nan*np.ones([M,2])
        for m in range(0,M-1):
            coefs[m,1] = (y[m+1]-y[m])/(x[m+1]-x[m])
            coefs[m,0] = y[m]-coefs[m,1]*x[m]
        coefs[M-1,1] = (y[M-1] - y[M-2])/(x[M-1]-x[M-2])
        coefs[M-1,0] = y[M-1] - coefs[M-1,1]*x[M-1]
        for n in range(0,N):
            if x_inter[n] < x[0] or x_inter[n] > x[M-1]:
                extrapolate(x_inter[n],x,y,extrapolation_options = interpolation_options)
            else:
                I_known, idx = value_in_list_returns_I_idx(x_inter[n],x)
                if I_known is True:
                    y_inter[n] = y[idx]
                    if idx == 0:
                        y_inter_deriv[n] = coefs[0,1]
                    elif idx == M-1:
                        y_inter_deriv[n] = coefs[M-1,1]
                    else:
                        y_inter_deriv[n] = 0.5*coefs[idx-1,1] + 0.5*coefs[idx,1]
                        y_inter_deriv[n] = coefs[idx,1]
                else:
                    idx_before, idx_after = idx_before_after_in_iterable(x_inter[n],x)
                    if interpolation_options["transition"] == "smooth":
                        w_before = (x[idx_after]-x_inter[n])/(x[idx_after]-x[idx_before])
                        y_before, y_after = coefs[idx_before,0] + coefs[idx_before,1]*x_inter[n], coefs[idx_after,0] + coefs[idx_after,1]*x_inter[n]

                        y_inter[n] = w_before*y_before + (1-w_before)*y_after
                        y_inter_deriv[n] = w_before*coefs[idx_before,1] + (1-w_before)*coefs[idx_after,1]
                    else:
                        y_inter[n] = coefs[idx_before,0] + coefs[idx_before,1]*x_inter[n]
                        y_inter_deriv[n] = coefs[idx_before,1]
    elif interpolation_options["method"] == "hermite":
        coefs = np.nan*np.ones([M,interpolation_options["degree"]+1])
        degrees = np.ones(M, dtype = "int")
        for m in range(0, M-1):
            left = min(int(interpolation_options["degree"]/2),m)
            right = min(M-1-m,int((interpolation_options["degree"]+1)/2))
            degrees[m] = left + right
            coefs[m,0:left+right+1] = hermfit(x[m-left:m+right+1],y[m-left:m+right+1],degrees[m])
        coefs[M-1], degrees[M-1] = coefs[M-2], degrees[M-2]
        for n in range(0,N):
            if x_inter[n] < x[0] or x_inter[n] > x[M-1]:
                y_inter[n], y_inter_deriv[n] = extrapolate(x_inter[n],x,y,extrapolation_options = interpolation_options)
            else:
                I_known, idx = value_in_list_returns_I_idx(x_inter[n],x)
                if I_known is True:
                    y_inter[n] = y[idx]
                    y_inter_deriv[n] = hermval(x_inter[n],hermder(coefs[idx,0:degrees[idx]+1]))
                else:
                    idx_before, idx_after = idx_before_after_in_iterable(x_inter[n],x)
                    if interpolation_options["transition"] == "smooth":
                        w_before = (x[idx_after]-x_inter[n])/(x[idx_after]-x[idx_before])
                        y_before = hermval(x_inter[n],coefs[idx_before,0:degrees[idx_before]+1])
                        y_after = hermval(x_inter[n],coefs[idx_after,0:degrees[idx_after]+1])
                        y_inter[n] = w_before*y_before + (1-w_before)*y_after
                        y_inter_deriv[n] = w_before*hermval(x_inter[n],hermder(coefs[idx_before,0:degrees[idx_before]+1])) + (1-w_before)*hermval(x_inter[n],hermder(coefs[idx_after,0:degrees[idx_after]+1]))
                    else:
                        y_inter[n] = hermval(x_inter[n],coefs[idx_before,0:degrees[idx_before]+1])
                        y_inter_deriv[n] = hermval(x_inter[n],hermder(coefs[idx_before,0:degrees[idx_before]+1]))
    elif interpolation_options["method"] == "nelson_siegel":
        coefs, type_fit = np.nan*np.ones([M,3]), M*[None]
        for m in range(1,M-1):
            if (y[m] > y[m-1] and y[m+1] > y[m]) or (y[m] < y[m-1] and y[m+1] < y[m]):
                coefs[m,0:3] = nelson_siegel_coef(x[m]-x[m-1],x[m+1]-x[m-1],y[m-1],y[m],y[m+1])
                type_fit[m] = "nelson_siegel"
            else:
                coefs[m,0:3] = hermfit(x[m-1:m+2],y[m-1:m+2],2)
                type_fit[m] = "hermite"
        for n in range(0,N):
            if x_inter[n] < x[0] or x_inter[n] > x[M-1]:
                y_inter[n], y_inter_deriv[n] = extrapolate(x_inter[n],x,y,extrapolation_options = interpolation_options)
            else:
                I_known, idx = value_in_list_returns_I_idx(x_inter[n],x)
                if I_known is True:
                    y_inter[n] = y[idx]
                    if idx == 0:
                        if type_fit[idx+1] == "nelson_siegel":
                            x_ns = x_inter[n] - x[idx]
                            y_inter_deriv[n] = -coefs[idx+1,1]*coefs[idx+1,2]*np.exp(-coefs[idx+1,2]*x_ns)
                        elif type_fit[idx+1] == "hermite":
                            y_inter_deriv[n] = hermval(x_inter[n],hermder(coefs[idx+1,0:3]))
                    elif idx == M-1:
                        if type_fit[idx-1] == "nelson_siegel":
                            x_ns = x_inter[n] - x[idx-2]
                            y_inter_deriv[n] = -coefs[idx-1,1]*coefs[idx-1,2]*np.exp(-coefs[idx-1,2]*x_ns)
                        elif type_fit[idx-1] == "hermite":
                            y_inter_deriv[n] = hermval(x_inter[n],hermder(coefs[idx-1,0:3]))
                    else:
                        if type_fit[idx] == "nelson_siegel":
                            x_ns = x_inter[n] - x[idx-1]
                            y_inter_deriv[n] = -coefs[idx,1]*coefs[idx,2]*np.exp(-coefs[idx,2]*x_ns)
                        elif type_fit[idx] == "hermite":
                            y_inter_deriv[n] = hermval(x_inter[n],hermder(coefs[idx,0:3]))
                else:
                    idx_before, idx_after = idx_before_after_in_iterable(x_inter[n],x)
                    if idx_before == 0:
                        if type_fit[idx_before+1] == "nelson_siegel":
                            x_ns = x_inter[n] - x[idx_before]
                            y_inter[n], y_inter_deriv[n] = coefs[idx_after,0] + coefs[idx_after,1]*np.exp(-coefs[idx_after,2]*x_ns), -coefs[idx_after,1]*coefs[idx_after,2]*np.exp(-coefs[idx_after,2]*x_ns)
                            y_inter_deriv[n] = -coefs[idx_after,1]*coefs[idx_after,2]*np.exp(-coefs[idx_after,2]*x_ns)
                        elif type_fit[idx_before+1] == "hermite":
                            y_inter[n], y_inter_deriv[n] = hermval(x_inter[n],coefs[idx_after,0:3]), hermval(x_inter[n],hermder(coefs[idx_after,0:3]))
                    elif idx_after == M-1:
                        if type_fit[idx_after-1] == "nelson_siegel":
                            x_ns = x_inter[n] - x[idx_after-2]
                            y_inter[n], y_inter_deriv[n] = coefs[idx_after-1,0] + coefs[idx_after-1,1]*np.exp(-coefs[idx_after-1,2]*x_ns), -coefs[idx_after-1,1]*coefs[idx_after-1,2]*np.exp(-coefs[idx_after-1,2]*x_ns)
                        elif type_fit[idx_after-1] == "hermite":
                            y_inter[n], y_inter_deriv[n] = hermval(x_inter[n],coefs[idx_after-1,0:3]), hermval(x_inter[n],hermder(coefs[idx_after-1,0:3]))
                    else:
                        if interpolation_options["transition"] == "smooth":
                            w_left = (x[idx_after]-x_inter[n])/(x[idx_after]-x[idx_before])
                            if type_fit[idx_before] == "nelson_siegel":
                                x_left = x_inter[n] - x[idx_before-1]
                                y_left, y_left_deriv = coefs[idx_before,0] + coefs[idx_before,1]*np.exp(-coefs[idx_before,2]*x_left), -coefs[idx_before,1]*coefs[idx_before,2]*np.exp(-coefs[idx_before,2]*x_left)
                            elif type_fit[idx_before] == "hermite":
                                y_left, y_left_deriv = hermval(x_inter[n],coefs[idx_before,0:3]), hermval(x_inter[n],hermder(coefs[idx_before,0:3]))
                            if type_fit[idx_after] == "nelson_siegel":
                                x_right = x_inter[n] - x[idx_after-1]
                                y_right, y_right_deriv = coefs[idx_after,0] + coefs[idx_after,1]*np.exp(-coefs[idx_after,2]*x_right), -coefs[idx_after,1]*coefs[idx_after,2]*np.exp(-coefs[idx_after,2]*x_right)
                            elif type_fit[idx_after] == "hermite":
                                y_right, y_right_deriv = hermval(x_inter[n],coefs[idx_after,0:3]), hermval(x_inter[n],hermder(coefs[idx_after,0:3]))
                            y_inter[n], y_inter_deriv[n] = w_left*y_left + (1-w_left)*y_right, w_left*y_left_deriv + (1-w_left)*y_right_deriv
                        else:
                            if type_fit[idx_before] == "nelson_siegel":
                                x_ns = x_inter[n] - x[idx_before-1]
                                y_inter[n], y_inter_deriv[n] = coefs[idx_before,0] + coefs[idx_before,1]*np.exp(-coefs[idx_before,2]*x_ns), -coefs[idx_before,1]*coefs[idx_before,2]*np.exp(-coefs[idx_before,2]*x_ns)
                            elif type_fit[idx_before] == "hermite":
                                y_inter[n], y_inter_deriv[n] = hermval(x_inter[n],coefs[idx_before,0:3]), hermval(x_inter[n],hermder(coefs[idx_before,0:3]))
    return y_inter, y_inter_deriv

def nelson_siegel_coef(x1,x2,y0,y1,y2):
    alpha = (y0-y2)/(y0-y1)
    b_hat = 2*(alpha*x1-x2)/(alpha*x1**2-x2**2)
    result = minimize(nelson_siegel_coef_obj,b_hat,method = "nelder-mead",args = (alpha,x1,x2),options={'xatol': 1e-12,"disp": False})
    if type(result.x) == np.ndarray:
        b = result.x[0]
    elif type(result.x) == int or type(result.x) == int or type(result.x) == np.int32 or type(result.x) == np.int64 or type(result.x) == np.float64:
        b = result.x
    a = (y0-y1)/(1-np.exp(-b*x1))
    f_inf = y0 - a
    return f_inf, a, b

def nelson_siegel_coef_obj(b,alpha,x1,x2):
    se = (alpha-(1-np.exp(-b*x2))/(1-np.exp(-b*x1)))**2
    return se

def swap_indices(data,T):
    for item in data:
        if item["instrument"] == "swap":
            indices = []
            if item["fixed_freq"] == "quarterly":
                for i in range(1,4*item["maturity"]+1):
                    indices.append(value_in_list_returns_I_idx(i*0.25,T)[1])
            elif item["fixed_freq"] == "semiannual":
                for i in range(1,2*item["maturity"]+1):
                    indices.append(value_in_list_returns_I_idx(i*0.5,T)[1])
            elif item["fixed_freq"] == "annual":
                for i in range(1,item["maturity"]+1):
                    indices.append(value_in_list_returns_I_idx(i,T)[1])
            item["indices"] = indices
    return data

def R_T_endo_from_R_T_fra(R_fra,T_fra,T_endo,fra_data):
    R_fra.reverse(), T_fra.reverse()
    R_endo = [None]*len(T_endo)
    for i in range(0,len(T_fra)):
        I_fra, dict_fra = value_in_list_of_dict_returns_I_idx(T_fra[i],fra_data,"maturity")
        if I_fra is True:
            idx_endo = value_in_list_returns_I_idx(dict_fra["exercise"],T_endo)[1]
            R_endo[idx_endo] = (R_fra[i]*T_fra[i] - np.log(1+(dict_fra["maturity"]-dict_fra["exercise"])*dict_fra["rate"]))/T_endo[idx_endo]
    R_fra.reverse(), T_fra.reverse()
    R_endo.reverse(), T_endo.reverse()
    for i in range(0,len(T_endo)):
        I_fra, dict_fra = value_in_list_of_dict_returns_I_idx(T_endo[i],fra_data,"maturity")
        if I_fra is True:
            idx_endo = value_in_list_returns_I_idx(dict_fra["exercise"],T_endo)[1]
            R_endo[idx_endo] = (R_endo[i]*T_endo[i] - np.log(1+(dict_fra["maturity"]-dict_fra["exercise"])*dict_fra["rate"]))/T_endo[idx_endo]
    R_endo.reverse(), T_endo.reverse()
    return R_endo

def zcb_curve_fra_fit_obj(R_fra,T_fra,T_known,T_endo,T_all,R_all,fra_data,interpolation_options,scaling = 1):
    sse = 0
    R_fra = list(R_fra)
    R_endo = R_T_endo_from_R_T_fra(R_fra,T_fra,T_endo,fra_data)
    for i in range(0,len(T_fra)):
        if T_fra[i] > min(T_known):
            sse += (R_all[value_in_list_returns_I_idx(T_fra[i],T_all)[1]] - R_fra[i])**2
    for i in range(0,len(T_endo)):
        if T_endo[i] > min(T_known):
            sse += (R_all[value_in_list_returns_I_idx(T_endo[i],T_all)[1]] - R_endo[i])**2
    sse *= scaling
    return sse

def zcb_curve_swap_fit_obj(R_knot,T_known,T_knot,T_all,R_known,swap_data,interpolation_options,scaling = 1):
    sse = 0
    R_knot = list(R_knot)
    R_all, R_deriv = interpolate(T_all,T_known + T_knot,R_known + R_knot,interpolation_options)
    p = zcb_prices_from_spot_rates(T_all,R_all)
    for n, swap in enumerate(swap_data):
        if swap["fixed_freq"] == "quarterly":
            alpha = 0.25
        if swap["fixed_freq"] == "semiannual":
            alpha = 0.5
        if swap["fixed_freq"] == "annual":
            alpha = 1
        S_swap = 0
        for idx in swap["indices"]:
            S_swap += alpha*p[idx]
        R_swap = (1 - p[swap["indices"][-1]])/S_swap
        sse += (R_swap - swap["rate"])**2
    sse *= scaling
    return sse

def value_in_list_returns_I_idx(value,list,precision = 1e-12):
    output = False, None
    for i, item in enumerate(list):
        if abs(value-item) < precision:
            output = True, i
            break
    return output

def idx_before_after_in_iterable(value,list):
    idx_before, idx_after = None, None
    if value < list[0]:
        idx_before, idx_after = None, 0
    elif list[-1] < value:
        idx_before, idx_after = len(list) - 1, None
    else:
        for i in range(0,len(list)):
            if list[i] < value:
                idx_before = i
            elif list[i] > value:
                idx_after = i
                break
    return idx_before, idx_after

def value_in_list_of_dict_returns_I_idx(value,L,name,precision = 1e-12):
    output = False, None
    for item in L:
        if abs(value-item[name]) < precision:
            output = True, item
            break
    return output
