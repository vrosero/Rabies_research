# -*- coding: utf-8 -*-
"""
Created on Sun May  7 16:43:11 2023

@author: elvee
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import datetime


# user-defined functions
from model import model
from data_import import data_import
from lhs_param import lhs_param
from param_est import param_est
from error_score import error_score
from param_min import param_min
from bootstrap import sol_bootstrap
from lhsprcc import lhsprcc

# For real-time file saving.
ct = datetime.datetime.now()
str_date_time = ct.strftime("%d%m%Y_%H%M%S")

np.random.seed(2)

#1. data import

# data load
rabiesdatat, ydata, fixvar, y0, params = data_import()

dogrec, dog_loss, dog_vax, ddeath, imp, dogrep, humrec, humloss, hdeath, hrabdeath = fixvar
b1, h1, h2, prevac, postvac = params
Sd0, Ed0, Vd0, Red0, Sh0, Eh0, Vh0 = y0

# Model Steps
start_time = rabiesdatat[0]   # first rabies case
end_time   = rabiesdatat[len(rabiesdatat)-1]  
dt         = .01
intervals  = int(end_time/dt)
mt         = np.linspace(start_time, end_time, intervals)
t          = np.linspace(start_time, end_time, intervals)

# Model Index to Compare to Data
findindex = lambda x:np.where(mt>=x)[0][0]
m = map(findindex, rabiesdatat)
l = list(m)

#2. LHS
no_params = len(params)
l_bounds = np.array([10e-20, 10e-20, 10e-20, 10e-20, 10e-20])
u_bounds = np.array([1, 1, 1, 1, 1])
no_sets = 10

lhs_samples = lhs_param(no_params, no_sets, l_bounds, u_bounds)

#3. Parameter Estimation
Nw = no_sets; w = 0; param = []; paramOpt = []; bestscore_mat = np.zeros((Nw,5))

for n in range(len(lhs_samples)):
    print('iter: ' + str(n))
    paramInit = lhs_samples[n,:]

    paramNew = param_est(no_params, paramInit, y0, t, fixvar, ydata,l)
    
    bestscore = error_score(y0, t, paramNew, fixvar, l, ydata)
    param = np.concatenate((param, paramNew))
    bestscore_mat[n,:] = bestscore
    
    w = w + 1

param_mat = param.reshape(w,no_params)

# Saving the parameter estimates and error from lhs-generated initial parameters
# np.savetxt("dhseiv_paramest_error.csv", param_mat, bestscore_mat, delimiter=",")

# Finding the best set of parameter estimates
lhs_param_sse = np.hstack((lhs_samples, param_mat, bestscore_mat))
lhs_param_sse_table = pd.DataFrame(data=lhs_param_sse, columns = ["init_b1", "init_h1", 
            "init_h2", "init_prevac", "init_postvac", "param_b1", 
            "param_h1", "param_h2", "param_prevac", "param_postvac",
            "RSE", "MSE", "MAE", "SSE", "MAPE"])

# To record and save csv file for the LHS estimates and errors
lhs_param_sse_table.to_csv(str_date_time + '_lhs_estimates.csv')

# Selecting the set of LHS estimated values with least SSE
parambest_set = param_min(lhs_param_sse_table)
parambest = parambest_set[0,0:5]

# Re-estimate using parambest
parambest_New = param_est(no_params, parambest, y0, t, fixvar, ydata, l)
bestscore_New = error_score(y0, t, parambest_New, fixvar, l, ydata)

# Selecting n rows from LHS estimates
# To record selected number of lhs estimates based on least SSE
# lhs_sort1 = lhs_param_sse_table.sort_values(by=["SSE"], ascending = [True])
# lhs_sort1_parameters = lhs_sort1[["param_b1","param_h1", "param_h2", "param_prevac", "param_postvac"]]   #select the parameter estimates
# lhs_arr1_param = lhs_sort1_parameters.to_numpy()

# To select the first n rows based on least SSE
#lhs_least_error = lhs_arr1_param[np.r_[0:10],:] 
# lhs_selected_arr = np.vstack(parambest_set)
# lhs_selected_arr2 = pd.DataFrame(data = lhs_selected_arr)   


# lhs_selected_set = param_min(lhs_param_sse_table)
# #lhs_selected = lhs_selected_set[0:10,:]
# lhs_selected_arr = np.vstack(lhs_selected_set)
# print(lhs_selected_arr)
# lhs_selected_arr2 = pd.DataFrame(data = lhs_selected_arr)

#4. PRCC
# LHS PRCC is in jupyter.
#prcc_val = lhsprcc()

#5. Bootstrap
#Generating solution for bootstrap
# realizations = 10
# p_init_true = parambest

# boot_param_est_mat, boot_sol_mat, boot_sol_l_mat, new_cum_data_mat, plims_sample = sol_bootstrap(realizations, p_init_true, y0, t, fixvar, ydata, l, no_params)

# # To record bootstrap estimates
# boot_param = np.hstack((boot_param_est_mat, boot_sol_l_mat))
# boot_param_table = pd.DataFrame(data=boot_param)
 
# boot_param_table.to_csv(str_date_time + '_bootstrap_estimates.csv')


#6. Plot

# Plotting actual data vs model
sol = odeint(model, y0, t, (parambest, fixvar))
plt.plot(rabiesdatat, ydata, ".", alpha=0.5, label="Actual Data")
# # plt.plot(t, sol[:,0], 'b', alpha=0.5, lw=3, label='Susceptible Dogs')
# # plt.plot(t, sol[:,1], 'y', alpha=0.5, lw=3, label='Suspected Rabid Dogs')
# # plt.plot(t, sol[:,2], 'g', alpha=0.5, lw=3, label='Vaccinated Dogs')
plt.plot(t, sol[:,3], 'r', alpha=0.5, lw=3, label='Model')
# # plt.plot(t, sol[:,4], 'x', alpha=0.5, lw=3, label='Susceptible Humans')
# # plt.plot(t, sol[:,5], 'o', alpha=0.5, lw=3, label='Exposed Humans')
# # plt.plot(t, sol[:,6], '*', alpha=0.5, lw=3, label='Vaccinated Humans')
plt.xlabel('Months', Fontsize = 12)
plt.ylabel('Reported Cases', Fontsize = 12)
plt.yscale('log')
plt.xticks(Fontsize = 12)
plt.yticks(Fontsize = 12)
plt.legend(fontsize = 10, loc='center right')
plt.title('Reported Dog Rabies Data vs Best Model Estimate', fontsize = 14)
plt.savefig(str_date_time + '_rabdash_plot1_' + str(Nw) + '.tiff', dpi=300, bbox_inches='tight')
plt.close()


# Plot all model estimates
for i in range(len(lhs_samples)):
    solh = odeint(model, y0, t, (param_mat[i,:], fixvar))
    # Sd, Ed, Vd, Red, Sh, Eh, Vh, t = odeint(model, y0, t, (param_mat[i,:], fixvar))
    Red = solh[:,3]
    series = pd.Series(Red)
    cumulative = series.cumsum()
    
    if (i == 0):
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.plot(rabiesdatat, ydata, ".", label="Actual Data")
        plt.plot(t, Red, 'r', alpha=0.5, lw=3, label='Model Estimate')
    else:
        plt.plot(t, Red, 'r', alpha=0.5, lw=3)
  
    plt.xlabel('Time')
    plt.ylabel('Reported Dog Rabies')
    plt.yscale('log')
    plt.xticks(Fontsize = 12)
    plt.yticks(Fontsize = 12)
    plt.legend(fontsize = 10)
    #plt.ylim(bottom = 0)
    plt.title('Dog Rabies Data vs' + str(Nw) + 'Model Estimates', fontsize = 14)
    

plt.savefig(str_date_time + '_rabdash_plot2_' + str(Nw) + '.tiff', dpi=300, bbox_inches='tight')
plt.close()


# Plot selected number of models
for i in range(len(parambest_set)):
    solh = odeint(model, y0, t, (parambest_set[i,:], fixvar))
    Red = solh[:,3]
    series = pd.Series(Red)
    cumulative = series.cumsum()
    
    if (i == 0):
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.plot(rabiesdatat, ydata, ".", label="Actual Data")
        plt.plot(t, Red, 'r', alpha=0.5, lw=3, label='Model Estimate')
    else:
        plt.plot(t, Red, 'r', alpha=0.5, lw=3)
  
    plt.xlabel('Time')
    plt.ylabel('Reported Dog Rabies')
    plt.yscale('log')
    plt.xticks(Fontsize = 12)
    plt.yticks(Fontsize = 12)
    plt.legend(fontsize = 10)
    #plt.ylim(bottom = 0)
    plt.title('Dog Rabies Data vs' + str(len(parambest_set)) + 'Model Estimates', fontsize = 14)
    

plt.savefig(str_date_time + '_rabdash_plot3_' + str(Nw) + '.tiff', dpi=300, bbox_inches='tight')
plt.close()