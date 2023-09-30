# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 18:30:24 2023

@author: User
"""

def sol_bootstrap(realizations, p_init_true, y0, t, fixvar, ydata, l, no_params):
    from scipy.integrate import odeint
    from scipy.stats import poisson
    import numpy as np
    from scipy.optimize import minimize
    
    
    from param_est import param_est
    from model import model
    from objective import objective
    from plims import plims
    
    
    n = len(ydata) #length of the data
    
    
    #store the parambest to ptrue
    ptrue = param_est(no_params, p_init_true, y0, t, fixvar, ydata, l)
    
    
    #model solution using ptrue
    sol = odeint(model, y0, t, (ptrue, fixvar))

    #using the fourth col of sol only    
    Red_sol = sol[:,3]
    newdatabase = Red_sol
    
    #creating matrix
    newdatabase_mat = np.zeros((realizations,len(ydata)))
    print(newdatabase_mat)
    new_cum_data_mat = np.zeros((realizations,len(ydata)))
    boot_param_est_mat = np.zeros((realizations,no_params))
    boot_sol_mat = np.zeros((realizations,len(t)))
    boot_sol_l_mat = np.zeros((realizations,len(l)))
    
    plims_mat = np.zeros((3,len(t)))
    plims_l_mat = np.zeros((3,len(l)))
    
    newdatabase_mat[:,0] = ydata[0]
    
    for n in range(realizations):
        print('realization: ' + str(n))
        
        for m in range(1,len(l)):
            newcases_t = newdatabase[m]-newdatabase[m-1]
            newdatabase_mat[n,m] = poisson.rvs(newcases_t,1,1)
            print(newcases_t)
            print(newdatabase_mat[n,m])
        
        new_cum_data_mat[n,:] = np.cumsum(newdatabase_mat[n,:]) 
        
    
    # for n in range(realizations):
        ydata_boot = new_cum_data_mat[n,:]
        ptrue = p_init_true
        
        cons_boot = ({'type': 'ineq', 'fun' : lambda x: x[0]},
            {'type': 'ineq', 'fun' : lambda x: x[1]},
            {'type': 'ineq', 'fun' : lambda x: x[2]},
            {'type': 'ineq', 'fun' : lambda x: x[3]},
            {'type': 'ineq', 'fun' : lambda x: x[4]})
        
        param_boot = minimize(objective, ptrue, (y0, t, fixvar, ydata_boot, l),
                    method='SLSQP', constraints=cons_boot, options={'maxiter':10000})
        
        # Matrix for bootstrapped parameter estimates
        param_boot_new = param_boot.get('x')    
        boot_param_est_mat[n,:] = param_boot_new
        
        # Fit model to bootstrap samples
        # Save the solution
        sol_boot = odeint(model, y0, t, (param_boot_new, fixvar))
        Red_boot = sol_boot[:,3]
        boot_sol_mat[n,:] = Red_boot
        boot_sol_l_mat[n,:] = Red_boot[l]
        
        # print(boot_param_est_mat)
        
    plims_sample = plims(boot_sol_mat)
    
    return [boot_param_est_mat, boot_sol_mat, boot_sol_l_mat, new_cum_data_mat, plims_sample]

    #add computed mean, std, mode, median