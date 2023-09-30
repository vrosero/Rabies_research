# -*- coding: utf-8 -*-
"""
Created on Sun May  7 18:23:52 2023

@author: elvee
"""

def param_est(no_params, paramInit, y0, t, fixvar, ydata, l):
    # from scipy.optimize import minimize
    from scipy.optimize import curve_fit
    
    from objective import objective
    
    # # with constraint, choose COBYLA or SLSQP
    # for n in range(no_params):
    #     if n == 0:
    #         cons = [{'type': 'ineq', 'fun' : lambda x: x[0]}]
    #     else:
    #         cons.append[{'type': 'ineq', 'fun' : lambda x: x[n]}]
            
    # cons = tuple(cons)
    answ, covariance = curve_fit(objective, paramInit, (y0, t, fixvar, ydata,l),
                                 method='trf', options={'maxiter':1000})
    # cons = ({'type': 'ineq', 'fun' : lambda x: x[0]},
    #     {'type': 'ineq', 'fun' : lambda x: x[1]},
    #     {'type': 'ineq', 'fun' : lambda x: x[2]},
    #     {'type': 'ineq', 'fun' : lambda x: x[3]},
    #     {'type': 'ineq', 'fun' : lambda x: x[4]})

    # answ = minimize(objective, paramInit, (y0, t, fixvar, ydata,l),
    #                 method='COBYLA', constraints=cons, options={'maxiter':10000})
    # answ = curve_fit(objective, paramInit, (y0, t, fixvar, ydata,l),
    #                 method='trf', constraints=cons, options={'maxiter':10000})

    b1_ops, h1_ops, h2_ops, c1_ops, c2_ops = answ  
    
    return b1_ops, h1_ops, h2_ops, c1_ops, c2_ops