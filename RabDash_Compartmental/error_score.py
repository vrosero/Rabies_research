# -*- coding: utf-8 -*-
"""
Created on Sun May  7 19:26:58 2023

@author: elvee
"""

def error_score(y0, t, params, fixvar, l, ydata):
    from scipy.integrate import odeint
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_absolute_error as mae
    from sklearn.metrics import mean_absolute_percentage_error as mape
    import numpy as np

    from model import model
     
    # Get Solution to system 
    
    sol = odeint(model, y0, t, (params, fixvar))
    Sd, Ed, Vd, Red, Sh, Eh, Vh = sol[:,0], sol[:,1], sol[:,2], sol[:,3], sol[:,4], sol[:,5],sol[:,6]
     
    # Score Difference between model and data points
    rel_se = ((Red[l]-ydata)**2).sum()/(ydata**2).sum()
    error_mse = mse(ydata, Red[l])  
    error_mae = mae(ydata, Red[l]) 
    error_sse = np.sum((Red[l]-ydata)**2)
    error_mape = mae(ydata, Red[l])*100 
     
    return [rel_se, error_mse, error_mae, error_sse, error_mape]