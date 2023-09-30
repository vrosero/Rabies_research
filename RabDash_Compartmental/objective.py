# -*- coding: utf-8 -*-
"""
Created on Sun May  7 17:22:43 2023

@author: elvee
"""

def objective(params, y0, t, fixvar, ydata,l):
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
#    from pyDOE import lhs
    from sklearn.utils import resample
    from scipy.optimize import minimize
    
    from model import model

    # Integrate the SEIV model
    sol = odeint(model, y0, t, (params, fixvar))
    # Calculate the residuals
    # resid = ydata - sol[:,0]
    resid = ydata - sol[l,3]
    
    objective_val = np.sum(resid**2)
    objective_val = float(objective_val)

    return objective_val