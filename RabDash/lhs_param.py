# -*- coding: utf-8 -*-
"""
Created on Sun May  7 17:57:52 2023

@author: elvee
"""

def lhs_param(no_params, no_sets, l_bounds, u_bounds):
    from scipy.stats.qmc import LatinHypercube
    from scipy.stats import qmc

    #To generate LHS samples of the estimates
    sampler = LatinHypercube(d=no_params, seed=2)
    sample = sampler.random(n=no_sets)


    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    
    return sample_scaled
