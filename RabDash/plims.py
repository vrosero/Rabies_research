# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 19:26:36 2023

@author: User
"""

import numpy as np

def plims(sol_boot):
     # Calculate percentiles from bootstrapped samples at each time.
     # Arguments:
         # sol_boot: a 2D array of bootstrapped samples, where each row represents a time and each column represents a sample.
         # percentiles: a list of percentiles to calculate
     sorted_samples = np.sort(sol_boot, axis=0)
     percentiles = [2.5, 50, 97.5]
     percentiles_array = np.percentile(sorted_samples, percentiles, axis=0)
     
     # View mean, standard deviation, & 95% bootstrapped confidence interval
     print("25th percentile:", percentiles_array[0])
     print("50th percentile:", percentiles_array[1])
     print("75th percentile:", percentiles_array[2])
     
     return percentiles_array