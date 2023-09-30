# -*- coding: utf-8 -*-
"""
Created on Sun May  7 21:15:22 2023

@author: elvee
"""

def param_min(lhs_param_sse_table):
    import numpy as np
    import pandas as pd
    
    sse_sort = lhs_param_sse_table.sort_values(by=["SSE"], ascending = [True])
    sse_sort_parameters = sse_sort[["param_b1","param_h1", "param_h2", "param_prevac", "param_postvac"]]   #select the parameter estimates
    sse_arr_param = sse_sort_parameters.to_numpy()
    # sse_least_error = sse_arr_param[0,:] #selecting first row based on least SSE
    sse_least_error = sse_arr_param[np.r_[0:50],:] #selecting first 20 rows based on least SSE
    # print(sse_least_error)

    return sse_least_error
    