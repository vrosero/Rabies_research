# -*- coding: utf-8 -*-
"""
Created on Sun May  7 17:01:35 2023

@author: elvee
"""

def data_import():
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    from sklearn.utils import resample
    from scipy.optimize import minimize
    import statistics
    
    # Retrieve Data from Excel
    xls  = pd.ExcelFile('rabiesdata.xlsx')                  
    
    # read the sheet
    sheet  = pd.read_excel(xls);
    
    rabiesdataI = sheet['I'].to_numpy()
    rabiesdatat  = sheet['Month'].to_numpy()
    newcumu_dataI = pd.Series(rabiesdataI)
    cumu_dataI = newcumu_dataI.cumsum()
    
    # Cumulative data
    ydata = cumu_dataI

    
    # Dog fixed parameters
    dogrec = 1800; # Dog recruitment rate = natural x Nd_0
    dog_vax = 0.08; # Dog vaccination rate 1/12 x percentage
    dog_loss = 0.08; # Dog loss of immunity 1/12 
    ddeath = 0.017 # Dog mortality rate 0.017 x percentage
    imp = 0.0013; # Impounding rate  
    dogrep = 0.2 # Dog reporting rate 

    # Human fixed parameters
    humrec = 1900; # Human recruitment rate mortality rate x Nh_0
    humloss = 0.083; # Human loss of immunity Zhang et al
    hdeath = 0.0012; # Human mortality rate 
    hrabdeath = 3; # Human death due to rabies
    
    # Parameters to estimate
    b1 = 0.1; # Rosero et al.
    h1 = 0.4;
    h2 = 0.4;
    prevac = 0.4;
    postvac = 0.4;

    # Model Initial Conditions
    # TO MODIFY: ON A MONTHLY BASIS
    Nd0 = 11000 #133000 #Animal Survey 2014
    Sd0  = 6650 #60% of Nd0 divide by 12
    Ed0 = 1995 #30% of Sd_0
    Vd0 = 2800 #Vaccinated Dogs January 2014
    Red0  = rabiesdataI[0] #rabies data
    Nh0 = 1593000 #Human Population 2014
    Sh0  = 1592405
    Eh0 = 457 # January 2016 #4020 Dog Bite Victims in 2016
    Vh0 = 138 # 30% of January 2016 #2613 #Dog Bite Victims in 2016 complete session
    # Note: 457 dog bite victims January 2016


    
    fixvar = [dogrec, dog_loss, dog_vax, ddeath, imp, dogrep, humrec, humloss, hdeath, hrabdeath]
    y0 = [Sd0, Ed0, Vd0, Red0, Sh0, Eh0, Vh0]
    params = [b1, h1, h2, prevac, postvac]
    
    return rabiesdatat, ydata, fixvar, y0, params