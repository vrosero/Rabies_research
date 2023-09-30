# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 19:14:58 2023

@author: User
"""

"Latin Hypercube Sampling and Partial Rank Correlation Coefficients"


def lhsprcc():    
    # Import Packages
    
    import numpy as np
    from scipy import special
    import random
    from ipywidgets import interact, interactive, fixed, interact_manual
    import ipywidgets as widgets
    from IPython.display import display
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Number of parameters to sample 
    parameterCount = 5;
    
    # Number of samples to draw for each parameter
    sampleCount = 1000; 
    
    
    # Function for specifying sampled parameter's names and distributions
    # as well as drawing samples from user-specified parameter distribution.
    # DOES NOT NEED ANY USER EDITS.
    
    def parNameDist(Name,Distribution):
        paramTemp = {}
        paramTemp['Name']=Name
        paramTemp['Dist']=Distribution
        
        return paramTemp
      
    def sampleDistrib(modelParamName,distrib,distribSpecs): 
        
        if distrib == 'uniform':
            
            mmin = distribSpecs[0].value
            mmax = distribSpecs[1].value
            
            intervalwidth = (mmax - mmin) / sampleCount      # width of each 
                                                             # sampling interval
            samples = []
            
            for sample in range(sampleCount):
                
                lower = mmin + intervalwidth * (sample-1)    # lb of interval
                upper = mmin + intervalwidth * (sample)      # ub of interval
                
                sampleVal = np.random.uniform(lower, upper)  # draw a random sample 
                                                             # within the interval
                samples.append(sampleVal)
    
        
        
        elif distrib == 'normal':
            
            mmean= distribSpecs[0].value
            mvar = distribSpecs[1].value
            
            lower = mvar*np.sqrt(2)*special.erfinv(-0.9999)+mmean # set lb of 1st
                                                                  # sample interval
            samples = []
            
            for sample in range(sampleCount):
              
                n = sample + 1
                
                if n != sampleCount:
                    upper = (np.sqrt(2*mvar)*special.erfinv(2*n/sampleCount-1)
                             + mmean)                        # ub of sample interval
                else:
                    upper = np.sqrt(2*mvar)*special.erfinv(0.9999) + mmean
     
                sampleVal = np.random.uniform(lower, upper)  # draw a random sample 
                                                             # within the interval
        
                samples.append(sampleVal)
    
                lower = upper           # set current ub as the lb for next interval
                
    
        
        elif distrib == 'triangle':
            
            mmin = distribSpecs[0].value
            mmax = distribSpecs[1].value
            mmode= distribSpecs[2].value
        
            samples = []
            
            for sample in range(sampleCount):
              
                n = sample + 1
                
                intervalarea = 1/sampleCount 
                
                ylower = intervalarea*(n-1) # use cdf to read off area as y's &
                yupper = intervalarea*(n)   # get corresponding x's for the pdf
            
            
                # Check to see if y values = cdf(x <= mmode) 
                # for calculating correxponding x values:
                
                if ylower <= ((mmode - mmin)/(mmax - mmin)):     
                    lower = np.sqrt(ylower*(mmax - mmin)*(mmode - mmin)) + mmin 
    
                else:
                    lower = mmax-np.sqrt((1 - ylower)*(mmax - mmin)*(mmax - mmode))
    
                
                if yupper <= ((mmode - mmin)/(mmax - mmin)):    
                    upper = np.sqrt(yupper*(mmax - mmin)*(mmode - mmin)) + mmin; 
    
                else:
                    upper = mmax-np.sqrt((1 - yupper)*(mmax - mmin)*(mmax - mmode))
    
                    
                sampleVal = np.random.uniform(lower, upper)  
                
                samples.append(sampleVal)
                
        
        b = int(np.ceil(sampleCount/10))
        plt.hist(samples, density = 1, bins = b) 
        
        B=str(b)
        
        plt.title('Histogram of ' + modelParamName 
                  + ' parameter samples for ' + B + ' bins')
        
        plt.ylabel('proportion of samples');
        plt.xlabel(modelParamName + ' value')
        
        plt.show()
        
        return samples
    
    
    # Calls the function to ask for user input to name parameters and specify distributions.
    params = {}
    
    for i in range(parameterCount):
      
        s=str(i)
        
        params[i] = interactive(parNameDist,
                                Name='Type parameter ' + s + ' name', 
                                Distribution=['uniform','normal','triangle'])
        
        display(params[i])
    
    # Input parameter distribution specifics in the interactive boxes that appear below after running this.
    distribSpecs={}
    
    for i in range(parameterCount):
      
        parName = params[i].result['Name']
        
        print('Enter distribution specifics for parameter ' + parName + ':')
        
        if params[i].result['Dist'] == 'normal':
    
            distribSpecs[parName] = {}
            
            distribSpecs[parName][0] = widgets.FloatText(
                    value=2,
                    description='Mean:'
                  )
            distribSpecs[parName][1] = widgets.FloatText(
                    value=1,
                    description='Variance:'
                  )
    
            display(distribSpecs[parName][0], distribSpecs[parName][1])
    
        elif params[i].result['Dist'] == 'uniform':
    
            distribSpecs[parName] = {}
    
            distribSpecs[parName][0] = widgets.FloatText(
                    value=0,
                    description='Minimum:'
                  )
            distribSpecs[parName][1] = widgets.FloatText(
                    value=2,
                    description='Maximum:'
                  )
    
            display(distribSpecs[parName][0], distribSpecs[parName][1])
    
    
        elif params[i].result['Dist'] == 'triangle':
          
            distribSpecs[parName] = {}
    
            distribSpecs[parName][0] = widgets.FloatText(
                    value=0,
                    description='Minimum:'
                  )
            distribSpecs[parName][1] = widgets.FloatText(
                    value=2,
                    description='Maximum:'
                  )
            distribSpecs[parName][2] = widgets.FloatText(
                    value=1,
                    description='Mode:'
                  )
    
            display(distribSpecs[parName][0], distribSpecs[parName][1], distribSpecs[parName][2])
            
    # This passes the distributions to the code for generating parameter samples, and histogram plots
    # of samples for each parameter will appear below.
    parameters = {}
    for j in range(parameterCount):
      
        parameters[params[j].result['Name']] = sampleDistrib(params[j].result['Name'],
                                                             params[j].result['Dist'],
                                                             distribSpecs[params[j].result['Name']])
    
    # Randomly permute each set of parameter samples in order to randomly
    # pair the samples to more fully sample the parameter space for the Monte
    # Carlo simulations.
    
    LHSparams=[]
    
    for p in parameters:
        
        temp = parameters[p]
        random.shuffle(temp)
        
        LHSparams.append(temp)
        
    
    # Define model function.
    def myodes(y, t, sampledParams, unsampledParams):
    
        Sd, Ed, Vd, Rd, Sh, Eh, Vh = y  
        Nd = Sd + Ed + Vd 
        Nh = Sh + Eh + Vh                         # unpack current values of y
    
        b1, h1, h2, prevac, postvac = sampledParams # unpack sampled parameters
    
        dogrec, dog_loss, dog_vax, ddeath, imp, dogrep, humrec, humloss, hdeath, hrabdeath = unsampledParams   # unpack unsampled parameters
    
        derivs = [dogrec + (dog_loss * Vd) - (b1 * Sd * Ed/Nd) - (dog_vax * Sd) - (ddeath * Sd) - (imp * Sd),
                  (b1 * Sd * Ed/Nd) - (ddeath * Ed) - (imp * Ed) - (dogrep * Ed),
                  (dog_vax * Sd) - (dog_loss * Vd) - (ddeath * Vd) - (imp * Vd),
                  dogrep * Ed,
                  humrec - (h1 * Ed * Sh/Nh) + (humloss * Vh) - (prevac * Sh) - (hdeath * Sh),
                  (h1 * Ed * Sh/Nh) - (hrabdeath * Eh) - (hdeath * Eh) + (h2 * Ed * Vh/Nh) - (postvac * Eh),
                  (postvac * Eh) - (h2 * Ed * Vh/Nh) - (humloss * Vh) + (prevac * Sh) - (hdeath * Vh)]# list of dy/dt=f functions
                  
        return derivs
    
    # Run Monte Carlo simulations for each parameter sample set.
    # Be sure to specify a call to your model function and any necessary arguments below.
    
    # EDIT THE FOLLOWING VARIABLES, UNSAMPLED PARAMETERS, & ANY OTHER ARGS HERE,
    # AS WELL AS THE CALL TO YOUR OWN MODEL FUNCTION INSIDE THE FOR LOOP BELOW
    
    # EXAMPLE CODE FOR A COUPLED ODE MODEL:
    
    import scipy.integrate as spi
    
    t = np.linspace(0,1,num=108) # time domain for myodes
    
    # odesic = [q0, r0]
    # Sd0, Ed0, Vd0, Rd0, Sh0, Eh0, Vh0, Nd0, Nh0
    odesic = [91537, 36615, 54922, 5, 1593228, 4020, 2613]
    
    dogrec = 2200 # Dog recruitment rate = natural x Nd_0
    dog_vax = 0.08 # Dog vaccination rate 1/12 x percentage
    dog_loss = 0.08 # Dog loss of immunity 1/12 
    ddeath = 0.017 # Dog mortality rate 0.017 x percentage
    imp = 0.0013 # Impounding rate  
    dogrep = 0.2 # Dog reporting rate 
    
    # Human fixed parameters
    humrec = 2280 # Human recruitment rate mortality rate x Nh_0
    humloss = 0.083 # Human loss of immunity Zhang et al
    hdeath = 0.0012 # Human mortality rate 
    hrabdeath = 3 # H
    
    unsampledParams = [dogrec, dog_loss, dog_vax, ddeath, imp, dogrep, humrec, humloss, hdeath, hrabdeath]
    
    Simdata={}
    
    Output = [] 
    
    for i in range(sampleCount):
      
        Simdata[i]={}
        
        Simdata[i]['Sd']=[]
        Simdata[i]['Ed']=[]
        Simdata[i]['Vd']=[]
        Simdata[i]['Rd']=[]
        Simdata[i]['Sh']=[]
        Simdata[i]['Eh']=[]
        Simdata[i]['Vh']=[]
        
        
    for j in range(sampleCount):
    
        sampledParams=[i[j] for i in LHSparams] 
      
        sol=spi.odeint(myodes, odesic, t, args=(sampledParams,unsampledParams)) 
    
        Simdata[j]['Sd'] = sol[:,0] # solution to the equation for variable Sd
        Simdata[j]['Eh'] = sol[:,1] # solution to the equation for variable Ed
        Simdata[j]['Vd'] = sol[:,2] # solution to the equation for variable Vd
        Simdata[j]['Rd'] = sol[:,3] # solution to the equation for variable Rd
        Simdata[j]['Sh'] = sol[:,4] # solution to the equation for variable Sh
        Simdata[j]['Eh'] = sol[:,5] # solution to the equation for variable Eh
        Simdata[j]['Vh'] = sol[:,6] # solution to the equation for variable Vh
        
        
        Ratio = np.divide(sol[:,3],sol[:,5]) # compute ratio to compare w/ param samples
        
        Output.append(Ratio) 
    
    labelstring = 'reported dogs to exposed humans (Rd/Eh)'; # id for fig labels, filenames
    
    # Plot the range simulation output generate by all of the Monte Carlo 
    # simulations using error bars.
    
    yavg = np.mean(Output, axis=0)
    yerr = np.std(Output, axis=0)
    
    plt.errorbar(t,yavg,yerr)
    #plt.xlabel('x')
    plt.xlabel('time (days)')   # for myodes
    plt.ylabel(labelstring)
    plt.title('Error bar plot of ' + labelstring + ' from LHS simulations')
    
    plt.show()
                                                                 
    
    # Compute partial rank correlation coefficients to compare simulation 
    # outputs with parameters.
    
    SampleResult=[]
    
    x_idx = 11          # time or location index of sim results 
    x_idx2= x_idx+1     #   to compare w/ param sample vals
    
    LHS=[*zip(*LHSparams)]
    LHSarray=np.array(LHS)
    Outputarray=np.array(Output)
    subOut=Outputarray[0:,x_idx:x_idx2]
    
    LHSout = np.hstack((LHSarray,subOut))
    SampleResult = LHSout.tolist()
    
    
    Ranks=[]
                  
    for s in range(sampleCount):
    
        indices = list(range(len(SampleResult[s])))
        indices.sort(key=lambda k: SampleResult[s][k])
        r = [0] * len(indices)
        for i, k in enumerate(indices):
            r[k] = i
    
        Ranks.append(r)
    
      
    C=np.corrcoef(Ranks);
    
    if np.linalg.det(C) < 1e-16: # determine if singular
        Cinv = np.linalg.pinv(C) # may need to use pseudo inverse
    else:
        Cinv = np.linalg.inv(C) 
    
    resultIdx = parameterCount+1
    prcc=np.zeros(resultIdx)
    
    for w in range(parameterCount): # compute PRCC btwn each param & sim result
        prcc[w]=-Cinv[w,resultIdx]/np.sqrt(Cinv[w,w]*Cinv[resultIdx,resultIdx])
    
    # Plot the PRCCs for each parameter.
    xp=[i for i in range(parameterCount)]
    
    plt.bar(xp,prcc[0:parameterCount], align='center')
    
    bLabels=list(parameters.keys())
    plt.xticks(xp, bLabels)
    
    plt.ylabel('PRCC value');
    
    N=str(sampleCount)
    loc=str(x_idx)
    plt.title('Partial rank correlation of params with ' + labelstring 
              + ' results \n from ' + N + ' LHS sims, at x = ' +loc);
    
    plt.show()
    
    return prcc
    
    
    "End of Code"