# -*- coding: utf-8 -*-
"""
Created on Sun May  7 16:48:56 2023

@author: elvee
"""

# Define the SEIV model
def model(y, t, params, fixvar):

    dogrec, dog_loss, dog_vax, ddeath, imp, dogrep, humrec, humloss, hdeath, hrabdeath = fixvar
    b1, h1, h2, prevac, postvac = params
    Sd, Ed, Vd, Red, Sh, Eh, Vh = y
    Nd = Sd + Ed + Vd 
    Nh = Sh + Eh + Vh
    

    dSddt = dogrec + (dog_loss * Vd) - (b1 * Sd * Ed/Nd) - (dog_vax * Sd) - (ddeath * Sd) - (imp * Sd)
    dEddt = (b1 * Sd * Ed/Nd) - (ddeath * Ed) - (imp * Ed) - (dogrep * Ed)
    dVddt = (dog_vax * Sd) - (dog_loss * Vd) - (ddeath * Vd) - (imp * Vd)
    dReddt = dogrep * Ed 
    dShdt = humrec - (h1 * Ed * Sh/Nh) + (humloss * Vh) - (prevac * Sh) - (hdeath * Sh)
    dEhdt = (h1 * Ed * Sh/Nh) - (hrabdeath * Eh) - (hdeath * Eh) + (h2 * Ed * Vh/Nh) - (postvac * Eh)
    dVhdt = (postvac * Eh) - (h2 * Ed * Vh/Nh) - (humloss * Vh) + (prevac * Sh) - (hdeath * Vh)

    return [dSddt, dEddt, dVddt, dReddt, dShdt, dEhdt, dVhdt]
    
    