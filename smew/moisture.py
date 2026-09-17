# SPDX-License-Identifier: AGPL-3.0-only
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:30:44 2019
"""
import numpy as np
import smew
from numba import njit


@njit
def moisture_balance(rain, Zr, soil, ET0, v, k_v, keyword_wb, s_in,t_end,dt):
    
    #constants
    [s_h, s_w, s_i, b, K_s, n] = smew.soil_const(soil)       
    
    # Initialization
    #--------------------------------------------------------------------------      
    if keyword_wb == 1:
        s=np.zeros((len(rain)))
        s[0] = s_in # initial value
    elif keyword_wb == 0:
        s = s_in*np.ones(round(t_end/dt))
        
    L = np.zeros((len(s)))
    E = np.zeros((len(s)))
    T = np.zeros((len(s)))
    Q = np.zeros((len(s)))
    Irr = np.zeros((len(s)))
    
    # moisture dynamics
    #--------------------------------------------------------------------------      
    if keyword_wb == 1:

        # Potential evaporation [m/d]
        E0 = 0.5*ET0
        
        for i in range(0, len(rain)-1):
            
            # Evaporation [m/d]
            if s[i]<=s_h:
                E[i] = 0
            elif s[i]<=s_i:
                E[i] = (s[i]-s_h)/(s_i-s_h)*E0[i]*(1-v[i]/k_v)
            elif s[i]<=1:
                E[i] = E0[i]*(1-v[i]/k_v)

            # Transpiration [m/d]
            if s[i]<=s_w:
                T[i] = 0
            elif s[i]<=s_i:
                T[i] = (s[i]-s_w)/(s_i-s_w)*ET0[i]*v[i]/k_v
            elif s[i]<=1:
                T[i] = ET0[i]*v[i]/k_v

            # Leakage [m/d]
            L[i] = K_s*s[i]**(3+2*b)
            
            # Moisture dynamics
            s[i+1] = s[i]+rain[i+1]/(n*Zr)-((E[i]+T[i]+L[i])/(n*Zr)*dt)

            #runoff [m]
            if s[i+1] >= 1:
                Q[i+1] = (s[i+1]-1)*(n*Zr)
                s[i+1] = 1  

            # Wetness ceiling, kept "to avoid numerical issues" -- but the water
            # it removes is now RECORDED as runoff instead of deleted.
            #
            # The clamp above it is properly accounted: excess over saturation
            # becomes Q, and I = rain - Q carries that through. This one fired
            # straight after it and dropped a further 0.02*n*Zr with no record
            # anywhere. Every solute concentration in this model is a pool over
            # a water volume, so water removed silently CONCENTRATES everything
            # left behind -- and the mass-balance ledger's water row could only
            # report the loss, not close.
            #
            # Routing it to Q is the conservative reading and the one consistent
            # with the line above: this is rain the soil could not take, so it
            # never infiltrated. It therefore carries no solutes out, which is
            # what makes it a bookkeeping fix rather than a new loss pathway.
            #
            # The ceiling can only be reached on a step that received rain --
            # with rain[i+1] = 0 the update is monotonically decreasing and s[i]
            # is already at or below 0.98 -- so attributing the water to runoff
            # is always available. The min() is a guard on the one case that
            # breaks that argument, an initial condition set above the ceiling;
            # it keeps I = rain - Q non-negative. Measured across the suite, it
            # never binds.
            if s[i+1] >= 0.98:
                excess = (s[i+1]-0.98)*(n*Zr)
                Q[i+1] = Q[i+1] + min(excess, max(rain[i+1]-Q[i+1], 0.0))
                s[i+1] = 0.98

        #infiltration [m]
        I=rain-Q
    
    # constant moisture
    #-------------------------------------------------------------------------- 
    if keyword_wb == 0:
          
        # leakage [m/d]
        L = K_s*s**(3+2*b)
        # Evaporation [m/d]
        if s_in>=s_h:
            E = (s_in-s_h)/((s_i+1)/2-s_h)*ET0*(1-v/k_v)
        # Transpiration [m/d]
        if s_in>=s_w and s_in<=s_i:
            T = (s-s_w)/(s_i-s_w)*ET0*v/k_v
        elif s_in>=s_i:
            T = ET0*v/k_v
       
        rain = E+T+L #[m]
        I = E+T+L 
        Q = np.zeros(len(s))       
    
    return s, s_w, s_i, I, L, T, E, Q, Irr, n