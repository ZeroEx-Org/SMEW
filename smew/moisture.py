# SPDX-License-Identifier: AGPL-3.0-only
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:30:44 2019
"""
import numpy as np
import smew
from numba import njit

# The default for `temp_soil`: no temperature series supplied, so no freezing.
# A module constant rather than None because this function is jitted and numba
# specialises on the type of a default, not on None-ness.
_NO_TEMP = np.empty(0, dtype=np.float64)


@njit
def moisture_balance(rain, Zr, soil, ET0, v, k_v, keyword_wb, s_in, t_end, dt,
                     temp_soil=_NO_TEMP, freeze_on=False):
    """Laio-type bucket water balance.

    FREEZING (P4). With `freeze_on` and a `temp_soil` series of the same length
    as the forcing, evaporation, transpiration and leakage are all set to zero
    wherever `temp_soil < 0`.

    This exists because the model disagreed with itself. biogeochem.py holds the
    chemistry through frost -- it resumes from the last unfrozen state -- while
    this function knew nothing about temperature and kept draining. So the water
    left and the solutes it was carrying did not. Measured on Vulkaneifel, which
    is frozen 30.6 % of its steps: 43.3 % of ALL drainage over the run happened
    during the hold, and every gram of solute in it stayed behind. The hold
    discarded more cation leaching than the model integrated.

    Two modules could have been changed and only one of them is right. Frozen
    soil has a hydraulic conductivity orders of magnitude below thawed soil, so
    it should not drain; making the chemistry leach through the frost instead
    would integrate solute transport in water that is not moving. Zeroing the
    fluxes here fixes both sides at once -- there is then no leaching for the
    hold to miss.

    The threshold is `temp_soil < 0`, the SAME test biogeochem.py uses. That is
    the point of the change: not that zero is the right freezing point for soil
    water (it is not, exactly -- solutes depress it), but that the two modules
    now agree on what "frozen" means.

    Evaporation goes too, not just leakage and transpiration. Ice does not
    evaporate as liquid, and the model has no sublimation term; keeping E alive
    would dry a frozen profile through a pathway that is not there.

    Rain is deliberately still added to storage. Water arriving on frozen ground
    is snow and would not infiltrate, but the model has no snowpack; retaining
    it in the profile and releasing it at thaw is the closer approximation of
    the two available, and it gives the spring pulse the model previously could
    not produce. The wetness ceiling (see P1) sends any excess to runoff, which
    it now records.

    `freeze_on=False` reproduces the previous behaviour exactly, and every
    benchmark case except Vulkaneifel is above freezing throughout, so those are
    bit-identical either way.
    """
    
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
        
        # Only honour freezing when a usable temperature series came with it.
        use_freeze = freeze_on and len(temp_soil) == len(s)

        for i in range(0, len(rain)-1):

            if use_freeze and temp_soil[i] < 0:
                # Frozen: no liquid water leaves the layer, so no solute does
                # either. E, T and L stay at their preallocated zeros and the
                # update below reduces to storage plus whatever fell.
                s[i+1] = s[i]+rain[i+1]/(n*Zr)
                if s[i+1] >= 1:
                    Q[i+1] = (s[i+1]-1)*(n*Zr)
                    s[i+1] = 1
                if s[i+1] >= 0.98:
                    excess = (s[i+1]-0.98)*(n*Zr)
                    Q[i+1] = Q[i+1] + min(excess, max(rain[i+1]-Q[i+1], 0.0))
                    s[i+1] = 0.98
                continue

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