# SPDX-License-Identifier: AGPL-3.0-only
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:34:44 2019
"""

import numpy as np
from numba import njit


#------------------------------------------------------------------------------
 # atmospheric CO2
    
def CO2_atm(conv_mol):
    
    CO2_atm = 412E-6/22.41*conv_mol # [mol-conv/l] 
                                                        
    return CO2_atm

#------------------------------------------------------------------------------
 # soil CO2 diffusivity 
    
def D_0():
    
    D_0 = 1.6E-5*3600*24 #free-air diffusion [m2/d]
                                                        
    return D_0

#------------------------------------------------------------------------------
 # solute diffusivity in soil water
    
def Dw_0():
    
    Dw_0 = 1e-9*3600*24 # [m2/d]
                                                        
    return Dw_0

#------------------------------------------------------------------------------
 # % of nutrients in plant dry matter (ideally plant dependent)
    
def plant_nutr_f():
    
    #current values are from Kelland's (2020) measurement for sorghum
    
    v_f_Ca = 0.005 #suggested 2%, range 0.1 - 5 %, Weil and Bredy 2017
    v_f_Mg = 0.001 #suggested, 0.5%, range 0.1 - 1 %, Weil and Bredy 2017
    v_f_K = 0.01 #suggested 2%, range 1 - 5 %, Weil and Bredy 2017
    v_f_Si = 0.01 #suggested 5%, range 1 - 10%, Epstein 1994, PNAS
    
    v_f = [v_f_Ca, v_f_Mg, v_f_K, v_f_Si]
    
                                                      
    return v_f

#------------------------------------------------------------------------------
 # soil constants

@njit
def soil_const(soil):
    
    # values are from Laio et al., (2001, Adv. Water Resources)
    
    if soil == 'sand':
        s_h = 0.08 #hygroscopic point
        s_w = 0.11 #wilting
        s_i = 0.33 #max transpiration
        #s_fc = 0.35 #field capacity - not needed in the current formulation
        b = 4.05 #power law exponent of the rentention curve
        K_s = 14 #(m/d)
        n = 0.35 #porosity
    elif soil == 'loamy sand':
        s_h = 0.08
        s_w = 0.11
        s_i = 0.31
        #s_fc = 0.52
        b = 4.4
        K_s = 13
        n = 0.42
    elif soil == 'sandy loam':
        s_h = 0.14
        s_w = 0.18
        s_i = 0.46
        #s_fc = 0.56
        b = 4.9
        K_s = 3
        n = 0.43
    elif soil == 'silt loam': # Clapp & Hornberger (2008)
        s_h = 0.22
        s_w = 0.28
        s_i = 0.71
        b = 5.30
        K_s = 0.62
        n = 0.485
    elif soil == 'silt': # Using values for silt loam because not in Clapp & Hornberger (2008)
        s_h = 0.22
        s_w = 0.28
        s_i = 0.71
        b = 5.30
        K_s = 0.62
        n = 0.485
    elif soil == 'loam':
        s_h = 0.19
        s_w = 0.24
        s_i = 0.57
        #s_fc = 0.65
        b = 5.4
        K_s = 0.6
        n = 0.45
    elif soil == 'sandy clay loam': 
        s_h = 0.27
        s_w = 0.32
        s_i = 0.61
        b = 7.12
        K_s = 0.54
        n = 0.42
    elif soil == 'silty clay loam': 
        s_h = 0.32
        s_w = 0.37
        s_i = 0.68
        b = 7.75
        K_s = 0.15
        n =  0.477
    elif soil == 'clay loam':
        s_h = 0.39
        s_w = 0.45
        s_i = 0.68
        b = 8.52
        K_s = 0.2
        n = 0.47
    elif soil == 'sandy clay': 
        s_h = 0.39
        s_w = 0.44
        s_i = 0.69
        b = 10.4
        K_s = 0.19
        n = 0.426
    elif soil == 'silty clay': 
        s_h = 0.43
        s_w = 0.49
        s_i = 0.76
        b = 10.4
        K_s = 0.09
        n = 0.492
    elif soil == 'clay':
        s_h = 0.47
        s_w = 0.52
        s_i = 0.78
        b = 11.4
        K_s = 0.11
        n = 0.5
    else:
        raise ValueError("Invalid soil type!")
                                                      
    return s_h, s_w, s_i, b, K_s, n

#------------------------------------------------------------------------------
 # EW mineral constants 

def min_const(mineral,conv_mol):

    if mineral == 'albite': #NaAlSi3O8
        MM_min = 262.219 # g/mol
        min_st = [0, 0, 0, 1, 1, 3]# Stochiometric coefficients [Ca, Mg, K, Na, Al, Si]

    elif mineral == 'anorthite': #CaAl2Si2O8
        MM_min = 278.204
        min_st = [1, 0, 0, 0, 2, 2]

    elif mineral == 'augite': #Mg0.45Fe0.275Ca0.275SiO3
        MM_min = 113.4
        min_st = [0.275, 0.45, 0, 0, 0, 1]# Stochiometric coefficients [Ca, Mg, K, Na, Al, Si]

    elif mineral == 'basalt_glass': #SiTi0.02Al0.36Fe0.19Mg0.28Ca0.26Na0.08K0.008O3.364
        MM_min = 122.566
        min_st = [0.26, 0.28, 0.008, 0.08, 0.36, 1]# Stochiometric coefficients [Ca, Mg, K, Na, Al, Si]

    elif mineral == 'chabazite_Ca': # CaAl2Si4O12:6H2O
        MM_min = 398.37
        min_st = [1, 0, 0, 0, 2, 4]

    elif mineral == 'clinoptilolite_Ca': #Ca1.5Al3Si15O36:12H2O
        MM_min = 1138.302
        min_st = [1.5, 0, 0, 0, 3, 15]

    elif mineral == 'clinoptilolite_Na': #Na3Al3Si15O36:10H2O
        MM_min = 1147.155
        min_st = [0, 0, 0, 3, 3, 15]

    elif mineral == 'diopside': #MgCaSi2O6
        MM_min = 216.547
        min_st = [1, 1, 0, 0, 0, 2]# Stochiometric coefficients [Ca, Mg, K, Na, Al, Si]

    elif mineral == 'forsterite': #Mg2SiO4
        MM_min = 140.692
        min_st = [0, 2, 0, 0, 0, 1]# Stochiometric coefficients [Ca, Mg, K, Na, Al, Si]
            
    #elif mineral == 'Fe_forsterite': #FeMgSiO4
    #    min_st = [0, 1, 0, 0, 0, 1]# Stochiometric coefficients [Ca, Mg, K, Na, Al, Si]

    elif mineral == 'heulandite_Ca': # CaAl2Si7O18:6H2O
        MM_min = 578.619
        min_st = [1, 0, 0, 0, 2, 7]

    elif mineral == 'heulandite_Na': # Na2Al2Si7O18:5H2O
        MM_min = 584.521
        min_st = [0, 0, 0, 2, 2, 7]

    elif mineral == 'hydroxyapatite': #Ca5(OH)(PO4)3
        MM_min = 502.31
        min_st = [5, 0, 0, 0, 0, 0]

    elif mineral == 'K_feldspar': #KAlSi3O8
        MM_min = 278.33
        min_st = [0, 0, 1, 0, 1, 3]
    
    elif mineral == 'labradorite': # Ca0.68Na0.32Al1.68Si2.32O8
        MM_min = 245.84
        min_st = [0.68, 0, 0, 0.32, 1.68, 2.32]

    elif mineral == 'leucite': #K(AlSi2O6) # From Bertagni et al. (2025) original code
        MM_min = 218
        min_st = [0, 0, 1, 0, 1, 2]# Stochiometric coefficients [Ca, Mg, K, Na, Al, Si]

    elif mineral == 'muscovite': #KAl3Si3O10(OH)2
        MM_min = 398.303
        min_st = [0, 0, 1, 0, 3, 3]# Stochiometric coefficients [Ca, Mg, K, Na, Al, Si]

    elif mineral == 'nepheline': #NaAlSiO4
        MM_min = 142.053
        min_st = [0, 0, 0, 1, 1, 1]# Stochiometric coefficients [Ca, Mg, K, Na, Al, Si]

    elif mineral == 'olivine': #Mg1.8Fe0.2SiO4
        MM_min = 147.31
        min_st = [0, 1.8, 0, 0, 0, 1]
            
    elif mineral == 'wollastonite': #CaSiO3
        MM_min = 117.1
        min_st = [1, 0, 0, 0, 0, 1]# Stochiometric coefficients [Ca, Mg, K, Na, Al, Si]

    elif mineral == 'alkali_feldspar': #K0.41Na0.56Ca0.03Al1.03Si2.97O8 (Kelland et al., 2020)
        MM_min = 156
        min_st = [0.03, 0, 0.41, 0.56, 1.03, 2.97]# Stochiometric coefficients [Ca, Mg, K, Na, Al, Si]

    else:
        raise ValueError("No data for this mineral")

    MM_min = MM_min/conv_mol
                                            
    return(MM_min,min_st)

#------------------------------------------------------------------------------
 # Carbonate weathering constants  
    

def carb_weath_const(conv_mol):
    
    # Carbonate solubility products
    #https://booksite.elsevier.com/9780120885305/appendices/Web_Appendices.pdf
    K_CaCO3 = 10**(-8.35)*conv_mol**2 # Calcite
    K_MgCO3 = 10**(-7.46)*conv_mol**2 # Magnesite
    
    #precipitation rate
    # modeling formulation follows https://nora.nerc.ac.uk/id/eprint/511084/1/Kirk%20et%20al%202015%20Geochmica%20et%20Cosmochimica%20Acta.pdf
    # parameter values are defined to reproduce experimental results
    r_CaCO3 = 3*1e7*(1e-9*1e6/(24*3600))*conv_mol # [mol-conv/d] 
    r_MgCO3 = 1e7*(1e-9*1e6/(24*3600))*conv_mol
    
    #dissolution timescale
    tau_CaCO3 = 30 # [d] 
    tau_MgCO3 = 40 
    
    return(K_CaCO3,K_MgCO3,r_CaCO3,r_MgCO3,tau_CaCO3,tau_MgCO3)
    
#------------------------------------------------------------------------------
 # CEC constants (Gaines-Thomas) 

def K_GT_CEC(soil, conv_mol):
            
    # Current values are from a meta-analysis of Dutch soils (0-30 cm, https://edepot.wur.nl/31605)
    # Site-specific coefficient estimates can be obtained with soil-water coupled measurements
    
    if soil in ['sand', 'loamy sand', 'sandy loam']:
        K_Ca_Mg = 10**(0.56); #[-]
        K_Ca_K = 10**(-1.16)*conv_mol #[conc]
        K_Ca_Na = 10**(0.75)*conv_mol #[conc]
        K_Ca_H = 10**(-5)*conv_mol #[conc] for tePas et al a better constant is 0.5 1e-9 *conv_mol,
        K_Ca_Al = 10**(-1.7)/conv_mol #[conc^-1] #heterovalent (make it higher to favor Ca adsorbed) 
        #K_Ca_AlOH = 10**(-1.7)
        #K_Ca_AlOH2 = 10**(-1.7)*conv_mol
    
    elif soil in ['loam', 'silty loam', 'silt']:
        K_Ca_Mg = 10**(0.1); #[-]
        K_Ca_K = 10**(-2)*conv_mol
        K_Ca_Na = 10**(0.38)*conv_mol
        K_Ca_H = 10**(-5.4)*conv_mol
        K_Ca_Al = 10**(-0.86)/conv_mol
        #K_Ca_AlOH = 10**(-0.86)
        #K_Ca_AlOH2 = 10**(-0.86)*conv_mol
        
    elif soil in ['clay', 'clay loam', 'silty clay']:
        K_Ca_Mg = 10**(0.39); #[-]
        K_Ca_K = 10**(-2.42)*conv_mol
        K_Ca_Na = 10**(0.774)*conv_mol
        K_Ca_H = 10**(-6.67)*conv_mol
        K_Ca_Al = 10**(-0.2)/conv_mol
        #K_Ca_AlOH = 10**(-0.2)
        #K_Ca_AlOH2 = 10**(-0.2)*conv_mol
    
    else:
        raise ValueError("Unknown soil type")
        
    # K_Na_K = (K_Ca_K/K_Ca_Na)**(1/2)]
    # K_Na_Al = (K_Ca_Al/K_Ca_Na**3)**(1/2)
   
    K_CEC = [K_Ca_Mg, K_Ca_K, K_Ca_Na, K_Ca_Al, K_Ca_H]
                                          
    return(K_CEC)

#------------------------------------------------------------------------------
 # Aluminium speciation (pag. 398 Weil and Brady)

def K_Al(conv_mol):
              
    pK1 = 5 
    pK2 = 5.1 
    pK3 = 6.7 
    pK4 = 6.2
    K1 = 10**(-pK1)*conv_mol
    K2 = 10**(-pK2)*conv_mol
    K3 = 10**(-pK3)*conv_mol
    K4 = 10**(-pK4)*conv_mol
   
    K_Al = [K1, K2, K3, K4]
                                          
    return(K_Al)

#------------------------------------------------------------------------------
 # carbonate speciation [Stumm and Morgan, 1996]

def K_C(T_K,conv_mol):
    
    T_ref = 25+273.15 # [K]: temperature standard conditions
    
    #carbonates
    pk1 = -(-356.309 - 0.0609 * T_K + 21834.37/T_K + 126.8339 * np.log10(T_K) - 1684915/(T_K)**2)
    pk2 = -(-107.887 - 0.032528 * (T_K) + 5151.79/(T_K) + 38.92561 * np.log10(T_K) - 563713.9/(T_K)**2)
    pk_w = -(-283.971 + 13323/(T_K) - 0.0507 * (T_K) + 102.24447 * np.log10(T_K) - 1119669/(T_K)**2) 
    k1 = 10**(-pk1)*conv_mol
    k2 = 10**(-pk2)*conv_mol
    
    #water
    k_w = 10**(-pk_w)*conv_mol**2
    
    #Henry
    k_H = 0.83*np.exp(2400*(1/T_K-1/T_ref))
                                          
    return(k1, k2, k_w, k_H)

#------------------------------------------------------------------------------
 # molar masses [g/mol]

def MM(conv_mol):
    
    MM_Mg = 24/conv_mol 
    MM_Ca = 40/conv_mol 
    MM_Na = 23/conv_mol
    MM_K = 39/conv_mol 
    MM_Si = 28/conv_mol 
    MM_C = 12/conv_mol
    MM_Anions = 62/conv_mol
    MM_Al = 27/conv_mol
                                          
    return(MM_Mg, MM_Ca, MM_Na, MM_K, MM_Si, MM_C, MM_Anions, MM_Al)