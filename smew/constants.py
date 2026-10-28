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
    """Molar mass and cation stoichiometry of one mineral.

    min_st is [Ca, Mg, K, Na, Al, Si, Fe] per formula unit. IRON IS COLUMN 6 and
    was added by P3: it is present in the formulas above -- augite Fe0.275,
    basalt glass Fe0.19, olivine Fe0.2 -- but had no column, so the rock lost
    its full formula mass while only part of its cations reached solution. See
    the Fe_tot balance in biogeochem.py for what is and is not done with it.

    Titanium is still dropped (basalt glass carries Ti0.02). That is deliberate:
    Ti is effectively immobile in soil solution and contributes no alkalinity,
    so it is a rounding term on the mass balance rather than a missing flux.
    """

    if mineral == 'albite': #NaAlSi3O8
        MM_min = 262.219 # g/mol
        min_st = [0, 0, 0, 1, 1, 3, 0]# [Ca, Mg, K, Na, Al, Si, Fe]

    elif mineral == 'anorthite': #CaAl2Si2O8
        MM_min = 278.204
        min_st = [1, 0, 0, 0, 2, 2, 0]

    elif mineral == 'augite': #Mg0.45Fe0.275Ca0.275SiO3
        MM_min = 113.4
        min_st = [0.275, 0.45, 0, 0, 0, 1, 0.275]# [Ca, Mg, K, Na, Al, Si, Fe]

    elif mineral == 'basalt_glass': #SiTi0.02Al0.36Fe0.19Mg0.28Ca0.26Na0.08K0.008O3.364
        MM_min = 122.566
        min_st = [0.26, 0.28, 0.008, 0.08, 0.36, 1, 0.19]# [Ca, Mg, K, Na, Al, Si, Fe]

    elif mineral == 'chabazite_Ca': # CaAl2Si4O12:6H2O
        MM_min = 398.37
        min_st = [1, 0, 0, 0, 2, 4, 0]

    elif mineral == 'clinoptilolite_Ca': #Ca1.5Al3Si15O36:12H2O
        MM_min = 1138.302
        min_st = [1.5, 0, 0, 0, 3, 15, 0]

    elif mineral == 'clinoptilolite_Na': #Na3Al3Si15O36:10H2O
        MM_min = 1147.155
        min_st = [0, 0, 0, 3, 3, 15, 0]

    elif mineral == 'diopside': #MgCaSi2O6
        MM_min = 216.547
        min_st = [1, 1, 0, 0, 0, 2, 0]# [Ca, Mg, K, Na, Al, Si, Fe]

    elif mineral == 'enstatite': #MgSiO3
        MM_min = 100.39
        min_st = [0, 1, 0, 0, 0, 1, 0]# [Ca, Mg, K, Na, Al, Si, Fe]

    elif mineral == 'forsterite': #Mg2SiO4
        MM_min = 140.692
        min_st = [0, 2, 0, 0, 0, 1, 0]# [Ca, Mg, K, Na, Al, Si, Fe]
            
    #elif mineral == 'Fe_forsterite': #FeMgSiO4
    #    min_st = [0, 1, 0, 0, 0, 1]# [Ca, Mg, K, Na, Al, Si, Fe]

    elif mineral == 'heulandite_Ca': # CaAl2Si7O18:6H2O
        MM_min = 578.619
        min_st = [1, 0, 0, 0, 2, 7, 0]

    elif mineral == 'heulandite_Na': # Na2Al2Si7O18:5H2O
        MM_min = 584.521
        min_st = [0, 0, 0, 2, 2, 7, 0]

    elif mineral == 'hydroxyapatite': #Ca5(OH)(PO4)3
        MM_min = 502.31
        min_st = [5, 0, 0, 0, 0, 0, 0]

    elif mineral == 'K_feldspar': #KAlSi3O8
        MM_min = 278.33
        min_st = [0, 0, 1, 0, 1, 3, 0]
    
    elif mineral == 'labradorite': # Ca0.68Na0.32Al1.68Si2.32O8
        MM_min = 245.84
        min_st = [0.68, 0, 0, 0.32, 1.68, 2.32, 0]

    elif mineral == 'leucite': #K(AlSi2O6) # From Bertagni et al. (2025) original code
        MM_min = 218
        min_st = [0, 0, 1, 0, 1, 2, 0]# [Ca, Mg, K, Na, Al, Si, Fe]

    elif mineral == 'muscovite': #KAl3Si3O10(OH)2
        MM_min = 398.303
        min_st = [0, 0, 1, 0, 3, 3, 0]# [Ca, Mg, K, Na, Al, Si, Fe]

    elif mineral == 'nepheline': #NaAlSiO4
        MM_min = 142.053
        min_st = [0, 0, 0, 1, 1, 1, 0]# [Ca, Mg, K, Na, Al, Si, Fe]

    elif mineral == 'olivine': #Mg1.8Fe0.2SiO4
        MM_min = 147.31
        min_st = [0, 1.8, 0, 0, 0, 1, 0.2]
            
    elif mineral == 'wollastonite': #CaSiO3
        MM_min = 117.1
        min_st = [1, 0, 0, 0, 0, 1, 0]# [Ca, Mg, K, Na, Al, Si, Fe]

    elif mineral == 'alkali_feldspar': #K0.41Na0.56Ca0.03Al1.03Si2.97O8 (Kelland et al., 2020)
        MM_min = 156
        min_st = [0.03, 0, 0.41, 0.56, 1.03, 2.97, 0]# [Ca, Mg, K, Na, Al, Si, Fe]

    elif mineral == 'sio2(a,gl)': #SiO2 (amorphous silica/glass); GFW and density from MINTEQA2
        MM_min = 60.0848
        min_st = [0, 0, 0, 0, 0, 1, 0]# [Ca, Mg, K, Na, Al, Si, Fe]
        # No KINEC/kinetic rate law available for this phase yet -> not usable in mineral_weathering()/biogeochem_balance.

    else:
        raise ValueError("No data for this mineral")

    MM_min = MM_min/conv_mol
                                            
    return(MM_min,min_st)

#------------------------------------------------------------------------------
 # Carbonate weathering constants  
    

def carb_weath_const(conv_mol, T_K=None, mg_phase="magnesite"):
    """Carbonate solubility products and precipitation/dissolution rates.

    T_K is the soil temperature [K]. Scalar or array, following K_C. Passing it
    makes CALCITE temperature-dependent; omitting it reproduces the fixed
    pre-existing value exactly, which is what keeps every other caller
    (ic.py, the notebooks) bit-identical.

    WHY CALCITE NEEDED THIS. It was pinned at pKsp 8.35 for every soil
    temperature. Every other equilibrium constant the model carries varies with
    temperature -- K_C does it properly -- and calcite sets the secondary
    carbonate term, which is the model's main CO2 LOSS pathway. A fixed value
    makes the model over-predict saturation, and it does so worst in warm soil:
    measured against the expression below, Omega_calcite came out 9 % high at
    2 C, 35 % at 25 C and 44 % at 30 C.

    The expression is Plummer & Busenberg (1982), Geochim. Cosmochim. Acta 46,
    1011-1040, in the analytic form phreeqc.dat carries verbatim:

        log10 Ksp = A1 + A2*T + A3/T + A4*log10(T)

    Its value at 25 C, -8.480, is independently reproduced by MINTEQA2 at
    -8.475 -- so this is not one database's opinion.

    THE MAGNESIUM PHASE IS A CHOICE, and the default keeps the old behaviour.

    `mg_phase="magnesite"` (default) uses the fixed -7.46 this model has always
    used. Its value is disputed across sources -- this model -7.46, pitzer.dat
    -7.834, MINTEQA2 -8.029, a spread of 0.57 log units -- and it is left fixed
    rather than improved, because the deeper problem is not the number.

    `mg_phase="nesquehonite"` (MgCO3:3H2O) is the phase that actually forms.
    Below roughly 100 C, anhydrous magnesite is kinetically inhibited and does
    not precipitate; what forms instead is nesquehonite, ageing through
    dypingite to hydromagnesite. So on a Mg-rich feedstock the default is
    modelling a mineral that is not there, at a solubility ~1.8 log units too
    LOW -- i.e. it precipitates Mg carbonate far too readily.

    Nesquehonite is OFF by default because the evidence has to come first: it
    should be switched on for a site where an Mg carbonate has actually been
    identified, not on the general argument above. Turning it on changes
    results; see the `mg_phase` note in biogeochem_balance.

    Values for nesquehonite: log Ksp = -5.621 at 25 C with dissolution enthalpy
    -5.789 kcal/mol, from MINTEQA2 (as carried by MIN3P's mineral.dbs), applied
    through van't Hoff. pitzer.dat gives -5.167 for the same phase, so treat the
    0.45 log-unit spread between compilations as the uncertainty on this number.
    The three waters of hydration are not tracked: precipitating a few micromoles
    of nesquehonite removes a negligible fraction of the ~55 mol/l of solvent,
    and Omega keeps the same form because a_H2O^3 is ~1 in soil solution.

    NOT ADJUSTED WITH THE PHASE: r_MgCO3 and tau_MgCO3, the precipitation rate
    and dissolution timescale, were tuned to reproduce experiments under the
    magnesite assumption. They are carried over unchanged, which makes
    nesquehonite a thermodynamic switch rather than a fully re-parameterised
    phase. Anyone turning it on should expect to revisit them.
    """
    if mg_phase not in ("magnesite", "nesquehonite"):
        raise ValueError("mg_phase must be 'magnesite' or 'nesquehonite', "
                         "got %r" % (mg_phase,))
    # Carbonate solubility products
    #https://booksite.elsevier.com/9780120885305/appendices/Web_Appendices.pdf
    if T_K is None:
        K_CaCO3 = 10**(-8.35)*conv_mol**2 # Calcite, fixed (pre-P5 behaviour)
    else:
        # Plummer & Busenberg (1982); coefficients as in phreeqc.dat
        logK = (-171.9065 - 0.077993*T_K + 2839.319/T_K
                + 71.595*np.log10(T_K))
        K_CaCO3 = 10**(logK)*conv_mol**2
    if mg_phase == "magnesite":
        K_MgCO3 = 10**(-7.46)*conv_mol**2   # fixed -- see the docstring
        if T_K is not None:
            # Broadcast to T_K's shape so every caller gets the same container
            # back, whether or not the constant depends on temperature.
            K_MgCO3 = K_MgCO3 * np.ones_like(np.asarray(T_K, dtype=float))
    else:
        # Nesquehonite, MgCO3:3H2O = Mg+2 + CO3-2 + 3 H2O.
        # van't Hoff from MINTEQA2's 25 C value and dissolution enthalpy:
        #   log K(T) = log K(298.15) - dH/(2.303 R) * (1/T - 1/298.15)
        _LK25, _DH = -5.621, -5.789*4184.0          # [-], [J/mol]
        if T_K is None:
            logK = _LK25
        else:
            T = np.asarray(T_K, dtype=float)
            logK = _LK25 - (_DH/(2.303*8.314))*(1.0/T - 1.0/298.15)
        K_MgCO3 = 10**(logK)*conv_mol**2
    
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
            
    # Source, in full, because the convention matters and a bare URL does not
    # record it: de Vries, W. & Posch, M. (2003), "Derivation of cation exchange
    # constants for sand, loess, clay and peat soils on the basis of field
    # measurements in the Netherlands", Alterra-rapport 701, Wageningen.
    # https://edepot.wur.nl/31605
    #
    # The values below are its 0-30 cm rows verbatim, for Ca-Mg / Ca-K / Ca-Na:
    #   sand  (Table 20-22)   0.557, -1.157, 0.745
    #   loess (what SMEW calls "loam")   0.108, -2.072, 0.378
    #   clay                  0.385, -2.428, 0.774
    # Several hundred Dutch non-agricultural soils, solution extracted by
    # centrifugation, pH 3-5, CEC normalised to a buffered pH of 6.5.
    #
    # THEY ARE CONCENTRATION-BASED, and the report says so on p.13: "Although
    # from a theoretical standpoint activities would be more appropriate, we
    # only consider exchange equations involving concentrations, since they are
    # mostly used in simple models." And on p.29: a model using activities
    # "would have to be corrected with the proper activity coefficients."
    # smew/activity.py does exactly that correction when gamma_on is set; do not
    # apply it twice.
    #
    # Their reported standard deviations are 0.35 to 0.79 log units -- factors
    # of 2 to 6 -- so these are constraints on a plausible range rather than
    # precise values, which is the report's own conclusion about them.
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
 # Weil, R. R. & Brady, N. C. (2016), The Nature and Properties of Soils, 15th ed.,
 # Pearson -- as cited by Bertagni et al. 2025 (JAMES), appendix "Aluminum
 # Speciation". A textbook, no DOI. Now only the legacy default of K_Al (T_K=None).

# Aluminium hydrolysis, CUMULATIVE reactions Al+3 + n H2O = Al(OH)n + n H+,
# as the analytic coefficients A1 + A2*T + A3/T + A4*log10(T). Taken verbatim
# from phreeqc.dat. Their source, checked line by line: WATEQ4F's Table 2,
# species 80-82 and 336 (Ball & Nordstrom 1991, USGS OFR 91-183,
# doi:10.3133/ofr91183), which cites them to Nordstrom et al. (1990), ACS Symp.
# Ser. 416, 398-413, doi:10.1021/bk-1990-0416.ch031. Nordstrom & May (1989/1996,
# The Environmental Chemistry of Aluminum, doi:10.1201/9780138736781-2) is the
# aluminium review behind that revision, not the table these numbers come from.
_AL_ANALYTIC = (
    (-38.253,  0.0,   -656.27,  14.327),   # AlOH+2
    ( 88.500,  0.0,  -9391.60, -27.121),   # Al(OH)2+
    (226.374,  0.0, -18247.80, -73.597),   # Al(OH)3
    ( 51.578,  0.0, -11168.90, -14.865),   # Al(OH)4-
)


def K_Al(conv_mol, T_K=None):
    """Aluminium hydrolysis constants, STEPWISE: K_n = [Al(OH)n][H+]/[Al(OH)n-1].

    T_K is the soil temperature [K]. Scalar or array, following K_C. Omitting it
    returns the fixed values this model has always used (Weil & Brady, The
    Nature and Properties of Soils, p.398), which is what keeps ic.py and the
    notebooks bit-identical.

    WHY TEMPERATURE MATTERS HERE. The fixed values are, to two decimals, exactly
    phreeqc.dat's at 25 C -- so they were never wrong, they were just frozen at
    a temperature most soils are nowhere near. Over 5-25 C the real constants
    move by up to 2.25 log units on the cumulative Al(OH)3 step. A model run at
    8 C was using 25 C chemistry for the one element whose speciation controls
    exchanger competition, toxicity, and gibbsite saturation.

    ADOPTING THIS SET ALSO MOVES TWO VALUES AT 25 C, and that is deliberate.
    Converted to stepwise, phreeqc.dat gives -5.00, -5.11, -6.83, -5.72 against
    this model's -5.00, -5.10, -6.70, -6.20. The first two agree; K3 differs by
    0.13 and K4 by 0.48 log units. Anchoring the temperature shape to the old
    25 C values instead would produce a ladder internally inconsistent with any
    published set, so the whole set is taken rather than spliced. Note the
    ladder is far from uniform in temperature sensitivity: over 5-35 C, K2 moves
    1.21 and K3 1.04 log units, while K4 moves only 0.14.
    """
    if T_K is None:
        pK1, pK2, pK3, pK4 = 5, 5.1, 6.7, 6.2
        K1 = 10**(-pK1)*conv_mol
        K2 = 10**(-pK2)*conv_mol
        K3 = 10**(-pK3)*conv_mol
        K4 = 10**(-pK4)*conv_mol
        return [K1, K2, K3, K4]

    T = np.asarray(T_K, dtype=float)
    # cumulative log betas, then differenced into the stepwise constants the
    # model's aluminium ladder is written in
    beta = [a[0] + a[1]*T + a[2]/T + a[3]*np.log10(T) for a in _AL_ANALYTIC]
    step = [beta[0], beta[1]-beta[0], beta[2]-beta[1], beta[3]-beta[2]]
    return [10**x * conv_mol for x in step]

#------------------------------------------------------------------------------
 # Aluminium hydroxide solubility [phreeqc.dat]

# Al(OH)3 + 3H+ = Al+3 + 3H2O: log K at 25 C and delta_h [kcal/mol], as
# phreeqc.dat carries them -- WATEQ4F Table 2 entries 51 (gibbsite, 8.11 +/- 0.2)
# and 140 (amorphous), both cited there to Nordstrom et al. 1990, the same source as the
# hydrolysis ladder above). The amorphous phase is what precipitates first and is
# ~10^2.7 more soluble; gibbsite is the stable, slow end.
_AL_OH3_PHASES = {
    "Al(OH)3(a)": (10.8, -26.5),
    "gibbsite": (8.11, -22.8),
}


def log_K_AlOH3(phase, T_K):
    """log10 K of Al(OH)3 + 3H+ = Al+3 + 3H2O, in ACTIVITIES and mol/l.

    van 't Hoff from phreeqc.dat's delta_h -- phreeqc.dat gives no analytic
    expression for either phase, so this is exactly what PHREEQC itself does.
    """
    if phase not in _AL_OH3_PHASES:
        raise ValueError(f"Unknown Al(OH)3 phase {phase!r}; "
                         f"choose from {sorted(_AL_OH3_PHASES)}")
    logK25, dH = _AL_OH3_PHASES[phase]
    R = 1.987204e-3                                       # [kcal/mol/K]
    return logK25 - dH / (R * np.log(10)) * (1 / T_K - 1 / 298.15)

#------------------------------------------------------------------------------
 # Secondary minerals [S1: phreeqc.dat; Palandri and Kharaka 2004]

# One table for every phase that can FORM in the soil (plan D5), instead of one
# `if mineral == ...` branch per phase. Each row:
#
#   st        what one mole releases on dissolving, over the seven columns of
#             min_st: [Ca, Mg, K, Na, Al, Si, Fe]
#   nu_H      H+ consumed per mole dissolved, as the phreeqc.dat reaction is
#             written. K = prod(a_i^st_i) / a_H^nu_H
#   log_K     phreeqc.dat. Either ("vh", logK25, dH [kcal/mol]) -- van 't Hoff,
#             what PHREEQC does when the database gives no analytic -- or
#             ("an", A1, A2, A3): log K = A1 + A2*T + A3/T, phreeqc.dat's own
#             analytic expression, which PHREEQC prefers when present
#   ppt       optional precipitation law, when growth is NOT the reverse of
#             dissolution: ("lnOmega", k_ref [mol/m2/s], T_ref [K], E [kJ/mol]),
#             R = k(T) A ln(Omega) for Omega > 1. Absent: the TST form below
#             runs both ways
#   kin       None: an EQUILIBRIUM phase. It relaxes to saturation in one step,
#             limited only by the solid present on the way down. Otherwise
#             Palandri and Kharaka (2004) mechanisms, each (log k25 [mol/m2/s],
#             E [kJ/mol], n) with k = k25 exp(-E/R (1/T - 1/298.15)) a_H^n:
#             "acid", "neutral" (n = 0), "base"
#   MM        molar mass [g/mol]
#   SSA       specific surface of the precipitate [m2/g] -- growth on what has
#             formed. ASSUMED, not measured; see A_seed
#   A_seed    surface available before any of the phase exists [m2 per m3 of
#             soil]: the nucleation floor. THE tuning knob (plan D2), defaulted,
#             to be swept rather than believed
#
# Al(OH)3(am) has no defensible rate constant -- it is the phase that forms first
# and fastest -- so it is an equilibrium phase. That is a modelling judgement,
# and the MOMBAK porewater (S0.5, at -1.1 SI) is the evidence for the ceiling,
# not for a rate.
#
# Gibbsite and amorphous silica use P&K's dissolution parameters in the TST
# form R = k A (1 - Omega) for BOTH directions, i.e. precipitation as the
# reverse of dissolution (detailed balance). For gibbsite near equilibrium that
# is the linear limit of Nagy and Lasaga (1992); far from it, precipitation is
# nonlinear in Omega and this understates nothing the limiter does not cap.
_SEC_PHASES = {
    "Al(OH)3(a)": dict(
        st=(0, 0, 0, 0, 1, 0, 0), nu_H=3,
        log_K=("vh", 10.8, -26.5),       # phreeqc.dat; WATEQ4F #140, Nordstrom+ 1990
        kin=None, MM=78.004, SSA=0.0, A_seed=0.0),
    "gibbsite": dict(
        st=(0, 0, 0, 0, 1, 0, 0), nu_H=3,
        log_K=("vh", 8.11, -22.8),       # phreeqc.dat; WATEQ4F #51, Nordstrom+ 1990
        # P&K 2004 (doi:10.3133/ofr20041068), oxides table, selected there
        # from Nagy and Lasaga 1992 (doi:10.1016/0016-7037(92)90291-P) and Packter
        # and Dhillon 1974, Colloid Polym. Sci. 252, 249-256 (doi:10.1007/BF01638106)
        kin={"acid": (-7.65, 47.5, 0.992), "neutral": (-11.50, 61.2, 0.0),
             "base": (-16.65, 80.1, -0.784)},
        MM=78.004, SSA=10.0, A_seed=1.0e3),
    "SiO2(a)": dict(
        st=(0, 0, 0, 0, 0, 1, 0), nu_H=0,
        log_K=("an", -0.26, 0.0, -731.0),  # phreeqc.dat; WATEQ4F #395 (-2.71 at 25 C)
        # P&K 2004 Table 6, amorphous silica, from Icenhower and Dove 2000
        # (doi:10.1016/S0016-7037(00)00487-7). Consistent with Rimstidt and
        # Barnes 1980 (doi:10.1016/0016-7037(80)90220-3): their precipitation
        # constant is this one divided by K
        kin={"neutral": (-12.23, 74.5, 0.0)},
        MM=60.084, SSA=100.0, A_seed=1.0e3),
    # S2. Takes Al AND Si: the first two-element phase (biogeochem._sec_extent).
    # log K: phreeqc.dat = WATEQ4F #46 (7.435, dH -35.3 kcal), Nordstrom et al.
    # 1990. Yang & Steefel 2008 (GCA 72, 99-116, doi:10.1016/j.gca.2007.10.011)
    # measured 10^7.57 at 22 C; this gives 7.70 there.
    # kin: P&K 2004 Table 29 (clay group), regressed there from Carroll & Walther
    # 1990, Ganor et al. 1995, Huertas et al. 1999a,b, Nagy et al. 1991, Soong
    # 1993. At 22 C, pH 4 it gives ~10^-13.2 mol/m2/s, against Yang & Steefel's
    # measured TST dissolution constant 1.15e-13.
    # PRECIPITATION is not the reverse of that. Nagy et al. 1991 (Am. J. Sci.
    # 291, 649-686, doi:10.2475/ajs.291.7.649) found it reversible at 80 C on
    # pre-activated seed; Yang & Steefel, at 22 C, only for ~110 h, after which
    # long-term growth followed their eq. 9, R_ppt = 8.0e-15 (dG/RT) mol/m2/s,
    # linear in ln(Omega), fitted over dG = 4.5-14.8 kJ/mol at pH ~4 (full text,
    # OSTI author manuscript 923297). At the Omega ~ 1e5 a soil reaches, the TST
    # form would be ~1e5 times faster than that, so it is eq. 9 that is used.
    # Its temperature dependence was not measured: P&K's neutral-mechanism E
    # (22.2 kJ/mol) is ASSUMED, from Yang & Steefel's 22 C.
    "kaolinite": dict(
        st=(0, 0, 0, 0, 2, 2, 0), nu_H=6,
        log_K=("vh", 7.435, -35.3),      # phreeqc.dat; WATEQ4F #46, Nordstrom+ 1990
        kin={"acid": (-11.31, 65.9, 0.777), "neutral": (-13.18, 22.2, 0.0),
             "base": (-17.05, 17.9, -0.472)},
        ppt=("lnOmega", 8.0e-15, 295.15, 22.2),
        # SSA: ASSUMED, a typical soil kaolinite order of magnitude
        MM=258.16, SSA=15.0, A_seed=1.0e3),
}


def sec_phase(name):
    """The _SEC_PHASES row for `name`, or a ValueError naming the choices."""
    if name not in _SEC_PHASES:
        raise ValueError(f"Unknown secondary phase {name!r}; "
                         f"choose from {sorted(_SEC_PHASES)}")
    return _SEC_PHASES[name]


def sec_log_K(name, T_K):
    """log10 K of a secondary phase's dissolution reaction, in activities."""
    lk = sec_phase(name)["log_K"]
    if lk[0] == "an":
        return lk[1] + lk[2] * T_K + lk[3] / T_K
    R = 1.987204e-3                                       # [kcal/mol/K]
    return lk[1] - lk[2] / (R * np.log(10)) * (1 / T_K - 1 / 298.15)


def sec_k_ppt(name, T_K):
    """Precipitation rate constant [mol/m2/s] of a phase with its own growth law
    (the `ppt` row), at T_K; None when growth is the reverse of dissolution."""
    ppt = sec_phase(name).get("ppt")
    if ppt is None:
        return None
    _, k_ref, T_ref, E = ppt
    R = 8.314462618e-3                                    # [kJ/mol/K]
    return k_ref * np.exp(-E / R * (1 / T_K - 1 / T_ref))


def sec_k(name, T_K, a_H):
    """Rate constant [mol/m2/s] at T_K and H+ activity a_H [mol/l].

    inf for an equilibrium phase. Each mechanism is Arrhenius from 25 C, as
    Palandri and Kharaka write it (their eqns 5-6), and they add.
    """
    kin = sec_phase(name)["kin"]
    if kin is None:
        return np.inf
    R = 8.314462618e-3                                    # [kJ/mol/K]
    k = 0.0
    for logk25, E, n_H in kin.values():
        k += 10 ** logk25 * np.exp(-E / R * (1 / T_K - 1 / 298.15)) * a_H ** n_H
    return k

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