import numpy as np
import smew

def mineral_weathering(mineral, Tk, Omega, H, Al, conv_mol, conv_Al):

    '''
    Docstring for min_const
    
    :param mineral: name of the mineral
    :param Tk: temperature in Kelvin
    :param conv_mol: conversion factor
    :param Omega: saturation index
    :param H: H+ concentration
    :param Al: Al concentration

    Important: Currently working with concentrations, not activities!
    '''

    # All values are from KINEC dataset

    R = 8.314/conv_mol # [J mol-1 K-1]: universal gas constant

    # Unit conversion
    H = H/conv_mol
    Al = Al/(conv_mol*conv_Al)

    if mineral == 'albite': #NaAlSi3O8; M 262.219 g/mol
        Aa = 0.7*24*3600*conv_mol	#mol.m-2.s-1
        An = 2.05*10**(-1)*24*3600*conv_mol #mol.m-2.s-1 
        Ab = 1.5*10**(-5)*24*3600*conv_mol #mol.m-2.s-1
        Ea = 58000/conv_mol  #J.mol-1
        En = 60000/conv_mol	#J.mol-1
        Eb = 50000/conv_mol	#J.mol-1
        na = 0.3
        nb = -0.3
        Sig = 3
		#rate equations
        rplusa = Aa* (np.exp(-Ea/ (R * Tk)))*(H**na )
        rplusn = An* (np.exp(-En/ (R * Tk)))
        rplusb = Ab* (np.exp(-Eb/ (R * Tk)))* (H**nb)
        rplus = rplusa + rplusn + rplusb
        rate = rplus * (1 - (Omega**(1/Sig)))

    elif mineral == 'anorthite': #CaAl2Si2O8; M 278.204 g/mol
        Aa = 9.82*10**(4)*24*3600*conv_mol	#mol.m-2.s-1
        An = 1.5*10**(-1)*24*3600*conv_mol	#mol.m-2.s-1 
        Ab = 1.5*10**(-5)*24*3600*conv_mol #mol.m-2.s-1
        Ea = 58000/conv_mol  #J.mol-1
        En = 60000/conv_mol	#J.mol-1
        Eb = 50000/conv_mol	#J.mol-1
        na = 1.22
        nb = -0.35
        Sig = 2
		#rate equations
        rplusa = Aa* (np.exp(-Ea/ (R * Tk)))*(H**na )
        rplusn = An* (np.exp(-En/ (R * Tk)))
        rplusb = Ab* (np.exp(-Eb/ (R * Tk)))* (H**nb)
        rplus = rplusa + rplusn + rplusb
        rate = rplus * (1 - (Omega**(1/Sig)))

    elif mineral == 'augite': #Mg0.45Fe0.275Ca0.275SiO3;M 113.4  g/mol
        Aa =1.52*10**(6)*24*3600*conv_mol
        An =350*24*3600*conv_mol
        Ea =81834/conv_mol
        En =83000/conv_mol
        R =  8.314/conv_mol
        Sig = 1
        na =0.7
        rplusa = Aa* (np.exp(-Ea/ (R * Tk)))*(H**na )
        rplusn = An* (np.exp(-En/ (R * Tk)))
        rplus = rplusa + rplusn
        rate = rplus * (1 - (Omega**(1/Sig)))


    elif mineral == 'basalt_glass': #SiTi0.02Al0.36Fe0.19Mg0.28Ca0.26Na0.08K0.008O3.364 , M 122.566 g/mol
        Aa = 1.08*10**(-4)*24*3600*conv_mol	#mol.m-2.s-1
        Ea = 21500/conv_mol	#J.mol-1
        ACTI = (H**3)/Al
        n = 1/3
        Sig = 1
		#rate equation
        rplus = Aa * ACTI**n  * np.exp(-Ea/ (R * Tk))
        rate = rplus * (1 - (Omega**(1/Sig)))
        
    elif mineral == 'chabazite_Ca': # CaAl2Si4O12:6H2O
        # CaAl2Si4O12:6H2O + 8 H+ = 2 Al+3 + Ca+2 + 4 SiO2 + 10 H2O
        #rho_min = 2.8*1e6 # [g/m3]: density
        Aa = 0.221*24*3600*conv_mol #mol.m-2.s-1
        An = 1.56*10**(-4)*24*3600*conv_mol	#mol.m-2.s-1 
        Ab = 4.94*10**(-5)*24*3600*conv_mol #mol.m-2.s-1
        Ea = 33700/conv_mol #J.mol-1
        En = 44200/conv_mol	#J.mol-1
        Eb = 44200/conv_mol	#J.mol-1
        na = 0.82
        nb = -0.2
        Sig = 4    
        #rate equations
        rplusa = Aa* (np.exp(-Ea/ (R * Tk)))*(H**na )
        rplusn = An* (np.exp(-En/ (R * Tk)))
        rplusb = Ab* (np.exp(-Eb/ (R * Tk)))*(H**nb )
        rplus = rplusa + rplusn +rplusb
        rate = rplus * (1 - (Omega**(1/Sig)))

    elif mineral == 'clinoptilolite_Ca':
        Aa = 2.48*10**(-2)*24*3600*conv_mol #mol.m-2.s-1
        An = 1.39*10**(-5)*24*3600*conv_mol	#mol.m-2.s-1 
        Ab = 3.5*10**(-6)*24*3600*conv_mol #mol.m-2.s-1
        Ea = 33700/conv_mol #J.mol-1
        En = 44200/conv_mol	#J.mol-1
        Eb = 44200/conv_mol	#J.mol-1
        na = 0.82
        nb = -0.2
        Sig = 15
		#rate equations
        rplusa = Aa* (np.exp(-Ea/ (R * Tk)))*(H**na )
        rplusn = An* (np.exp(-En/ (R * Tk)))
        rplusb = Ab* (np.exp(-Eb/ (R * Tk)))*(H**nb )
        rplus = rplusa + rplusn +rplusb
        rate = rplus * (1 - (Omega/Sig))

    elif mineral == 'diopside': #CaMgSi2O6; M 216.55 g/mol
        Aa = 8.55*10**(-5)*24*3600*conv_mol #mol.m-2.s-1
        An = 4.30*10**(-4)*24*3600*conv_mol #mol.m-2.s-1
        Ea = 32654/conv_mol #J.mol-1
        En = 43866/conv_mol #J.mol-1
        ACTI = H
        na = 0.25
        Sig = 2
		#rate equations
        rplusa = Aa* (np.exp(-Ea/ (R * Tk)))*(ACTI**na )
        rplusn = An* (np.exp(-En/ (R * Tk)))
        rplus = rplusa + rplusn
        rate = rplus * (1 - (Omega**(1/Sig)))

    elif mineral == 'forsterite': #Mg2SiO4, M 140.692 g/mol
        Aa =14.8*10**4*24*3600*conv_mol # mol.m-2.s-1
        Ab =220*24*3600*conv_mol # mol.m-2.s-1
        Ea =70400/conv_mol # J/mol
        Eb =60900/conv_mol # J/mol
        Sig = 1   
        na = 0.44
        nb = 0.22
		#Rate Equation
        rplusa = Aa* (np.exp(-Ea/ (R * Tk)))*(H**na)
        rplusb = Ab* (np.exp(-Eb/ (R * Tk)))* (H**nb)
        rplus = rplusa + rplusb
        rate = rplus * (1 - Omega**(1/Sig))

    #elif mineral == 'Fe_forsterite':

    elif mineral == 'heulandite_Ca': # CaAl2Si7O18:6H2O # or 'heulandite_Na': # Na2Al2Si7O18:5H2O
        Aa = 2.48*10**(-2)*24*3600*conv_mol #mol.m-2.s-1
        An = 1.39*10**(-5)*24*3600*conv_mol	#mol.m-2.s-1 
        Ab = 3.5*10**(-6)*24*3600*conv_mol #mol.m-2.s-1
        Ea = 33700/conv_mol #J.mol-1
        En = 44200/conv_mol	#J.mol-1
        Eb = 44200/conv_mol	#J.mol-1
        na = 0.82
        nb = -0.2
        Sig = 7
		#rate equations
        rplusa = Aa* (np.exp(-Ea/ (R * Tk)))*(H**na )
        rplusn = An* (np.exp(-En/ (R * Tk)))
        rplusb = Ab* (np.exp(-Eb/ (R * Tk)))*(H**nb )
        rplus = rplusa + rplusn +rplusb
        rate = rplus * (1 - (Omega**(1/Sig)))

    elif mineral == 'hydroxyapatite': #Ca5(OH)(PO4)3 ; M 502.31 g/mol
        Aa = 80*24*3600*conv_mol #mol.m-2.s-1
        Ab = 3*10**(-2)*24*3600*conv_mol #mol.m-2.s-1
        Ea = 43000/conv_mol    #J.mol-1
        Eb = 43000/conv_mol #J.mol-1
        ACTI = H
        na = 0.8
        nb = 0.2
        Sig = 5   
		#rate equation
        rplusa = Aa * ACTI**na  * np.exp(-Ea/ (R * Tk))
        rplusb = Ab * ACTI**nb  * np.exp(-Eb/ (R * Tk))
        rplus = rplusa + rplusb
        rate = rplus * (1 - (Omega**(1/Sig)))

    elif mineral == 'K_feldspar': #KAlSi3O8; M 278.33 g/mol
        Aa = 0.05*24*3600*conv_mol	# mol.m-2.s-1 
        An = 1.08e-2*24*3600*conv_mol	# mol.m-2.s-1
        Ab = 1.2e-10*24*3600*conv_mol	# mol.m-2.s-1
        Ea = 51700/conv_mol   # J/mol
        En = 60000/conv_mol   # J/mol
        Eb = 62195/conv_mol   # J/mol 
        ACTI = H #
        Sig = 3
        nA = 0.45
        nb = -0.75
		#Rate Equation
        rplusa = Aa * ACTI**nA * np.exp (-Ea/ (R * Tk))
        rplusn = An * np.exp (-En/ (R * Tk))
        #rplusb = Ab* (np.exp(-Eb/ (R * Tk)))* ACTI**(nC) * S #!!! nC not specified in database!!!
        rplusb = Ab* (np.exp(-Eb/ (R * Tk)))* ACTI**(nb)
        rplus = rplusa + rplusn + rplusb
        rate = rplus * (1 - Omega**(1/Sig))

    elif mineral == 'labradorite': # Ca0.68Na0.32Al1.68Si2.32O8, M 245.84 g/mol
        Aa = 5886.557*24*3600*conv_mol #mol.m-2.s-1
        An = 0.17*24*3600*conv_mol	#mol.m-2.s-1 
        Ab = 1.5*10**(-5)*24*3600*conv_mol #mol.m-2.s-1
        Ea = 58000/conv_mol  #J.mol-1
        En = 60000/conv_mol	#J.mol-1
        Eb = 50000/conv_mol	#J.mol-1
        na = 1.0
        nb = -0.35
        Sig = 2.32
		#rate equations
        rplusa = Aa* (np.exp(-Ea/ (R * Tk)))*(H**na )
        rplusn = An* (np.exp(-En/ (R * Tk)))
        rplusb = Ab* (np.exp(-Eb/ (R * Tk)))* (H**nb)
        rplus = rplusa + rplusn + rplusb
        #SR_Labradorite=(SR ("Albite")*0.32)*(SR ("Anorthite")*0.68)
        rate = rplus * (1 - (Omega**(1/Sig)))

    #elif mineral == 'leucite': ## Not in database!!

    elif mineral == 'muscovite':
        Aa = 0.000126*24*3600*conv_mol #mol.m-2.s-1  
        An = 0.00000631*24*3600*conv_mol #mol.m-2.s-1
        Ab = 0.0000316*24*3600*conv_mol #mol.m-2.s-1 
        Ea = 41311/conv_mol    #J.mol-1  
        En = 39301/conv_mol    #J.mol-1 
        Eb = 56950/conv_mol    #J.mol-1 
        nA = 0.37
        nb = -0.22
        Sig = 3
		#rate equations
        rplusa = Aa* (np.exp(-Ea/ (R * Tk)))*(H**nA )
        rplusn = An* (np.exp(-En/ (R * Tk)))
        rplusb = Ab* (np.exp(-Eb/ (R * Tk)))*(H**nb)
        rplus = rplusa + rplusn + rplusb
        rate = rplus * (1 - (Omega**(1/Sig)))

    elif mineral == 'nepheline': #NaAlSiO4
        Aa = 5*10**7*24*3600*conv_mol #mol.m-2.s-1
        An = 0.1*24*3600*conv_mol	#mol.m-2.s-1 
        Ab = 7.5*10**(-5)*24*3600*conv_mol #mol.m-2.s-1
        Ea = 63000/conv_mol #J.mol-1
        En = 58500/conv_mol	#J.mol-1
        Eb = 58000/conv_mol	#J.mol-1
        na = 1.0
        nb = -0.4
        Sig = 1
		#rate equations
        rplusa = Aa* (np.exp(-Ea/ (R * Tk)))*(H**na )
        rplusn = An* (np.exp(-En/ (R * Tk)))
        rplusb = Ab* (np.exp(-Eb/ (R * Tk)))*(H**nb )
        rplus = rplusa + rplusn +rplusb
        rate = rplus * (1 - (Omega**(1/Sig)))

    elif mineral == 'olivine': #Mg1.8Fe0.2SiO4;M   147.31 g/mol  
        Aa  =14.8*10**(4)*24*3600*conv_mol # mol.m-2.s-1 # Forsterite rate!
        Ab  =220*24*3600*conv_mol # mol.m-2.s-1  # Forsterite rate!
        Ea  =70400/conv_mol # J/mol
        Eb  =60900/conv_mol # J/mol
        Sig = 1   
        na = 0.44
        nb = 0.22
        #Rate Equation
        rplusa = Aa* (np.exp(-Ea/ (R * Tk)))*(H**na)
        rplusb = Ab* (np.exp(-Eb/ (R * Tk)))* (H**nb)
        rplus = rplusa + rplusb
        #SR_Olivine=(SR ("Forsterite")*0.9)*(SR ("Fayalite")*0.1)
        rate = rplus * (1 - (Omega**(1/Sig)))

    elif mineral == 'wollastonite': #CaSiO3;M 117.1 g/mol
        Aa = 700*24*3600*conv_mol #mol.m-2.s-1 
        Ab = 20*24*3600*conv_mol #mol.m-2.s-1  
        Ea = 56000/conv_mol #J.mol-1
        Eb = 52000/conv_mol #J.mol-1
        ACTI = H
        na = 0.4
        nb = 0.15
        Sig = 1
		#rate equations 
        rplusa = Aa* (np.exp(-Ea/ (R * Tk)))*(ACTI**na )
        rplusb = Ab* (np.exp(-Eb/ (R * Tk)))*(ACTI**nb )
        rplus = rplusa + rplusb
        rate = rplus * (1 - (Omega**(1/Sig)))

    else:
        raise ValueError("No data for this mineral")
  
    # Output: weathering in [mol-conv/d]
    # moles = rate * time

    return(rate)

#------------------------------------------------------------------------------
 # Carbonate weathering [mol-conv/d]
 #In soil,precipitates form as discontinuous coatings on the surfaces of soil pores, so the precipitation surface area and geometry are indeterminate. https://nora.nerc.ac.uk/id/eprint/511084/1/Kirk%20et%20al%202015%20Geochmica%20et%20Cosmochimica%20Acta.pdf
   
def carb_W(CaCO3, MgCO3, Omega_CaCO3, Omega_MgCO3, s, Zr, r_CaCO3, r_MgCO3, tau_CaCO3, tau_MgCO3):
        
    #CaCO3
    if Omega_CaCO3 <= 1:
        W_CaCO3 = s*CaCO3*(1-Omega_CaCO3)/tau_CaCO3 # dissolution
    else:
        W_CaCO3 = r_CaCO3*Zr*(1-Omega_CaCO3)        # precipitation 
    
    #MgCO3
    if Omega_MgCO3 <= 1:
        W_MgCO3 = s*MgCO3*(1-Omega_MgCO3)/tau_MgCO3 # dissolution
    else:
        W_MgCO3 = r_MgCO3*Zr*(1-Omega_MgCO3)        # precipitation
                                      
    return (W_CaCO3, W_MgCO3)


#------------------------------------------------------------------------------
# Silicate saturation index (Omega)
#------------------------------------------------------------------------------
    
def Omega_sil(mineral, Ca, Mg, K, Na, Si, H, Al, Fe, K_sp, conv_mol, conv_Al):

    Al = Al/conv_Al

    if mineral == 'albite': #NaAlSi3O8; M 262.219 g/mol
        # NaAlSi3O8 + 4 H+ = Al+3 + Na+ + 2 H2O + 3 SiO2
        Omega = min(1,(((Na/conv_mol)*(Al/conv_mol)*(Si/conv_mol)**3)/(H/conv_mol)**4) / 10**2.7645)

    elif mineral == 'anorthite': #CaAl2Si2O8; M 278.204 g/mol
        # CaAl2(SiO4)2 + 8 H+ = Ca+2 + 2 Al+3 + 2 SiO2 + 4 H2O
        Omega = min(1,(((Ca/conv_mol)*(Al/conv_mol)**2*(Si/conv_mol)**2)/(H/conv_mol)**8) / 10**26.578)

    elif mineral == 'augite': #Mg0.45Fe0.275Ca0.275SiO3;M 113.4  g/mol
        # (SR ("Wollastonite")*0.45)*(SR ("Ferrosilite")*0.275)*(SR ("Enstatite")*0.275)
        Wollastonite45 = ((((Ca/conv_mol)*(Si/conv_mol))/(H/conv_mol)**2)/10**13.7605)*0.45
        #Ferrosilite275 =  ((((Fe/conv_mol)*(Si/conv_mol))/(H/conv_mol)**2)/10**7.4471)*0.275
        Ferrosilite275 =  (((1*(Si/conv_mol))/(H/conv_mol)**2)/10**7.4471)*0.275 # Removed Fe for now because it is not modeled
        Enstatite275 =  ((((Mg/conv_mol)*(Si/conv_mol))/(H/conv_mol)**2)/10**11.3269)*0.275
        Omega = min(1, (Wollastonite45 * Ferrosilite275 * Enstatite275))

    elif mineral == 'basalt_glass': #SiTi0.02Al0.36Fe0.19Mg0.28Ca0.26Na0.08K0.008O3.364 , M 122.566 g/mol
        # Si1.00Al0.35O2(OH)1.05 + 0.35 OH- = 0.35 Al(OH)4- + SiO2
        Omega = min(1,((Al/conv_mol)**0.35*(Si/conv_mol))/(H/conv_mol)**0.35/10**(-2.36449))

    elif mineral == 'chabazite_Ca': # CaAl2Si4O12:6H2O
        # CaAl2Si4O12:6H2O + 8 H+ = 2 Al+3 + Ca+2 + 4 SiO2 + 10 H2O
        Omega = min(1,((Al/conv_mol)**2*(Si/conv_mol)**4*(Ca/conv_mol))/(H/conv_mol)**8/10**(14.7771))

    elif mineral == 'chabazite_Na':
        # Na2Al2Si4O12:6H2O + 8 H+ = 2 Al+3 + 2 Na+ + 4 SiO2 + 10 H2O
        Omega = min(1,((Al/conv_mol)**2*(Si/conv_mol)**4*(Na/conv_mol)**2)/(H/conv_mol)**8/10**(16.9077))

    elif mineral == 'clinoptilolite_Ca':
        # Ca1.5Al3Si15O36:12H2O + 12 H+ = 3 Al+3 + 1.5 Ca+2 + 15 SiO2 + 18 H2O
        Omega = min(1,(((Ca/conv_mol)**1.5*(Al/conv_mol)**3*(Si/conv_mol)**15)/(H/conv_mol)**12)/10**(-6.46186))

    elif mineral == 'clinoptilolite_Na':
        # Na3Al3Si15O36:10H2O + 12 H+ = 3 Al+3 + 3 Na+ + 15 SiO2 + 16 H2O
        Omega = min(1,(((Na/conv_mol)**3*(Al/conv_mol)**3*(Si/conv_mol)**15)/(H/conv_mol)**12)/10**(-9.10501))

    elif mineral == 'diopside': #CaMgSi2O6; M 216.55 g/mol
        # CaMgSi2O6 + 4 H+ = Ca+2 + Mg+2 + 2 H2O + 2 SiO2
        Omega = min(1,(((Ca/conv_mol)*(Mg/conv_mol)*(Si/conv_mol)**2)/(H/conv_mol)**4)/10**20.9643)

    elif mineral == 'forsterite': #Mg2SiO4, M 140.692 g/mol
        # Mg2SiO4 + 4 H+ =  SiO2 + 2 H2O + 2 Mg+2
        Omega = min(1,(((Mg/conv_mol)**2*(Si/conv_mol))/((H/conv_mol)**4))/10**27.8626)

    #elif mineral == 'Fe_forsterite':
    #    Forsterite50 = ((((Mg/conv_mol)**2*(Si/conv_mol))/((H/conv_mol)**4))/10**27.8626) * 0.5
    #    #Fayalite50 =  ((((Fe/conv_mol)**2*(Si/conv_mol))/((H/conv_mol)**4))/10**19.113)*0.5
    #    Fayalite50 =  (((Si/conv_mol)/((H/conv_mol)**4))/10**19.113)*0.5
    #    Omega = min(1, (Forsterite50 * Fayalite50))

    elif mineral == 'heulandite_Ca': # CaAl2Si7O18:6H2O # or 'heulandite_Na': # Na2Al2Si7O18:5H2O
        # CaAl2Si7O18:6H2O + 8 H+ = 2 Al+3 + Ca+2 + 7 SiO2 + 10 H2O
        Omega = min(1,(((Al/conv_mol)**2*(Ca/conv_mol)*(Si/conv_mol)**7)/((H/conv_mol)**8))/10**3.436)

    elif mineral == 'heulandite_Na':
        # Na2Al2Si7O18:5H2O + 8 H+ = 2 Al+3 + 2 Na+ + 7 SiO2 + 9 H2O
        Omega = min(1,(((Al/conv_mol)**2*(Na/conv_mol)**2*(Si/conv_mol)**7)/((H/conv_mol)**8))/10**6.5703)

    elif mineral == 'hydroxyapatite': #Ca5(OH)(PO4)3 ; M 502.31 g/mol
        # Ca5(OH)(PO4)3 + 4 H+ = H2O + 3 HPO4-2 + 5 Ca+2
        Omega = min(1,(((Ca/conv_mol)**5)/((H/conv_mol)**4))/10**(-3.0746))

    elif mineral == 'K_feldspar': #KAlSi3O8; M 278.33 g/mol
        # KAlSi3O8 + 4 H+ = Al+3 + K+ + 2 H2O + 3 SiO2
        Omega =  min(1, (((K/conv_mol)*(Al/conv_mol)*(Si/conv_mol)**3)/(H/conv_mol)**4)/10**(-0.2753))

    elif mineral == 'labradorite': # Ca0.68Na0.32Al1.68Si2.32O8, M 245.84 g/mol
        # SR_Labradorite=(SR ("Albite")*0.32)*(SR ("Anorthite")*0.68)
        Anorthite68 = ((((Ca/conv_mol)*(Al/conv_mol)**2*(Si/conv_mol)**2)/(H/conv_mol)**8) / 10**26.578)*0.68 # CaAl2(SiO4)2 + 8 H+ = Ca+2 + 2 Al+3 + 2 SiO2 + 4 H2O
        Albite32 =  ((((Na/conv_mol)*(Al/conv_mol)*(Si/conv_mol)**3)/(H/conv_mol)**4) / 10**2.7645)*0.32 # NaAlSi3O8 + 4 H+ = Al+3 + Na+ + 2 H2O + 3 SiO2
        Omega = min(1, (Anorthite68 * Albite32))

    elif mineral == 'leucite': # From Bertagni et al. (2025), original code
        Omega =  min(1, (((K/conv_mol)*(Al/conv_mol)*(Si/conv_mol))/(H/conv_mol)**4)/10**10.8987)  ### Equilbrium constant for Kalsilite used 

    elif mineral == 'muscovite':
        # KAl3Si3O10(OH)2 + 10 H+ = K+ + 3 Al+3 + 3 SiO2 + 6 H2O
        Omega =  min(1, (((K/conv_mol)*(Al/conv_mol)**3*(Si/conv_mol)**3)/(H/conv_mol)**10)/10**13.5858)

    elif mineral == 'nepheline': #NaAlSiO4
        # NaAlSiO4 + 4 H+ = Al+3 + Na+ + SiO2 + 2 H2O
        Omega =  min(1, (((Na/conv_mol)*(Al/conv_mol)*(Si/conv_mol))/(H/conv_mol)**4)/10**13.8006)
        #Nepheline75 = ((((Na/conv_mol)*(Al/conv_mol)*(Si/conv_mol))/(H/conv_mol)**4) / 10**13.8006)*0.75
        #Kalsilite25 = ((((K/conv_mol)*(Al/conv_mol)*(Si/conv_mol))/(H/conv_mol)**4) / 10**10.8987)*0.25
        #Omega = min(1, (Nepheline75 * Kalsilite25))

    elif mineral == 'olivine': #Mg1.8Fe0.2SiO4;M   147.31 g/mol  
        # SR_Olivine=(SR ("Forsterite")*0.9)*(SR ("Fayalite")*0.1)
        Forsterite90 = ((((Mg/conv_mol)**2*(Si/conv_mol))/((H/conv_mol)**4))/10**27.8626) * 0.9
        #Fayalite10 =  ((((Fe/conv_mol)**2*(Si/conv_mol))/((H/conv_mol)**4))/10**19.113)*0.1 # Fe2SiO4 + 4 H+ = SiO2 + 2 Fe+2 + 2 H2O
        Fayalite10 =  (((Si/conv_mol)/((H/conv_mol)**4))/10**19.113)*0.1
        Omega = min(1, (Forsterite90 * Fayalite10))

    elif mineral == 'wollastonite': #CaSiO3;M 117.1 g/mol
        # CaSiO3 + 2 H+ = Ca+2 + H2O + SiO2
        Omega = min(1,(((Ca/conv_mol)*(Si/conv_mol))/(H/conv_mol)**2)/10**13.7605)

    elif mineral == 'alkali_feldspar':
        Anorthite03 = ((((Ca/conv_mol)*(Al/conv_mol)**2*(Si/conv_mol))/(H/conv_mol)**8) / 10**26.578)*0.03
        Albite65 =  ((((Na/conv_mol)*(Al/conv_mol)*(Si/conv_mol)**3)/(H/conv_mol)**4) / 10**2.7645)*0.65
        Sanidine41 =  ((((K/conv_mol)*(Al/conv_mol)*(Si/conv_mol)**3)/(H/conv_mol)**4) / 10**0.9239)*0.41
        Omega = min(1, (Anorthite03 * Albite65 * Sanidine41))


    #elif mineral in ['apatite','muscovite', 'chabazite', 'heulandite', 'clinoptilolite']:
    #    Omega = 0
    
    else:
        raise ValueError("Unknown mineral")
        
    return Omega