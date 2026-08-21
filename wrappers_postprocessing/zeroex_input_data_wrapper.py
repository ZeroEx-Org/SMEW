import sys, pathlib

import os
import time; start_time = time.time()
import importlib as imp
import smew
#import figEW

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib import rc
import numpy as np
import warnings

# import dill
import pandas as pd
import json

from types import SimpleNamespace

def soil_texture_classifier(sand, clay):
    
    """
    Input parameters: typle or list of Sand, Silt, Clay - %
    
    Output - soil type
    
    """
    
    silt = 100 - sand - clay
    #     sand,silt,clay = list(soil_properties)
    assert (type(sand)==float)or(type(sand)==int), 'Houston, we have problems: Sand, Silt, Clay should be float or int (54,4% or 44%)'
    #     assert np.round(silt+clay+sand, decimals=1) == 100, 'Houston, we have problems: Sand, Silt, Clay should be bigger 0 and less 100'
    soil_type='random soil'
    if silt+(1.5*clay)<15:
        soil_type='sand'
    elif silt+(1.5*clay)>=15 and silt+2*clay<30:
        soil_type='loamy sand'
    elif (20>clay>=7 and sand>52 and silt+2*clay>=30) or \
         (clay<7 and silt<50 and silt + 2*clay > 30):
        soil_type='sandy loam'
    elif 7<=clay<27 and 28<=silt<50 and sand<=52:
        soil_type = 'loam'
    elif (50<=silt and 12<=clay<27) or (50<=silt<80 and clay<12):
        soil_type = 'silt loam'
    elif 80<=silt and clay<12:
        soil_type = 'silt'
    elif 20<=clay<=35 and silt<28 and sand>45:
        soil_type = 'sandy clay loam'
    elif 27<=clay<40 and 20<sand<=45:
        soil_type = 'clay loam'
    elif 27<=clay<40 and sand<=20:
        soil_type = 'silty clay loam'
    elif 35<=clay and 45<sand:
        soil_type = 'sandy clay'
    elif 40<=clay and 40<=silt:
        soil_type = 'silty clay'
    elif 40<=clay and sand<=45 and silt<40:
        soil_type = 'clay'
    else:
        soil_type = 'sandy loam'
    
    return soil_type

def read_input_data(project_name, value_col):
    print('--------------------------------------------------------------------------------------')
    print('Reading input files for',project_name,'...')

    input_data = {}

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    input_file_name = repo_root / 'Projects' / project_name / f'Data_Sheet_{project_name}.xlsx'

    # Read single value inputs
    climate_inputs = pd.read_excel(input_file_name,sheet_name='Climatic',usecols=['Parameter',value_col])
    soil_inputs = pd.read_excel(input_file_name,sheet_name='Soil',usecols=['Parameter',value_col])
    feedstock_inputs = pd.read_excel(input_file_name,sheet_name='Feedstock',usecols=['Parameter',value_col])
    plant_inputs = pd.read_excel(input_file_name,sheet_name='Plant',usecols=['Parameter',value_col])
    # Add all values into list
    for df in [climate_inputs,soil_inputs,feedstock_inputs,plant_inputs]:
        new_dict = dict(zip(df['Parameter'], df[value_col]))
        input_data = input_data | new_dict

    # Read multi value inputs
    # Stochastic rain
    rain = pd.read_excel(input_file_name,sheet_name='Rain')
    input_data['lambda_rain'] = np.array(rain['lambda (1/d)'].values)
    input_data['alfa_rain'] = np.array(rain['alfa (mm)'].values)

    # mineralogy
    mineralogy = pd.read_excel(input_file_name,sheet_name='Mineralogy')
    input_data['mineral'] = mineralogy['Mineral (%)'].values
    input_data['rock_f_in'] = np.array(mineralogy[value_col].values) / 100

    # particle size distribution
    psd = pd.read_excel(input_file_name,sheet_name='GrainSizeDistribution')
    input_data['d_in'] = psd['Diameter (μm)'].values*1e-6 # [m]
    input_data['psd_perc_in'] = psd['dist_percent'].values

    # Get texture class from % sand, % silt, % clay if not explicitly specified
    if pd.notna(input_data['texture']):
        input_data['soil'] = input_data['texture'] # use class if specified
    else:
        sand = float(soil_inputs.loc[soil_inputs['Parameter']=='sand',value_col].values[0])
        clay = float(soil_inputs.loc[soil_inputs['Parameter']=='clay',value_col].values[0])
        # Use functio
        input_data['soil'] = soil_texture_classifier(sand,clay)


    print('Input files read successfully.')
    print('--------------------------------------------------------------------------------------')

    return input_data

def run_SMEW(project_name, input_data):
    print('--------------------------------------------------------------------------------------')
    print('Running SMEW for',project_name,'...')

    # Unpack input variables
    data_in = SimpleNamespace(**input_data)


    # ---------------------------------------- #
    # TIME
    # ---------------------------------------- #

    t_end = 1*365 * data_in.n_years # [d]: number of simulated days
    dt = 1/(24*6) # [time resolution (d)]
    t=np.arange(0,t_end,dt)

    #units
    conv_mol = 1e6 # Conversion from moles to micromols 
    conv_Al = 1e3 # Conversion for Al species from micromols to nanomols

    #water balance
    keyword_wb = 1 # 1 = varying soil moisture. 0 = constant soil moisture

    #background inputs of cations and anions
    keyword_add = 1 # 1 = balance background losses. 0 = no addition

    # ---------------------------------------- #
    # HYDROCLIMATE AND WATER BALANCE
    # ---------------------------------------- #

    latitude = data_in.latitude*np.pi/180 # [radians]

    rho_bulk = data_in.rho_bulk*1e6 #soil dry mass (g/m3)   

    s_in = 0.5

    wind = 1*np.ones(len(t)) #[m/s]

    #temp [Celsius]
    [temp_air,temp_soil,temp_min,temp_max] = smew.temp(latitude, data_in.temp_av, data_in.temp_ampl_yr, data_in.temp_ampl_d, data_in.Zr,t_end,dt,data_in.day1)

    #ET0 [m/d]
    ET0 = smew.ET0(latitude,data_in.altitude,temp_air,temp_soil,temp_min,temp_max, wind,data_in.albedo,data_in.Zr,data_in.coastal,t_end,dt,data_in.day1)


    #stochastic rain with seasonality [m] (only works with multi-year)
    alfa_rain = data_in.alfa_rain*10**(-3) # Convert from mm to m
    rain = smew.rain_stoc_season(data_in.lambda_rain, alfa_rain, t_end, dt)

    #vegetation [g/m2]
    v_in = 0
    t0_v = data_in.t0_v-data_in.day1
    v = smew.veg(v_in, data_in.T_v, data_in.k_v, t0_v, temp_soil, dt)

    #moisture balance (I, Q [m], L, T, E [m/d])
    [s, s_w, s_i, I, L, T, E, Q, Irr, n] = smew.moisture_balance(rain, data_in.Zr, data_in.soil, ET0, v, data_in.k_v, keyword_wb, s_in,t_end,dt)

    # Set soil moisture to (almost) zero at negative soil temperatures = freezing 
    s[temp_soil<0]=0.01

    # Create filtered version for plots
    x = np.arange(len(s))
    mask = s != 0.01  # other parameters are zero where T_soil < 0
    s_filtered = np.interp(x, x[mask], s[mask])  # interpolation     

    # ---------------------------------------- #
    # ORGANIC CARBON and RESPIRATION
    # ---------------------------------------- #

    CO2_atm = smew.CO2_atm(conv_mol)

    #initial pCO2
    CO2_air_in = 10*CO2_atm # CO2 in soil air [mol-conv/l] 
    ratio_aut_het = 1

    ###################### 

    #Initial organic carbon
    SOC_in = rho_bulk*data_in.SOC_perc/100 #[gOC/m3]

    #SOC balance and respiration
    [SOC, r_het, r_aut, D] = smew.respiration(data_in.ADD, SOC_in, CO2_air_in, ratio_aut_het, data_in.soil, s, v, data_in.k_v, data_in.Zr, temp_soil,dt,conv_mol)

    # ---------------------------------------- #
    # SOIL BIOGEOCHEMISTRY
    # ---------------------------------------- #

    #CEC 
    CEC_tot = data_in.CEC_tot*1e-5*rho_bulk*data_in.Zr*conv_mol # [mol_c]

    f_CEC_in = np.array([data_in.f_Ca_in, data_in.f_Mg_in, data_in.f_K_in, data_in.f_Na_in, data_in.f_Al_in, data_in.f_H_in]) 
    if abs(sum(f_CEC_in)-1) > 1e-3:
        raise ValueError("Sum of fractions must be 1")

    [conc_in, K_CEC] = smew.f_CEC_to_conc(f_CEC_in, data_in.pH_in, data_in.soil, conv_mol, conv_Al)

    d_in = data_in.d_in*1e-6 # [m]
    psd_perc_in = data_in.psd_perc_in/100

    #print(sum(psd_perc_in))

    data = smew.biogeochem_balance(
        n, s, L, T, I, v, data_in.k_v, data_in.RAI, data_in.root_d, data_in.Zr, r_het, r_aut, D, temp_soil, 
        data_in.pH_in, conc_in, f_CEC_in, K_CEC, CEC_tot, data_in.Si_in, data_in.CaCO3_in, data_in.MgCO3_in, 
        data_in.M_rock_in, data_in.t_app, data_in.mineral, data_in.rock_f_in, d_in, psd_perc_in, data_in.SSA_in, data_in.diss_f, dt, conv_Al, conv_mol, keyword_add)

    # Interpolate zeros during frozen periods
    for  param in ['pH', 'Alk', 'M_rock', 'f_Ca', 'f_Mg', 'f_K', 'f_Na', 'f_H', 'f_Al', 'HCO3', 'CO3', 'CO2_air','CO2_w']:
        y = data[param]
        x = np.arange(len(y))
        # find where to interpolate
        mask = y != 0  # zero values where T_soil < 0
        data[param] = np.interp(x, x[mask], y[mask])  # interpolation       
        
    # include parameters not returned by biogeochem
    data['t'] = t
    data['rain'] = rain
    data['v'] = v
    data['s_filtered'] = s_filtered

    '''
    Print rock weathering fractions for every year
    '''
    for i in range(0,int(data_in.n_years)+1):
        idx = np.where(data['t'] == i*365)[0]
        weathered_fraction = (1-data['M_rock'][idx]/data['M_rock'][0])*100
        print('Weathered fraction in year ',i,':',weathered_fraction,'idx:',idx)

    print('Model run finished.')
    print('--------------------------------------------------------------------------------------')

    return data

def create_plots(project_name, model_results):
    print('--------------------------------------------------------------------------------------')
    print('Plotting outputs for',project_name,'...')

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    project_path = repo_root / 'Projects' / project_name

    # Create a figure and subplots
    fig = plt.figure(figsize=(12,15))
    gs = gridspec.GridSpec(5, 2)

    #--------------------------------------------------------------------------------
    # First panel - Hydroclimatic
    #--------------------------------------------------------------------------------

    axs1 = plt.subplot(gs[0, :])

    #Temp
    color = 'darkorange'
    axs1.set_xlabel('t (d)')
    axs1.set_ylabel('Temp (°C)', color=color)
    axs1.plot(model_results['t'], model_results['temp_soil'], color=color)
    axs1.tick_params(axis='y', labelcolor=color)

    # rainfall
    ax2 = axs1.twinx()
    color = 'navy'
    ax2.set_ylabel('Rain (mm)', color=color)
    ax2.plot(model_results['t'], model_results['rain']*1e3, color=color)  # Note the swapping of x and y data
    ax2.tick_params(axis='y', labelcolor=color)

    #moisture
    ax3 = axs1.twinx()
    color = 'tab:blue'
    ax3.set_ylabel('s (-)', color=color)
    ax3.plot(model_results['t'], model_results['s'], color=color)
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.spines['right'].set_position(('outward', 55))

    #--------------------------------------------------------------------------------
    # Second panel - pH and Alk
    #--------------------------------------------------------------------------------

    axs2 = plt.subplot(gs[1, 0])

    #pH
    color = 'darkorange'
    axs2.set_xlabel('t (d)')
    axs2.set_ylabel('pH', color=color)
    axs2.plot(model_results['t'], model_results['pH'], color=color)
    axs2.tick_params(axis='y', labelcolor=color)

    # Alk
    ax2 = axs2.twinx()
    color = 'navy'
    ax2.set_ylabel(r'[Alk] ($\mu$mol/l)', color=color)
    ax2.plot(model_results['t'], model_results['Alk'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    #axs2.set_xticklabels([])

    #--------------------------------------------------------------------------------
    # Third panel - IC
    #--------------------------------------------------------------------------------

    axs3 = plt.subplot(gs[1, 1])

    cumulative_sum = 0
    i = 0
    labels = [r'[CO$_3^{2-}$]',  r'[HCO$_3^{-}$]', r'[CO$_2]_\mathrm{w}$', r'[CO$_2]_\mathrm{a}$']
    colors = ['darkgreen', 'navy', 'grey', 'darkorange']
    for element in ['CO3', 'HCO3', 'CO2_w', 'CO2_air']:
        axs3.plot(model_results['t'], model_results[element] + cumulative_sum, label=labels[i], color = colors[i])
        axs3.fill_between(model_results['t'], cumulative_sum, cumulative_sum + model_results[element], alpha=0.5, color= colors[i])
        i = i + 1
        cumulative_sum += model_results[element]
        
    axs3.set_ylabel(r'$\mu$mol/l')
    axs3.set_xlabel('t(d)')
    #axs3.set_xticklabels([])
    axs3.yaxis.tick_right()
    axs3.yaxis.set_label_position("right")   
    axs3.legend()

    #--------------------------------------------------------------------------------
    # Forth panel - Weathering
    #--------------------------------------------------------------------------------

    axs4 = plt.subplot(gs[2, 0])

    color = 'grey'
    axs4.set_xlabel('t (d)')
    axs4.set_ylabel('Rock mass (%)')
    axs4.plot(model_results['t'], model_results['M_rock'][:]/model_results['M_rock'][0]*100, color=color)
    #axs4.tick_params(axis='y', labelcolor=color)

    #--------------------------------------------------------------------------------
    # Fifth panel - CEC
    #--------------------------------------------------------------------------------

    axs5 = plt.subplot(gs[2, 1])

    cumulative_sum = 0
    labels = [r'f$_\mathrm{H}$', r'f$_\mathrm{Na}$', r'f$_\mathrm{K}$', r'f$_\mathrm{Ca}$', r'f$_\mathrm{Mg}$', r'f$_\mathrm{Al}$']
    i=0
    for element in ['f_H', 'f_Na', 'f_K', 'f_Ca', 'f_Mg', 'f_Al']:
        axs5.plot(model_results['t'], model_results[element] + cumulative_sum, label=labels[i])
        axs5.fill_between(model_results['t'], cumulative_sum, cumulative_sum + model_results[element], alpha=0.5)
        cumulative_sum += model_results[element]
        i=i+1

    axs5.set_ylabel('CEC fraction')
    axs5.set_xlabel('t(d)')
    axs5.yaxis.tick_right()
    axs5.yaxis.set_label_position("right")
    axs5.legend()

    #--------------------------------------------------------------------------------
    # Sixth panel - Omega values
    #--------------------------------------------------------------------------------

    axs6 = plt.subplot(gs[3, 0])

    colors = ['tab:blue','goldenrod','darkgreen','firebrick','grey']
    mineral = model_results['mineral']
    for i in range(0,len(mineral)):
        axs6.plot(model_results['t'],model_results['Omega'][i,:], color=colors[i], label = mineral[i])
    #axs6.set_yscale('log')
    axs6.tick_params(axis='y', labelcolor='k')
    axs6.set_ylabel(r'Omega',color='k')
    axs6.legend(loc='lower right')


    #--------------------------------------------------------------------------------
    # Seventh panel - Mineral dissolution rates
    #--------------------------------------------------------------------------------

    axs7 = plt.subplot(gs[3, 1])

    colors = ['tab:blue','goldenrod','darkgreen','firebrick','grey']
    mineral = model_results['mineral']
    for i in range(0,len(mineral)):
        axs7.plot(model_results['t'],model_results['EW'][i,:]/(model_results['conv_mol']*24*3600), color=colors[i], label = mineral[i])
    axs7.set_yscale('log')
    axs7.tick_params(axis='y', labelcolor='k')
    axs7.set_ylabel(r'Weathering rate (mol m$^{-2}$ s$^{-1}$)',color='k')
    axs7.yaxis.tick_right()
    axs7.yaxis.set_label_position("right")
    axs7.legend(loc='lower right')

    #--------------------------------------------------------------------------------
    # Eighth panel - Cation concentrations
    #--------------------------------------------------------------------------------

    axs8 = plt.subplot(gs[4, 0])

    ions = ['Ca', 'Mg', 'Na', 'K', 'H', 'Si', 'Al_w', 'Al']

    for ion in ions:
        axs8.plot(model_results['t'],model_results[ion], label = ion)

    axs8.set_ylabel(r'Ion concentration ($\mu$mol/l)')
    axs8.legend(loc='lower right')

    #--------------------------------------------------------------------------------
    # Nineth panel - Vegetation
    #--------------------------------------------------------------------------------

    axs9 = plt.subplot(gs[4, 1])

    # temp
    axs9_2 = axs9.twinx()
    color = 'darkorange'
    axs9_2.set_ylabel('Temp (°C)', color=color)
    axs9_2.plot(model_results['t'], model_results['temp_soil'], color=color)
    axs9_2.tick_params(axis='y', labelcolor=color)

    color = 'darkgreen'
    axs9.plot(model_results['t'],model_results['v'], color = color)
    axs9.tick_params(axis='y', labelcolor=color)
    axs9.set_ylabel(r'Vegetation (g m$^{-2}$)',color=color)
    axs9.set_xlabel('t(d)')
    #axs6.legend()

    #-------------------------------------------------------------------
    #plot labels
    axs1.text(0.95, 0.87, '(a)', transform=axs1.transAxes, fontsize=12, fontweight='bold', zorder = 3)
    axs2.text(0.92, 0.87, '(b)', transform=axs2.transAxes, fontsize=12, fontweight='bold', zorder = 3)
    axs3.text(0.92, 0.87, '(c)', transform=axs3.transAxes, fontsize=12, fontweight='bold', zorder = 3)
    axs4.text(0.92, 0.87, '(d)', transform=axs4.transAxes, fontsize=12, fontweight='bold', zorder = 3)
    axs5.text(0.92, 0.87, '(e)', transform=axs5.transAxes, fontsize=12, fontweight='bold', zorder = 3)
    axs6.text(0.92, 0.87, '(f)', transform=axs6.transAxes, fontsize=12, fontweight='bold', zorder = 3)
    axs7.text(0.92, 0.87, '(g)', transform=axs7.transAxes, fontsize=12, fontweight='bold', zorder = 3)

    #plotting
    plt.tight_layout(pad=0.2)
    plt.savefig(project_path / 'out.png', dpi=300)


    # ----------------------------------------------------------------------------------------------------------------------------

    print('Figures completed.')
    print('--------------------------------------------------------------------------------------')




def main():
    # Set plot parameters
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
    mpl.rcParams['axes.linewidth'] = 0.5

    project = 'Vulkaneifel'
    field_name = 'Field C'

    input_data = read_input_data(project_name=project,value_col=field_name)
    output_data = run_SMEW(project_name=project,input_data=input_data)
    create_plots(project_name=project,model_results=output_data)

        




if __name__ == '__main__':
    main()


