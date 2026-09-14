import sys, pathlib

import os
import time; start_time = time.time()
import importlib as imp
import smew
#import figEW

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import numpy as np
import warnings

# import dill
import pandas as pd
import json

from types import SimpleNamespace

# Plot styling.
#
# Categorical palette is Okabe-Ito (the CVD-safe scientific standard), with two
# substitutions for line legibility on a light surface: the pale yellow #F0E442
# (1.3:1 contrast, invisible as a thin line) becomes a dark gold, and pure black
# becomes a dark gray. Validated with the dataviz palette validator in all-pairs
# mode -- the mode that applies when every series shares one plot and the lines
# cross: worst normal-vision pair dE 15.6 (PASS), worst CVD pair dE 6.3 (warn
# band). The warn band is only legal with a secondary encoding, which is why
# _series_style() dashes slots 5-8: that splits every confusable pair
# (green/pink, orange/vermillion, gold/vermillion) across solid vs dashed.
#
# Colors are assigned by fixed slot order and never cycled, so the same slot
# means the same series across panels.
PALETTE = ['#0072B2',  # blue
           '#E69F00',  # orange
           '#009E73',  # green
           '#7A6A00',  # dark gold
           '#56B4E9',  # sky blue
           '#D55E00',  # vermillion
           '#CC79A7',  # pink
           '#333333',  # dark gray
           '#4b006e'   #royal purple
           ]  

# Stacked-area panels draw from this subset instead of the slot order above.
# The dark gold and dark gray are chosen for line legibility and turn muddy as
# large fills, so fills skip them: blue, orange, green, sky, pink, vermillion.
# Validated in adjacent-pair mode (the mode that applies to a stack): worst
# adjacent CVD dE 9.6, worst adjacent normal-vision dE 16.4 -- both PASS.
FILL_SLOTS = [0, 1, 2, 4, 6, 5]
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
SURFACE = '#fcfcfb'

# Line weights. The series are ~260k points over 1800+ days, so dense
# oscillations blob together above ~1.2pt; thin lines with round caps stay
# legible where the trace is spiky.
LW_MAIN = 1.4      # single-series panels (pH, Alk, rock mass, vegetation)
LW_SERIES = 1.0    # multi-series line panels (ions, minerals)
LW_EDGE = 0.8      # boundary line on stacked fills
LW_CONTEXT = 0.9   # secondary/context traces (rain, temp behind vegetation)

XLABEL = 'Time [Days]'


def _apply_style():
    """Figure-wide rcParams. Set here rather than in main() so notebooks that
    call create_plots() directly get the same styling."""
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Lato', 'Helvetica Neue', 'Helvetica',
                                       'Avenir Next', 'Arial', 'DejaVu Sans']
    # Keep math ($\mu$, $\Omega$, subscripts) in the same face as the body text
    # instead of falling back to DejaVu.
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'Helvetica Neue'
    mpl.rcParams['mathtext.it'] = 'Helvetica Neue:italic'
    mpl.rcParams['mathtext.bf'] = 'Helvetica Neue:bold'
    mpl.rcParams['axes.linewidth'] = 0.8
    mpl.rcParams['axes.axisbelow'] = True
    mpl.rcParams['lines.solid_capstyle'] = 'round'
    mpl.rcParams['lines.solid_joinstyle'] = 'round'
    mpl.rcParams['lines.dash_capstyle'] = 'round'
    mpl.rcParams['lines.antialiased'] = True


def _fill_color(i):
    """Color for band `i` of a stacked-area panel."""
    return PALETTE[FILL_SLOTS[i % len(FILL_SLOTS)]]


def _series_style(i):
    """Color + linestyle for categorical slot `i`.

    Slots 5-8 are dashed: the secondary encoding that makes the CVD warn-band
    pairs separable regardless of hue perception.
    """
    color = PALETTE[i % len(PALETTE)]
    linestyle = '-' if i % len(PALETTE) < 4 else (0, (5, 1.5))
    return color, linestyle


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.spines['top'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
        spine.set_linewidth(0.8)
    ax.tick_params(axis='x', colors=INK_SECONDARY, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _add_legend(ax, ncol=1, **kwargs):
    """Legend above the axes, right-aligned.

    Placed outside the data area on purpose: these series (log-scale weathering
    rates, Omega square waves) fill the plot box, so any in-axes location sits
    on top of lines.
    """
    leg = ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.005), ncol=ncol,
                     frameon=False, fontsize=8, handlelength=1.7,
                     handletextpad=0.5, columnspacing=1.1, labelspacing=0.3,
                     borderaxespad=0.0, **kwargs)
    for text in leg.get_texts():
        text.set_color(INK_SECONDARY)
    return leg


def _panel_label(ax, label):
    """Panel letter above the axes on the left, opposite the legend."""
    ax.text(0.0, 1.005, label, transform=ax.transAxes, fontsize=11,
            fontweight='bold', color=INK, va='bottom', ha='left')


def _plot_hydroclimatic(ax, model_results):
    color_temp, color_rain, color_moist = PALETTE[5], PALETTE[0], PALETTE[8]

    ax.set_xlabel(XLABEL)
    ax.set_ylabel('Temp (°C)', color=color_temp)
    ax.plot(model_results['t'], model_results['temp_soil'], color=color_temp, linewidth=LW_MAIN, alpha=0.9)
    _style_axis(ax)
    ax.tick_params(axis='y', colors=color_temp, labelsize=8)

    ax_rain = ax.twinx()
    ax_rain.set_ylabel('Rain (mm)', color=color_rain)
    ax_rain.plot(model_results['t'], model_results['rain'] * 1e3, color=color_rain, linewidth=0.7, alpha=0.7)
    ax_rain.tick_params(axis='y', colors=color_rain, labelsize=8)
    ax_rain.grid(False)
    ax_rain.spines['top'].set_visible(False)

    ax_moist = ax.twinx()
    ax_moist.spines['right'].set_position(('outward', 55))
    ax_moist.set_ylabel('s (-)', color=color_moist)
    ax_moist.plot(model_results['t'], model_results['s'], color=color_moist, linewidth=LW_CONTEXT, alpha=0.85)
    ax_moist.tick_params(axis='y', colors=color_moist, labelsize=8)
    ax_moist.grid(False)
    ax_moist.spines['top'].set_visible(False)
    

def _plot_pH_alk(ax, model_results):
    color_pH, color_alk = PALETTE[1], PALETTE[0]

    ax.set_xlabel(XLABEL)
    ax.set_ylabel('pH', color=color_pH)
    ax.plot(model_results['t'], model_results['pH'], color=color_pH, linewidth=LW_MAIN)
    _style_axis(ax)
    ax.tick_params(axis='y', colors=color_pH, labelsize=8)

    ax_alk = ax.twinx()
    ax_alk.set_ylabel(r'[Alk] ($\mu$mol/l)', color=color_alk)
    ax_alk.plot(model_results['t'], model_results['Alk'], color=color_alk, linewidth=LW_MAIN, alpha=0.9)
    ax_alk.tick_params(axis='y', colors=color_alk, labelsize=8)
    ax_alk.grid(False)
    ax_alk.spines['top'].set_visible(False)
    return ax_alk


def _plot_cec(ax, model_results):
    elements = ['f_H', 'f_Na', 'f_K', 'f_Ca', 'f_Mg', 'f_Al']
    labels = [r'f$_\mathrm{H}$', r'f$_\mathrm{Na}$', r'f$_\mathrm{K}$', r'f$_\mathrm{Ca}$', r'f$_\mathrm{Mg}$', r'f$_\mathrm{Al}$']

    cumulative = np.zeros_like(model_results['t'])
    for i, (element, label) in enumerate(zip(elements, labels)):
        color = _fill_color(i)
        y = model_results[element]
        ax.plot(model_results['t'], cumulative + y, color=color, linewidth=LW_EDGE, label=label)
        ax.fill_between(model_results['t'], cumulative, cumulative + y, color=color, alpha=0.55, linewidth=0)
        cumulative = cumulative + y

    ax.set_ylabel('CEC fraction')
    ax.set_xlabel(XLABEL)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    _style_axis(ax)
    return _add_legend(ax, ncol=6)


def _plot_ion_concentrations(ax, model_results):
    """Cation/anion concentration panel.

    Al and Al_w are solved internally in nmol/l (an extra `conv_Al` factor on
    top of the `conv_mol` scaling shared by Ca/Mg/Na/K/H/Si) -- see
    smew/ic.py `f_CEC_to_conc` and smew/biogeochem.py's `Al_s [mol-conv_Al]`
    residual scaling. Divide by `conv_Al` here so they plot on the same
    micromol/l axis as the other ions instead of appearing ~1000x too high.
    """
    ions = ['Ca', 'Mg', 'Na', 'K', 'H', 'Si', 'Al_w', 'Al']
    nmol_ions = {'Al_w', 'Al'}
    conv_Al = model_results['conv_Al']

    for i, ion in enumerate(ions):
        color, linestyle = _series_style(i)
        y = model_results[ion]
        if ion in nmol_ions:
            y = y / conv_Al
        ax.plot(model_results['t'], y, color=color, linestyle=linestyle,
                 linewidth=LW_SERIES, alpha=0.9, label=ion)

    ax.set_ylabel(r'Ion concentration ($\mu$mol/l)')
    ax.set_xlabel(XLABEL)
    _style_axis(ax)
    return _add_legend(ax, ncol=8)


def _plot_carbonate(ax, model_results):
    elements = ['CO3', 'HCO3', 'CO2_w', 'CO2_air']
    labels = [r'[CO$_3^{2-}$]', r'[HCO$_3^{-}$]', r'[CO$_2]_\mathrm{w}$', r'[CO$_2]_\mathrm{a}$']

    cumulative = np.zeros_like(model_results['t'])
    for i, (element, label) in enumerate(zip(elements, labels)):
        color = _fill_color(i)
        y = model_results[element]
        ax.plot(model_results['t'], cumulative + y, color=color, linewidth=LW_EDGE, label=label)
        ax.fill_between(model_results['t'], cumulative, cumulative + y, color=color, alpha=0.55, linewidth=0)
        cumulative = cumulative + y

    ax.set_ylabel(r'$\mu$mol/l')
    ax.set_xlabel(XLABEL)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    _style_axis(ax)
    return _add_legend(ax, ncol=4)


def _plot_weathering_rate(ax, model_results):
    mineral = model_results['mineral']
    for i in range(len(mineral)):
        color, linestyle = _series_style(i)
        ax.plot(model_results['t'], model_results['EW'][i, :] / (model_results['conv_mol'] * 24 * 3600),
                 color=color, linestyle=linestyle, linewidth=LW_SERIES, alpha=0.9, label=mineral[i])

    ax.set_yscale('log')
    ax.set_ylabel(r'Weathering rate (mol m$^{-2}$ s$^{-1}$)')
    ax.set_xlabel(XLABEL)
    _style_axis(ax)
    return _add_legend(ax, ncol=min(len(mineral), 4))


def _plot_omega(ax, model_results):
    mineral = model_results['mineral']
    for i in range(len(mineral)):
        color, linestyle = _series_style(i)
        ax.plot(model_results['t'], model_results['Omega'][i, :], color=color,
                 linestyle=linestyle, linewidth=LW_SERIES, alpha=0.9, label=mineral[i])

    ax.set_ylabel(r'$\Omega$')
    ax.set_xlabel(XLABEL)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    _style_axis(ax)
    return _add_legend(ax, ncol=min(len(mineral), 4))


def _plot_rock_mass(ax, model_results):
    ax.plot(model_results['t'], model_results['M_rock'][:] / model_results['M_rock'][0] * 100,
             color=PALETTE[0], linewidth=LW_MAIN)
    ax.set_ylabel('Rock mass (%)')
    ax.set_xlabel(XLABEL)
    _style_axis(ax)


def _plot_vegetation(ax, model_results):
    color_temp, color_veg = PALETTE[1], PALETTE[2]

    ax_temp = ax.twinx()
    ax_temp.set_ylabel('Temp (°C)', color=color_temp)
    ax_temp.plot(model_results['t'], model_results['temp_soil'], color=color_temp, linewidth=LW_CONTEXT, alpha=0.7)
    ax_temp.tick_params(axis='y', colors=color_temp, labelsize=8)
    ax_temp.grid(False)
    ax_temp.spines['top'].set_visible(False)

    ax.plot(model_results['t'], model_results['v'], color=color_veg, linewidth=LW_MAIN)
    ax.set_ylabel(r'Vegetation (g m$^{-2}$)', color=color_veg)
    ax.set_xlabel(XLABEL)
    _style_axis(ax)
    ax.tick_params(axis='y', colors=color_veg, labelsize=8)


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
    # Keep the column/field name with the inputs so it can travel through
    # run_SMEW into the results and label the figures.
    input_data['value_col'] = value_col

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

def run_SMEW(project_name, input_data, interpolate_frozen=True, seed=None):
    """Run the full SMEW pipeline for a project data sheet.

    interpolate_frozen : rewrite the 13 series listed below across frozen steps.
        True is the historical behaviour and stays the default. Pass False to
        get the raw biogeochem output, which is what a golden master needs --
        see the note at that block.
    seed : passed to rain_stoc_season. None keeps the unseeded default, so
        repeated runs differ; an int makes the run reproducible, which is what
        tests/freeze_cases.py uses.
    """
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
    conv_Al = 1#e3  Conversion for Al species from micromols to nanomols; decreased as there seems to be an error in the Al species conversion. 

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
    # day1 must be passed so the monthly lambda/alfa align with the start month
    rain = smew.rain_stoc_season(data_in.lambda_rain, alfa_rain, t_end, dt, data_in.day1, seed=seed)

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
    #
    # NOTE: this rewrites the 13 series below in place, so the raw biogeochem
    # output is not recoverable from what this function returns. On a
    # Vulkaneifel-climate January start that is ~30% of the record, and up to
    # 7.6 pH units. It matters for golden-master tests, which otherwise pin the
    # model and this interpolation together and cannot tell a regression in one
    # from a change in the other.
    #
    # interpolate_frozen=False returns the raw series instead. Default unchanged.
    if interpolate_frozen:
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
    data['value_col'] = input_data.get('value_col')

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

def create_plots(project_name, model_results, value_col=None):
    """Draw the standard output figure set.

    value_col : the field/column being modelled, used in the figure title.
        Defaults to whatever read_input_data recorded and run_SMEW carried
        through, so callers normally do not need to pass it; pass it
        explicitly to override, or for results produced before that existed.
    """
    print('--------------------------------------------------------------------------------------')
    print('Plotting outputs for',project_name,'...')

    if value_col is None:
        value_col = model_results.get('value_col')

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    project_path = repo_root / 'Projects' / project_name

    _apply_style()

    # Create a figure and subplots
    #
    # Panel layout:
    #   1 (row 0, full width): hydroclimatic forcing
    #   2 (row 1): pH/Alk (left) | CEC (right)
    #   3 (row 2): ion concentrations (left) | carbonate system (right)
    #   4 (row 3): weathering rate, log scale (left) | Omega (right)
    #   5 (row 4): rock mass (left) | vegetation (right)
    fig = plt.figure(figsize=(12, 16))
    fig.patch.set_facecolor(SURFACE)
    gs = gridspec.GridSpec(5, 2)

    axs1 = plt.subplot(gs[0, :])
    _plot_hydroclimatic(axs1, model_results)

    axs2 = plt.subplot(gs[1, 0])
    _plot_pH_alk(axs2, model_results)
    axs3 = plt.subplot(gs[1, 1])
    _plot_cec(axs3, model_results)

    axs4 = plt.subplot(gs[2, 0])
    _plot_ion_concentrations(axs4, model_results)
    axs5 = plt.subplot(gs[2, 1])
    _plot_carbonate(axs5, model_results)

    axs6 = plt.subplot(gs[3, 0])
    _plot_weathering_rate(axs6, model_results)
    axs7 = plt.subplot(gs[3, 1])
    _plot_omega(axs7, model_results)

    axs8 = plt.subplot(gs[4, 0])
    _plot_rock_mass(axs8, model_results)
    axs9 = plt.subplot(gs[4, 1])
    _plot_vegetation(axs9, model_results)

    #-------------------------------------------------------------------
    #plot labels
    panel_axes = [axs1, axs2, axs3, axs4, axs5, axs6, axs7, axs8, axs9]
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
    for ax, label in zip(panel_axes, panel_labels):
        _panel_label(ax, label)

    #plotting
    # suptitle before tight_layout, with rect reserving the top strip -- called
    # after, it would sit on top of panel (a)'s legend row.
    title = f'Project: {project_name}'
    if value_col:
        title = f'{title} – {value_col}'
    fig.suptitle(title, fontsize=14, fontweight='bold', color=INK, y=0.998)
    plt.tight_layout(pad=0.4, h_pad=1.6, rect=[0, 0, 1, 0.982])
    plt.savefig(project_path / 'out.png', dpi=300, facecolor=SURFACE)
    plt.close(fig)

    # ----------------------------------------------------------------------------------------------------------------------------
    # Separate exports for panels 2, 3 and 4, alongside the main figure
    # ----------------------------------------------------------------------------------------------------------------------------
    panel_exports = [
        ('out_panel2_pH_CEC.png', _plot_pH_alk, _plot_cec, ('(a)', '(b)')),
        ('out_panel3_ions_carbonate.png', _plot_ion_concentrations, _plot_carbonate, ('(a)', '(b)')),
        ('out_panel4_weathering_omega.png', _plot_weathering_rate, _plot_omega, ('(a)', '(b)')),
    ]
    for filename, plot_left, plot_right, labels in panel_exports:
        fig_p, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.6))
        fig_p.patch.set_facecolor(SURFACE)
        plot_left(ax_left, model_results)
        plot_right(ax_right, model_results)
        _panel_label(ax_left, labels[0])
        _panel_label(ax_right, labels[1])
        plt.tight_layout(pad=0.4)
        plt.savefig(project_path / filename, dpi=300, facecolor=SURFACE)
        plt.close(fig_p)

    print('Figures completed.')
    print('--------------------------------------------------------------------------------------')




def main():
    # Set plot parameters (create_plots applies the same style itself, so
    # notebooks calling it directly get identical output)
    _apply_style()

    project = 'Vulkaneifel'
    field_name = 'Field C'

    input_data = read_input_data(project_name=project,value_col=field_name)
    output_data = run_SMEW(project_name=project,input_data=input_data)
    create_plots(project_name=project,model_results=output_data,value_col=field_name)

        




if __name__ == '__main__':
    main()


