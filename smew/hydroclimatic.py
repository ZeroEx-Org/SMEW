# SPDX-License-Identifier: AGPL-3.0-only
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:30:44 2019
"""
import numpy as np
import pyeto as eto
import scipy.stats


#--------------------------------------------------------------------------------------------------
# air and soil temperature

def temp(latitude,temp_av, temp_ampl_yr, temp_ampl_d, Zr,t_end,dt,day1):
         
    #air temperature
    x0 = 2*np.pi*day1/365 #initial day
    TT = (365/dt)
    x = x0 + (2*np.pi)*np.arange(0, t_end/dt)/TT #argument of sin and cos
    if latitude<0:
        x_max = 20*2*np.pi/365 #day of max temperature
    elif latitude>0:
        x_max = 200*2*np.pi/365
    phi = np.pi/2-x_max # Phase lag
    temp_air = temp_av + temp_ampl_yr*np.sin(x+phi) # [°C]
    temp_min = temp_air-temp_ampl_d/2 
    temp_max = temp_air + temp_ampl_d/2 
    
    # soil temperature
    TD = 0.203    # Thermal diffusivity [mm^2/s]
    TD = TD / 10**6 * 86400 # convert to m^2/d
    dd = (365*TD/np.pi)**(1/2) # Damping depth [m]
    temp_soil = temp_av + dd*temp_ampl_yr/(2*Zr)*(np.sin(x+phi)-np.cos(x+phi)+np.exp(-Zr/dd)*(np.cos(x-Zr/dd+phi)-np.sin(x-Zr/dd+phi)))
                                                        
    return(temp_air,temp_soil,temp_min,temp_max)


#--------------------------------------------------------------------------------------------------    
# ET0 with Penman-Monteith approach (standard grass ref - FAO - Allen et al., 1998, https://www.fao.org/3/X0490E/x0490e00.htm#Contents)

def ET0(latitude,altitude,temp_air,temp_soil,temp_min,temp_max, wind,albedo,Zr,coastal,t_end,dt,day1):
    
    # initialization
    ET0 = np.zeros(len(temp_air))
        
    for i in range(0, len(ET0)):

        #atmosphere
        atm_p = eto.atm_pressure(altitude) # atmospheric pressure [kPa]
        svp = eto.svp_from_t(temp_air[i]) # saturation vapor pressure [kPa]
        avp = eto.avp_from_tmin(temp_min[i]) # actual vapor pressure [kPa]
        delta_svp = eto.delta_svp(temp_air[i]) # Slope of saturation vapour pressure curve [kPa C-1]
        psy = eto.psy_const(atm_p) #psychrometric constant [kPa degC-1].
        
        #sun/light
        j = day1+np.floor(dt*i)
        j = j - np.floor((j-1)/365)*365
        sol_dec = eto.sol_dec(j) # solar declination [radians]
        ird = eto.inv_rel_dist_earth_sun(j) #Inverse relative distance between earth and the sun
        sha = eto.sunset_hour_angle(latitude, sol_dec) # Sunset hour angle [rad]
        
        #radiation
        et_rad = eto.et_rad(latitude, sol_dec, sha, ird) # Extraterrestrial radiation [MJ m-2 d-1]
        cs_rad = eto.cs_rad(altitude, et_rad) # Clear sky radiation [MJ m-2 day-1]
        sol_rad = eto.sol_rad_from_t(et_rad, cs_rad, temp_min[i], temp_max[i], coastal) # Gross incoming solar radiation [MJ m-2 d-1]
        ni_sw_rad = eto.net_in_sol_rad(sol_rad, albedo) # net-incoming shortwave rad [MJ m-2 day-1]
        no_lw_rad = eto.net_out_lw_rad(273+temp_min[i], 273+temp_max[i], sol_rad, cs_rad, avp) # net outgoing long wave rad [MJ m-2 d-1]
        net_rad = eto.net_rad(ni_sw_rad,no_lw_rad) # net incoming solar radiation [MJ m-2 day-1]
        shf = 0 # Soil heat flux (G) [MJ m-2 day-1]

        #potential ET
        ET0[i] = eto.fao56_penman_monteith(net_rad, temp_air[i]+273.15, wind[i], svp, avp, delta_svp, psy, shf) #[mm d-1]
    
    ET0 = ET0/1000 #[m d-1]
    
    return(ET0)


#--------------------------------------------------------------------------------------------------
# stochastic rain 

def rain_stoc(lamda, alfa, t_end, dt, seed = None):
    
    # seed = None  -> draws from numpy's global random state, series differs every call (legacy behaviour)
    # seed = <int> -> draws from a local generator, series is bit-for-bit reproducible and
    #                 independent of any other random call elsewhere in the program
    rng = None if seed is None else np.random.default_rng(seed)
    
    # simulated event number. 2x the expected count (lamda*t_end) is a buffer for the
    # Poisson spread in the realized count; the +10 protects short runs, where the plain
    # 2x buffer is exhausted often enough to matter (lamda*t_end ~ 8 -> ~1% of
    # realizations) and the series was then silently truncated before t_end.
    nb_ev = int(2*lamda*t_end) + 10
    
    # interarrival time [d]
    tau = scipy.stats.expon.rvs(scale = 1/lamda, loc = 0, size = int(nb_ev), random_state = rng)

    # intensity [m]
    h = scipy.stats.expon.rvs(scale = alfa, loc = 0, size = int(nb_ev), random_state = rng)

    # rainfall [m]
    rain = np.zeros(int(t_end/dt))
    t_event = 0. # storm arrival time [d], accumulated in continuous time
    for i in range(0, nb_ev):
        # accumulate the arrival time in days and snap to the grid once, at the end.
        # Snapping inside the running sum (t_event = int(t_event + tau/dt)) truncated
        # up to one timestep off every interarrival time, which compressed the storm
        # sequence by ~dt/2 per event and made the annual total depend on dt.
        t_event = t_event + tau[i]
        idx = int(t_event/dt)
        if idx < len(rain): # strict '<': idx == len(rain) indexed one past the end
            # '+=', not '=': two storms falling in the same timestep must add. With '='
            # the earlier one was silently overwritten and its depth lost -- common at
            # coarse dt, and the second reason the annual total drifted with dt (it
            # pulled the total down while the truncation above pulled it up).
            rain[idx] += h[i]
        else:
            break
                
    return rain


#--------------------------------------------------------------------------------------------------
# stochastic rain with seasonality 

def rain_stoc_season(lamda, alfa, t_end, dt, day1, seed = None):
    
    # lamda and alfa are 12-element arrays, one entry per month.
    # day1 = first day of the simulation, required and positional exactly as in temp()
    # and ET0(). Deliberately given no default: a default of 1 (January) is correct for
    # one site only, and a caller that forgot it would silently get the wrong season
    # rather than a TypeError.
    # see rain_stoc for the meaning of seed. One generator is shared across all
    # months/years, so the whole multi-year series is reproducible as a unit.
    rng = None if seed is None else np.random.default_rng(seed)
    
    days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) # month days
    
    # lamda/alfa are indexed January..December, but a simulation can start in any
    # month. Rotate all three arrays so that index 0 is the month containing day1.
    # Without this the first month of the run always drew January's rain statistics:
    # a run starting in October (day1 = 291, as in Projects/Vulkaneifel) got the whole
    # seasonal cycle shifted by nine months relative to its temperature and ET0.
    # Whole months only -- the run is taken to begin on the 1st of that month.
    month0 = int(np.searchsorted(np.cumsum(days), (day1 - 1) % 365 + 1))
    days  = np.roll(days, -month0)
    lamda = np.roll(np.asarray(lamda, dtype = float), -month0)
    alfa  = np.roll(np.asarray(alfa,  dtype = float), -month0)
    
    # note: only whole years are generated (int(t_end/365)), so a t_end that is not a
    # multiple of 365 d returns a rain array shorter than np.arange(0, t_end, dt).
    rain = np.array([]) # np.array() with no argument raises TypeError

    for j in range(0, int(t_end/365)):#year
        for i in range(0, len(days)):#month
            
            # simulated rainfall events per month (see rain_stoc for the +10 buffer)
            nb_ev = int(2*lamda[i]*days[i]) + 10
    
            # interarrival time [d]
            tau = scipy.stats.expon.rvs(scale = 1/lamda[i], loc = 0, size = int(nb_ev), random_state = rng)

            # intensity [m]
            h = scipy.stats.expon.rvs(scale = alfa[i], loc = 0, size = int(nb_ev), random_state = rng)

            # rainfall array [m]
            rain_month = np.zeros(int(days[i]/dt))
            t_event = 0. # storm arrival time [d] within the month, continuous
            for ii in range(0, nb_ev):
                t_event = t_event + tau[ii] # accumulate in days, snap to the grid once
                idx = int(t_event/dt)
                if idx < len(rain_month):
                    rain_month[idx] += h[ii] # '+=' so coincident storms add, not overwrite
                else:
                    break # t_event is monotonic, so every later draw is past month end
            rain = np.append(rain, rain_month)
    return rain