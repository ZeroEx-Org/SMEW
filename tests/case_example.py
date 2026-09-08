# SPDX-License-Identifier: AGPL-3.0-only
# -*- coding: utf-8 -*-
"""Examples/Example.ipynb as a headless, seeded function. One of the F0 cases.

run() is deterministic: same seed -> bit-identical output. diss_f is exposed so a
known-size perturbation can be injected to prove the golden master detects it.
"""
import numpy as np
import smew


def run(seed=42, t_end=365, dt=1 / (24 * 6), diss_f=1.0, day1=1, mineral="forsterite",
        temp_av=13, temp_ampl_yr=11, latitude=40 * np.pi / 180, altitude=33):
    t = np.arange(0, t_end, dt)
    conv_mol, conv_Al = 1e6, 1e3
    soil, Zr, rho_bulk = "loam", 0.3, 1.2e6

    temp_air, temp_soil, temp_min, temp_max = smew.temp(
        latitude, temp_av, temp_ampl_yr, 5, Zr, t_end, dt, day1)
    ET0 = smew.ET0(latitude, altitude, temp_air, temp_soil, temp_min, temp_max,
                   np.ones(len(t)), 0.25, Zr, False, t_end, dt, day1)

    lamda = 0.25
    rain = smew.rain_stoc(lamda, (1.2 / lamda) / 365, t_end, dt, seed=seed)

    k_v = 3000
    v = smew.veg(1 * k_v, 100, k_v, 0, temp_soil, dt)
    s, s_w, s_i, I, L, T, E, Q, Irr, n = smew.moisture_balance(
        rain, Zr, soil, ET0, v, k_v, 1, 0.5, t_end, dt)
    SOC, r_het, r_aut, D = smew.respiration(
        1, rho_bulk * 0.05 / 100, 10 * smew.CO2_atm(conv_mol), 1,
        soil, s, v, k_v, Zr, temp_soil, dt, conv_mol)

    f_CEC_in = np.array([0.30, 0.15, 0.10, 0.05, 0.00, 0.40])
    conc_in, K_CEC = smew.f_CEC_to_conc(f_CEC_in, 4, soil, conv_mol, conv_Al)

    data = smew.biogeochem_balance(
        n, s, L, T, I, v, k_v, 10, 0.4e-3, Zr, r_het, r_aut, D, temp_soil,
        4, conc_in, f_CEC_in, K_CEC, 10 * 1e-5 * rho_bulk * Zr * conv_mol,
        0, 0, 0, 1000, 0, [mineral], np.array([1]), np.array([100]) * 1e-6,
        np.array([1]), np.nan, diss_f, dt, conv_Al, conv_mol, 1)

    # series produced outside biogeochem_balance that the notebooks still plot
    extra = {"rain": rain, "s": s, "ET0": ET0, "SOC": SOC,
             "temp_soil": temp_soil, "L": L, "Q": Q, "v": v}
    return data, t, extra
