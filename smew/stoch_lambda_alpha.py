# SPDX-License-Identifier: AGPL-3.0-only
# -*- coding: utf-8 -*-
"""
Derive the monthly stochastic-climate parameters used elsewhere in smew
(rain_stoc_season's lamda/alfa, temp's temp_av/temp_ampl_yr/temp_ampl_d) from a
raw hourly weather-station CSV, e.g. the INMET-format station files used in
Projects/Earthstone.
"""
import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------
# Of the ~27 columns in a raw INMET station CSV (pressure, radiation, humidity,
# wind, station metadata, ...), only six feed anything smew actually consumes:
# date/hr to build the timestamp, prcp for lamda/alfa, and temp/tmax/tmin for
# temp_av/temp_ampl_yr/temp_ampl_d (the exact inputs of smew.temp()). Everything
# else in the source file (stp, gbrd, hmdy, wdsp, lat/lon, ...) is unused by the
# notebook this was extracted from and is not read here.
REQUIRED_COLUMNS = ('date', 'hr', 'prcp', 'temp', 'tmax', 'tmin')


def stoch_lambda_alpha(csv_path, date_col='date', hour_col='hr', prcp_col='prcp',
                        temp_col='temp', tmax_col='tmax', tmin_col='tmin',
                        rain_threshold_mm=0.1):

    df = pd.read_csv(csv_path)

    missing = [c for c in (date_col, hour_col, prcp_col, temp_col, tmax_col, tmin_col)
               if c not in df.columns]
    if missing:
        raise ValueError(f"csv_path is missing required column(s): {missing}")

    df['datetime'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[hour_col].astype(str))
    df['month'] = df['datetime'].dt.month

    daily = df.set_index('datetime').resample('D').agg(
        total_rain_mm=(prcp_col, 'sum'),
        month=('month', 'mean'),
        temp=(temp_col, 'mean'),
        tmax=(tmax_col, 'max'),
        tmin=(tmin_col, 'min'),
    )
    daily['daily_temp_range'] = np.abs(daily['tmax'] - daily['tmin'])

    # temperature parameters for smew.temp(): yearly mean, half the yearly range,
    # half the mean daily range
    temp_av = daily['temp'].mean()
    temp_ampl_yr = (daily['temp'].max() - daily['temp'].min()) / 2
    temp_ampl_d = daily['daily_temp_range'].mean() / 2

    monthly = daily.groupby('month').agg(
        total_rain_mm=('total_rain_mm', 'sum'),
        rain_days=('total_rain_mm', lambda x: (x > rain_threshold_mm).sum()),
        total_days=('total_rain_mm', 'count'),
    )
    # reindex Jan..Dec: a month absent from the data (e.g. a station record that
    # doesn't span a full year) would otherwise silently shift every later month
    # into the wrong slot instead of getting a defined (zero-rain) entry
    monthly = monthly.reindex(np.arange(1, 13), fill_value=0)

    # lamda [d^-1]: fraction of days in the month with measurable rain, used
    # directly as the exponential/Poisson-process rate in rain_stoc_season
    lamda = (monthly['rain_days'] / monthly['total_days']).to_numpy()

    # alfa [m]: mean rain depth conditional on a rain day (total_rain_mm / rain_days),
    # converted from mm to m -- NOT total_rain_mm / n_years (that is total monthly
    # accumulation, a different quantity that inflates the annual total generated
    # by rain_stoc_season by an order of magnitude; see rain_days == 0 guard below)
    with np.errstate(invalid='ignore', divide='ignore'):
        alfa = 1e-3 * (monthly['total_rain_mm'] / monthly['rain_days']).to_numpy()
    alfa = np.where(monthly['rain_days'].to_numpy() == 0, 0.0, alfa)

    return {
        'lamda': lamda,
        'alfa': alfa,
        'temp_av': temp_av,
        'temp_ampl_yr': temp_ampl_yr,
        'temp_ampl_d': temp_ampl_d,
    }
