# SPDX-License-Identifier: AGPL-3.0-only
# -*- coding: utf-8 -*-
"""Derive monthly stochastic rainfall parameters from a weather-station CSV."""
import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------
REQUIRED_COLUMNS = ('date', 'prcp')
DATE_ALIASES = ('date', 'datetime', 'datahora')
PRCP_ALIASES = ('prcp', 'precipitation', 'valorMedida_mm')


def stoch_lambda_alpha(csv_path, date_col='date', prcp_col='prcp',
                        rain_threshold_mm=0.1):

    df = pd.read_csv(csv_path, sep=None, engine='python')

    def resolve_column(requested, aliases):
        if requested in df.columns:
            return requested
        normalized = {str(column).strip().lower(): column for column in df.columns}
        for alias in (requested,) + aliases:
            if alias.lower() in normalized:
                return normalized[alias.lower()]
        return None

    date_col = resolve_column(date_col, DATE_ALIASES)
    prcp_col = resolve_column(prcp_col, PRCP_ALIASES)
    missing = [name for name, column in (('date', date_col), ('prcp', prcp_col))
               if column is None]
    if missing:
        raise ValueError(f"csv_path is missing required column(s): {missing}")

    df['datetime'] = pd.to_datetime(df[date_col], dayfirst=False, errors='raise')
    df[prcp_col] = pd.to_numeric(
        df[prcp_col].astype(str).str.strip().str.replace(',', '.', regex=False),
        errors='coerce',
    ).fillna(0.0)
    df['month'] = df['datetime'].dt.month

    daily = df.set_index('datetime').resample('D').agg(
        total_rain_mm=(prcp_col, 'sum'),
        month=('month', 'mean'),
    )

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
    total_days = monthly['total_days'].to_numpy()
    rain_days = monthly['rain_days'].to_numpy()
    lamda = np.divide(rain_days, total_days,
                      out=np.zeros(12, dtype=float), where=total_days != 0)

    # alfa [m]: mean rain depth conditional on a rain day (total_rain_mm / rain_days),
    # converted from mm to m -- NOT total_rain_mm / n_years (that is total monthly
    # accumulation, a different quantity that inflates the annual total generated
    # by rain_stoc_season by an order of magnitude; see rain_days == 0 guard below)
    with np.errstate(invalid='ignore', divide='ignore'):
        alfa = 1e-3 * np.divide(
            monthly['total_rain_mm'].to_numpy(), rain_days,
            out=np.zeros(12, dtype=float), where=rain_days != 0)

    result = {
        'lamda': lamda,
        'alfa': alfa,
    }

    return result
