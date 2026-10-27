"""Batch runs of the ZeroEx field-data pipeline.

V2 of zeroex_input_data_wrapper: read one project data sheet once, then run
SMEW for several values of the feedstock inputs -- typically the application
rate `M_rock_in` [g/m2] and the specific surface area `SSA_in` [m2/g] -- and
keep one output_data dict per run.

The single-run functions (read_input_data, run_SMEW, create_plots) are reused
from V1 unchanged, so a batch run with the sheet's own values is the same run
V1 makes.

    import zeroex_input_data_wrapper_v2 as wp2

    batch = wp2.run_batch('Vulkaneifel', 'Field C',
                          scenarios={'M_rock_in': [2000, 4000, 8000],
                                     'SSA_in':    [1.0, 6.66]},
                          mode='grid', seed=1)
    batch.summary                    # one row per run
    batch.outputs['run_003']         # output_data of one run

Scenarios
---------
- dict of lists, mode='grid' : every combination (3 x 2 = 6 runs above).
- dict of lists, mode='zip'  : paired element-wise (lists of equal length).
- DataFrame                  : one row per run, columns are input names;
                               `mode` is ignored. A 'run_id' column, if
                               present, names the runs.
Any input that is not listed keeps the value from the sheet. SSA_in = nan
means "estimate SSA from the grain size distribution", as in the sheet.

Only the varied inputs change between runs: the same `seed` is passed to every
run, so all runs see the same rainfall series. With seed=None one seed is
drawn for the whole batch and recorded in the summary.
"""

import copy
import inspect
import itertools
import pathlib
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from . import zeroex_input_data_wrapper as wp
except ImportError:
    # imported as a top-level module (wrappers_postprocessing/ on sys.path),
    # which is how the project notebooks import V1
    import zeroex_input_data_wrapper as wp

# re-exported so a notebook only needs to import V2
read_input_data = wp.read_input_data
run_SMEW = wp.run_SMEW
create_plots = wp.create_plots

# Inputs the batch is designed around. Other scalar sheet inputs can also be
# varied (e.g. t_app, diss_f, n_years), but these two are validated.
BATCH_VARIABLES = ('M_rock_in', 'SSA_in')


@dataclass
class BatchResult:
    """What run_batch returns.

    summary   : DataFrame, one row per run -- the varied inputs plus scalar
                metrics (see _summarise).
    outputs   : {run_id: output_data}, empty when keep_outputs=False (then
                reload single runs with load_run).
    scenarios : DataFrame of the inputs each run used, indexed by run_id.
    out_dir   : where results were saved, or None.
    """
    summary: pd.DataFrame
    outputs: dict = field(default_factory=dict)
    scenarios: pd.DataFrame = None
    out_dir: pathlib.Path = None


def build_scenarios(scenarios, mode='grid'):
    """Turn a dict of value lists or a DataFrame into a scenario DataFrame.

    Returns a DataFrame indexed by run_id ('run_000', ...) with one column per
    varied input.
    """
    if isinstance(scenarios, pd.DataFrame):
        df = scenarios.copy()
        if 'run_id' in df.columns:
            df = df.set_index('run_id')
        else:
            df.index = [f'run_{i:03d}' for i in range(len(df))]
    elif isinstance(scenarios, dict):
        names = list(scenarios)
        values = [np.atleast_1d(np.asarray(scenarios[k], dtype=float)) for k in names]
        if mode == 'grid':
            rows = list(itertools.product(*values))
        elif mode == 'zip':
            lengths = {k: len(v) for k, v in zip(names, values)}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"mode='zip' needs lists of equal length, got {lengths}")
            rows = list(zip(*values))
        else:
            raise ValueError(f"mode must be 'grid' or 'zip', got {mode!r}")
        df = pd.DataFrame(rows, columns=names)
        df.index = [f'run_{i:03d}' for i in range(len(df))]
    else:
        raise TypeError('scenarios must be a dict of lists or a pandas DataFrame')

    df.index.name = 'run_id'
    if df.empty:
        raise ValueError('no scenarios to run')
    if df.index.duplicated().any():
        raise ValueError(f'duplicate run_id: {list(df.index[df.index.duplicated()])}')
    return df


def _validate(scenario_df, input_data):
    for name in scenario_df.columns:
        if name not in input_data:
            raise KeyError(f"'{name}' is not an input of this sheet "
                           f"(batch variables: {BATCH_VARIABLES})")
        if not np.isscalar(input_data[name]):
            raise TypeError(f"'{name}' is not a scalar input; only scalar inputs can be varied")
    if 'M_rock_in' in scenario_df and (scenario_df['M_rock_in'] < 0).any():
        raise ValueError('M_rock_in [g/m2] must be >= 0')
    if 'SSA_in' in scenario_df:
        ssa = scenario_df['SSA_in']
        if (ssa.notna() & (ssa <= 0)).any():
            # biogeochem_balance only uses SSA_in when > 0; 0 would silently
            # fall back to the PSD estimate. nan says that explicitly.
            raise ValueError('SSA_in [m2/g] must be > 0, or nan to estimate it from the PSD')


def _label(row):
    return ', '.join(f'{k}={v:g}' for k, v in row.items())


def _summarise(data, n_years):
    """Scalar metrics of one run, for the summary table."""
    t = data['t']
    M_rock = data['M_rock']
    out = {
        'pH_final': data['pH'][-1],
        'Alk_final': data['Alk'][-1],                 # [umol/l]
        'Alk_mean_last_year': np.mean(data['Alk'][t >= t[-1] - 365]),
        'pH_mean_last_year': np.mean(data['pH'][t >= t[-1] - 365]),
    }
    # weathered fraction of the applied rock [%], end of run and per year --
    # the same quantity run_SMEW prints
    if M_rock[0] > 0:
        out['weathered_pct_final'] = (1 - M_rock[-1] / M_rock[0]) * 100
        for y in range(1, int(n_years) + 1):
            idx = np.searchsorted(t, y * 365)
            if idx < len(t):
                out[f'weathered_pct_y{y}'] = (1 - M_rock[idx] / M_rock[0]) * 100
    else:
        out['weathered_pct_final'] = np.nan
    return out


def _run_one(project_name, input_data, run_id, overrides, seed, run_kwargs,
             out_dir, plot, save_outputs):
    """One batch member. Module-level so it can run in a worker process."""
    run_input = copy.deepcopy(input_data)
    run_input.update(overrides)
    label = _label(overrides)
    print(f'===== {run_id}: {label} =====')

    t0 = time.time()
    data = wp.run_SMEW(project_name, run_input, seed=seed, **run_kwargs)
    elapsed = time.time() - t0

    data['run_id'] = run_id
    data['batch_overrides'] = dict(overrides)

    summary = {'run_id': run_id, **overrides,
               **_summarise(data, run_input['n_years']),
               'seed': seed, 'runtime_s': elapsed}

    if out_dir is not None:
        run_dir = out_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        if save_outputs:
            with open(run_dir / 'output_data.pkl', 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        if plot:
            if 'out_dir' not in inspect.signature(wp.create_plots).parameters:
                # a notebook kernel that imported V1 before create_plots gained
                # out_dir keeps that old module object; V2 then gets it too
                raise RuntimeError(
                    'zeroex_input_data_wrapper is a stale import (create_plots has '
                    'no out_dir). Restart the kernel, or importlib.reload() V1 '
                    'and then V2.')
            wp.create_plots(project_name, data, out_dir=run_dir, title_suffix=label)
    return run_id, summary, data


def run_batch(project_name, value_col, scenarios, mode='grid', seed=None,
              out_dir='default', plot=False, save_outputs=True,
              keep_outputs=True, n_jobs=1, input_data=None, **run_kwargs):
    """Run SMEW once per scenario, from one data sheet.

    project_name, value_col : as in read_input_data (e.g. 'Vulkaneifel',
        'Field C').
    scenarios : dict of lists or DataFrame -- see the module docstring.
    mode : 'grid' (all combinations) or 'zip' (paired), for dict scenarios.
    seed : rainfall seed shared by every run. None draws one for the batch.
    out_dir : 'default' -> Projects/<project>/runs/batch_<value_col>_<timestamp>/;
        a path -> that directory; None -> nothing is written to disk.
        Each run gets <out_dir>/<run_id>/ (output_data.pkl, figures), and the
        batch gets scenarios.csv and summary.csv.
    plot : write the standard figure set per run (into its run directory).
    save_outputs : pickle each run's output_data into its run directory.
    keep_outputs : keep every output_data in memory in the returned result.
        output_data is ~47 MB per simulated year (~240 MB for a 5-year run),
        in memory and on disk; for large batches use keep_outputs=False and
        load_run() the runs you need.
    n_jobs : runs in parallel processes when > 1 (each process compiles the
        numba kernels once, so this pays off for batches of several runs).
        From a .py script (not a notebook), call run_batch under
        `if __name__ == '__main__':` -- macOS starts workers by re-importing
        the script.
    input_data : an already-read (and possibly edited) input dict; skips
        reading the sheet.
    **run_kwargs : passed to run_SMEW (interpolate_frozen, freeze_on).
    """
    if input_data is None:
        input_data = wp.read_input_data(project_name, value_col)
    scenario_df = build_scenarios(scenarios, mode)
    _validate(scenario_df, input_data)

    if seed is None:
        seed = int(np.random.default_rng().integers(2**31 - 1))
        print(f'No seed given; using seed={seed} for every run in the batch.')

    if out_dir == 'default':
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        stamp = time.strftime('%Y%m%d_%H%M%S')
        field_tag = str(value_col).replace(' ', '_')
        out_dir = repo_root / 'Projects' / project_name / 'runs' / f'batch_{field_tag}_{stamp}'
    if out_dir is not None:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        scenario_df.to_csv(out_dir / 'scenarios.csv')
    elif plot or save_outputs:
        if plot:
            raise ValueError('plot=True needs an out_dir')
        save_outputs = False

    jobs = [(project_name, input_data, run_id, row.to_dict(), seed, run_kwargs,
             out_dir, plot, save_outputs)
            for run_id, row in scenario_df.iterrows()]

    print(f'Batch: {len(jobs)} runs of {project_name} / {value_col}, '
          f'varying {list(scenario_df.columns)}')

    results = []
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(_run_one, *job) for job in jobs]
            for fut in futures:
                run_id, summary, data = fut.result()
                results.append((run_id, summary, data if keep_outputs else None))
    else:
        for job in jobs:
            run_id, summary, data = _run_one(*job)
            results.append((run_id, summary, data if keep_outputs else None))
            del data

    summary = pd.DataFrame([r[1] for r in results]).set_index('run_id')
    outputs = {r[0]: r[2] for r in results if r[2] is not None}

    if out_dir is not None:
        summary.to_csv(out_dir / 'summary.csv')
        print(f'Batch results written to {out_dir}')

    return BatchResult(summary=summary, outputs=outputs,
                       scenarios=scenario_df, out_dir=out_dir)


def load_run(out_dir, run_id):
    """Reload one run's output_data saved by run_batch."""
    with open(pathlib.Path(out_dir) / run_id / 'output_data.pkl', 'rb') as f:
        return pickle.load(f)


def load_batch(out_dir, load_outputs=False):
    """Reload a saved batch: summary, scenarios and (optionally) all outputs."""
    out_dir = pathlib.Path(out_dir)
    summary = pd.read_csv(out_dir / 'summary.csv', index_col='run_id')
    scenario_df = pd.read_csv(out_dir / 'scenarios.csv', index_col='run_id')
    outputs = {}
    if load_outputs:
        outputs = {run_id: load_run(out_dir, run_id) for run_id in summary.index}
    return BatchResult(summary=summary, outputs=outputs,
                       scenarios=scenario_df, out_dir=out_dir)


def main():
    wp._apply_style()

    project = 'Vulkaneifel'
    field_name = 'Field C'

    batch = run_batch(project, field_name,
                      scenarios={'M_rock_in': [2000, 4000, 8000],
                                 'SSA_in': [1.0, 6.66]},
                      mode='grid', seed=1)
    print(batch.summary)


if __name__ == '__main__':
    main()
