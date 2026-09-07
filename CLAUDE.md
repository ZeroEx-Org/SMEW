# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SMEW (Soil Model for Enhanced Weathering) is a scientific Python package implementing a dynamic, depth-averaged
ecohydrological and biogeochemical model of enhanced rock weathering (ERW) in the upper soil layer, published in
Bertagni et al., 2025, JAMES (https://doi.org/10.1029/2024MS004224). This is a research codebase, not a service —
there is no server, CLI entrypoint, or test suite. Work happens through Jupyter notebooks that import the `smew`
package and run simulations.

## Setup and commands

```bash
pip install -e .            # editable install (uses pyproject.toml)
pip install -r requirements.txt   # numpy, pandas, scipy, dill, matplotlib, numba
jupyter notebook             # run notebooks in Examples/
```

There is no build, lint, or test tooling configured (no test suite, no linter config). Validate changes by running
the relevant notebook(s) end-to-end in `Examples/` and checking outputs/plots look physically sensible — this is how
correctness is actually checked in this repo. `Examples/Example.ipynb` is the canonical smoke test for the core
pipeline.

## Package layout

- `smew/` — the numerical model. `smew/__init__.py` re-exports the public API used by notebooks (functions are
  called as `smew.xxx(...)`, not via submodule paths).
- `pyeto/` — vendored evapotranspiration library (Mark Richards, pyeto.readthedocs.io), used internally by
  `smew/hydroclimatic.py` for `ET0`.
- `Examples/` — Jupyter notebooks: `Example.ipynb` (generic simulation walkthrough) plus model-vs-experiment
  comparison notebooks (`Vials_Dietzen`, `Bottles_tePas`, `Mesocosm_Amann`, `Mesocosm_Kelland`) that validate the
  model against the datasets in `Exp_Data/`.
- `Exp_Data/` — digitized experimental data (via WebPlotDigitizer) from the published studies used for
  model-experiment comparison.
- `Projects/` — applied case studies / field-site spreadsheets (e.g. `Vulkaneifel`) that feed the model via the
  wrapper below.
- `wrappers_postprocessing/zeroex_input_data_wrapper.py` — converts real-world field/soil data sheets (sand/silt/clay
  %, etc.) into `smew` model inputs (e.g. `soil_texture_classifier` maps texture % to the `soil` string type used
  throughout `smew`, such as `"loam"`, `"sandy clay loam"`).
- `docs/` — reference PDFs (published papers this model is based on / compared against).

## Model architecture

The model has no single "run" function — a notebook wires together a pipeline of independent stages, each producing
time-series arrays over a shared time vector `t = np.arange(0, t_end, dt)`. Reading `Examples/Example.ipynb` top to
bottom is the fastest way to understand the data flow. In order:

1. **Hydroclimatic forcing** (`hydroclimatic.py`): `smew.temp(...)` generates soil/air temperature, `smew.ET0(...)`
   (wraps `pyeto`) generates reference evapotranspiration, `smew.rain_stoc(...)` / `rain_stoc_season(...)` generate
   stochastic rainfall.
2. **Vegetation** (`vegetation.py`): `smew.veg(...)` produces above-ground biomass `v` over time; `up_act` is used
   later for active nutrient uptake.
3. **Moisture balance** (`moisture.py`): `smew.moisture_balance(rain, Zr, soil, ET0, v, k_v, keyword_wb, s_in, t_end,
   dt)` — a Laio-type (2001) bucket model — returns relative soil moisture `s` plus water fluxes `I, L, T, E, Q, Irr`
   and porosity `n`, all needed downstream. `keyword_wb` toggles constant (0) vs. dynamic (1) soil moisture.
4. **Organic carbon / respiration** (`organic_carbon.py`): `smew.respiration(...)` returns `SOC, r_het, r_aut, D`
   (heterotrophic/autotrophic respiration and CO2 diffusivity), which feed soil CO2/pH chemistry.
5. **Biogeochemistry** (`biogeochem.py`, and `biogeochem2psd.py` for the particle-size-distribution–resolved
   variant): `smew.biogeochem_balance(...)` is the core solve — an ODE system for Ca, Mg, K, Na, Al, Si, alkalinity,
   inorganic carbon, and CEC-adsorbed cation fractions, coupled at each timestep to an implicit nonlinear system
   (`_biogeochem_equations_numba`, solved via `scipy.optimize.fsolve`) for carbonate/aluminum speciation and cation
   exchange equilibrium. It returns a `dict` of time series (e.g. `data['pH']`, `data['Alk']`).
   - `keyword_add` toggles background solute-loss replacement (1 = on, 0 = off).
   - Freezing is handled explicitly inside `biogeochem_balance`: when `temp_soil < 0`, chemistry is held at the
     state from just before the freezing period (`idx_before_freezing`) rather than integrated through it.
6. **Weathering kinetics**: silicate dissolution rates/saturation state and carbonate weathering come from
   `weathering.py` (`sil_Wr`, `sil_Omega`, `carb_W`, `psd_evol`) and `weathering_kinec.py` (`mineral_weathering`,
   `Omega_sil`, `carb_W`) — the latter implements the KINEC v3 kinetics database (current default; see recent commit
   history) and is imported directly by `biogeochem.py`. Mineral properties are looked up by name (e.g.
   `mineral = ["forsterite"]`) alongside per-mineral fraction, particle-diameter classes, and PSD weights.
7. **Initial conditions** (`ic.py`): converts between total concentrations, aqueous concentrations, and CEC-adsorbed
   fractions (`conc_to_f_CEC`, `f_CEC_to_conc`, `total_to_f_CEC_and_conc`, `f_CEC_and_conc_to_K`), plus
   dataset-specific IC helpers (`Amann`, `Kelland`) used by the corresponding comparison notebooks.
8. **Constants** (`constants.py`): all physical/soil/mineral constants, including `soil_const(soil)` — a lookup by
   soil-texture string (`"sand"`, `"loamy sand"`, `"sandy loam"`, `"loam"`, `"clay loam"`, `"clay"`, ...) for Laio
   et al. (2001) retention-curve parameters, used consistently across `moisture.py`, `biogeochem.py`, and the input
   wrapper's `soil_texture_classifier`.
9. **Plotting helpers** (`complementary.py`): `fig_CEC`, `fig_IC`, `mov_avg` for the standard result plots used
   across the example/comparison notebooks.

Hot inner loops (the moisture bucket model, the biogeochemical ODE/implicit-system integration) are JIT-compiled
with `@njit` from `numba` — keep numba-compatible code (no arbitrary Python objects, limited use of dicts/strings)
inside those functions and their `_numba`-suffixed helpers.

## Working conventions

- Units matter and are inconsistent by design across the model (moles vs. micromoles/nanomoles via `conv_mol` /
  `conv_Al`, meters vs. micrometers for particle diameters, mmol_c/100g vs. mol_c for CEC, etc.) — every function
  signature and inline comment documents its expected units; preserve/propagate the same conventions rather than
  normalizing them when editing.
- Multi-mineral / multi-application inputs are passed as parallel arrays/lists (`mineral`, `rock_f_in`, `d_in`,
  `psd_perc_in`, `SSA_in`, `M_rock_in`, `t_app`) — when adding a mineral or amendment, extend all of these in lockstep.
- Soil type is threaded through the codebase as a plain string keyed into `soil_const` — don't introduce a separate
  enum/constant without updating every lookup site.
- License is AGPL-3.0-only; keep the `SPDX-License-Identifier: AGPL-3.0-only` header on new/edited `smew`/`pyeto`
  source files consistent with existing files.
