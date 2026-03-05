# Kondo Two-Layer Zigzag Data Pipeline

## Scope
This workflow standardizes how to:
1. run multi-`D` measurements for finite `J_perp` in `kondo_two_layer_zigzag_vmps`,
2. sync raw JSON from cluster `data/` to local `data/`,
3. post-process and plot layer-resolved localized-spin correlations with consistent style.

The local plotting pipeline is:
`plot/kondo_two_layer_zigzag/postprocess.py`

## 1. Cluster Run (Multi-D)
Use a `params.json` with multiple `Dmax` values, e.g.

```json
"Dmax": [500, 1000, 2000, 3000]
```

Executable:

```bash
mpirun -n <NPROC> ./kondo_two_layer_zigzag_vmps params.json
```

The code now measures after each `D` and writes files like:
- `l0szszJperp...Jk...t2...U...Ly...Lx...D<d>.json`
- `l0spsm...`
- `l0smsp...`
- `l1szsz...`
- `l1spsm...`
- `l1smsp...`
- `sz_loc0...`, `sz_loc1...`, `sz_elec...`, `n_elec...`

## 2. Sync Cluster Data to Local `data/`
From local repo root (`LNOsuperconductivity`), use include/exclude rsync so only one case is synced:

```bash
rsync -av --prune-empty-dirs \
  --include='l0szszJperp0.5Jk-4t2-0.6U18Ly4Lx20D*.json' \
  --include='l0spsmJperp0.5Jk-4t2-0.6U18Ly4Lx20D*.json' \
  --include='l0smspJperp0.5Jk-4t2-0.6U18Ly4Lx20D*.json' \
  --include='l1szszJperp0.5Jk-4t2-0.6U18Ly4Lx20D*.json' \
  --include='l1spsmJperp0.5Jk-4t2-0.6U18Ly4Lx20D*.json' \
  --include='l1smspJperp0.5Jk-4t2-0.6U18Ly4Lx20D*.json' \
  --exclude='*' \
  <user>@<cluster>:/path/to/LNOsuperconductivity/data/ ./data/
```

Replace the parameter token and remote path for your case.

## 3. Local Post-Process + Plot
From repo root:

```bash
python3 plot/kondo_two_layer_zigzag/postprocess.py \
  --data-dir ./data \
  --out-dir ./plot/kondo_two_layer_zigzag/figures \
  --jperp 0.5 --jk -4 --t2 -0.6 --u 18 --lx 20 --ly 4 \
  --fit-order auto
```

Optional filters:
- explicit `D` list: `--d-list 500,1000,2000,3000`
- threshold filter: `--min-d 1000`

## 4. Outputs
The script writes:
- per-`D` profiles (CSV):
  - `layer0_profiles_*.csv`
  - `layer1_profiles_*.csv`
- `D->∞` extrapolated profiles (CSV):
  - `layer0_extrapolated_*.csv`
  - `layer1_extrapolated_*.csv`
- figures:
  - `spincorr_profile_*.pdf`
  - `spincorr_profile_*.png`

Plot style follows current project conventions (Arial, thick lines, muted palette).

## 5. Sanity Checks
The pipeline enforces:
- complete `D` sets (all six `l0/l1` spin-correlation files must exist),
- layer purity (`l0*` must contain only layer-0 localized indices; same for `l1*`),
- fixed reference-site consistency inside each file.

If any check fails, fix data-generation or sync selection before plotting.
