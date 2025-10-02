GRB dataset simulation with simulate_random_dataset.py
======================================================

This document explains how to use `simulate_random_dataset.py` to create GRB datasets on an HEALPix grid. You can configure:

- **Flux**: fixed (single value) or uniformly varying within a "min:max" range.
- **Spectrum**: fixed by choosing one of the three predefined spectra, or random among the three.
- **Number of GRBs per pixel**: one per pixel or multiple per pixel.

For each grid direction, the tool creates a job folder with `GRB.source` and `job.slurm`, and submits the simulations via Docker/SLURM.

Requirements and important notes
--------------------------------

- Ensure the host directory for data is available: it is mounted into the container as `/data` using `--path-data` (default `/cosi-grb/data`).
- The `--path-analysis` argument must point to a host path, typically `$HOME/cosi-grb/data/analysis/<run-name>`.
- The `--run-name` argument must be the name of the run subfolder (i.e., `<run-name>`), so that inside the container it corresponds to `/data/analysis/<run-name>`.
- `--geometry-path` must be a path valid from inside the container, e.g., `/data/models/.../*.geo.setup`, i.e., under one of the mounted directories (`/data`, `/data01`, `/data02`).

Predefined spectra
------------------

Three spectra indexed 0, 1, 2 are available in the code:

- **0 (soft)**: `Band 10 10000 -1.9 -3.7 230`
- **1 (medium)**: `Band 10 10000 -1 -2.3 699.9`
- **2 (hard)**: `Comptonized 10 10000 -0.5 1500`

General usage
-------------

Generic invocation example (adjust paths):

```bash
RUN="my_run"
BASE="$HOME/cosi-grb/data/analysis/$RUN"

python3 simulate_random_dataset.py \
  --path-analysis="$BASE" \
  --run-name="$RUN" \
  --nside=8 \
  --grbs-per-pixel=-1 \
  --seed=-1 \
  --random-flux="1:30" \
  --random-spectrum="yes" \
  --limit-grb-number=-1 \
  --geometry-path="/data/models/massmodel-cosi-smex-detailed/COSISMEX.sim_BGOreading.geo.setup" \
  --path-repository="/cosi-grb/codidl/" \
  --path-data="/cosi-grb/data" \
  --noise="true" \
  --max-job=200
```

Meaning of key options
----------------------

- **--nside**: HEALPix resolution (number of pixels = 12 × nside²).
- **--grbs-per-pixel**:
  - `-1`: a single GRB per pixel.
  - `N > 0`: N GRBs for each pixel (replicas with different seeds).
- **--random-flux**:
  - Single value (e.g., `--random-flux=10`) for fixed flux.
  - Range `"min:max"` (e.g., `"1:30"`) for a uniform draw within the range.
- **--random-spectrum**:
  - `"yes"`: randomly picks one among spectra 0/1/2.
  - `"0"`, `"1"`, `"2"`: fixes to soft/medium/hard, respectively.
- **--limit-grb-number**: if >0, randomly samples that many coordinates from the grid (useful to limit the dataset size).
- **--seed**: global seed; `-1` generates a different seed per job.
- **--noise**: string forwarded to post-processing (e.g., `true`/`false`).
- **--max-job**: maximum number of simultaneous jobs in the SLURM queue.

Recipes for the requested datasets
----------------------------------

Below are the instructions to create the three main dataset modes and the dataset for Aitoff plots.

1) Training dataset (Deep Learning)
-----------------------------------

- **nside**: 128
- **GRB limit**: 100000 (use `--limit-grb-number=100000`)
- **Spectra**: random (`--random-spectrum="yes"`)
- **Fluxes**: uniform between 1 and 30 (`--random-flux="1:30"`)
- **GRBs per pixel**: one (`--grbs-per-pixel=-1`)

Example command:

```bash
RUN="train_n128"
BASE="$HOME/cosi-grb/data/analysis/$RUN"

python3 simulate_random_dataset.py \
  --path-analysis="$BASE" \
  --run-name="$RUN" \
  --nside=128 \
  --grbs-per-pixel=-1 \
  --seed=-1 \
  --random-flux="1:30" \
  --random-spectrum="yes" \
  --limit-grb-number=100000 \
  --geometry-path="/data/models/massmodel-cosi-smex-detailed/COSISMEX.sim_BGOreading.geo.setup" \
  --path-repository="/cosi-grb/codidl/" \
  --path-data="/cosi-grb/data" \
  --noise="true" \
  --max-job=400
```

2) Test dataset (Deep Learning)
--------------------------------

- **nside**: 64
- **Spectra**: random (`--random-spectrum="yes"`)
- **Fluxes**: uniform between 1 and 30 (`--random-flux="1:30"`)
- **GRBs per pixel**: one (`--grbs-per-pixel=-1`)

Example command:

```bash
RUN="test_n64"
BASE="$HOME/cosi-grb/data/analysis/$RUN"

python3 simulate_random_dataset.py \
  --path-analysis="$BASE" \
  --run-name="$RUN" \
  --nside=64 \
  --grbs-per-pixel=-1 \
  --seed=-1 \
  --random-flux="1:30" \
  --random-spectrum="yes" \
  --limit-grb-number=-1 \
  --geometry-path="/data/models/massmodel-cosi-smex-detailed/COSISMEX.sim_BGOreading.geo.setup" \
  --path-repository="/cosi-grb/codidl/" \
  --path-data="/cosi-grb/data" \
  --noise="true" \
  --max-job=300
```

3) Look-up tables for chi² fit (nside=32, fixed flux=100)
---------------------------------------------------------

Create three distinct datasets, one for each spectral model: soft (0), medium (1), hard (2). In all three cases the flux is fixed at 100.

Example commands:

```bash
RUN="lut_soft_n32"
BASE="$HOME/cosi-grb/data/analysis/$RUN"
python3 simulate_random_dataset.py \
  --path-analysis="$BASE" \
  --run-name="$RUN" \
  --nside=32 \
  --grbs-per-pixel=-1 \
  --seed=-1 \
  --random-flux=100 \
  --random-spectrum="0" \
  --limit-grb-number=-1 \
  --geometry-path="/data/models/massmodel-cosi-smex-detailed/COSISMEX.sim_BGOreading.geo.setup" \
  --path-repository="/cosi-grb/codidl/" \
  --path-data="/cosi-grb/data" \
  --noise="true" \
  --max-job=200

RUN="lut_medium_n32"
BASE="$HOME/cosi-grb/data/analysis/$RUN"
python3 simulate_random_dataset.py \
  --path-analysis="$BASE" \
  --run-name="$RUN" \
  --nside=32 \
  --grbs-per-pixel=-1 \
  --seed=-1 \
  --random-flux=100 \
  --random-spectrum="1" \
  --limit-grb-number=-1 \
  --geometry-path="/data/models/massmodel-cosi-smex-detailed/COSISMEX.sim_BGOreading.geo.setup" \
  --path-repository="/cosi-grb/codidl/" \
  --path-data="/cosi-grb/data" \
  --noise="true" \
  --max-job=200

RUN="lut_hard_n32"
BASE="$HOME/cosi-grb/data/analysis/$RUN"
python3 simulate_random_dataset.py \
  --path-analysis="$BASE" \
  --run-name="$RUN" \
  --nside=32 \
  --grbs-per-pixel=-1 \
  --seed=-1 \
  --random-flux=100 \
  --random-spectrum="2" \
  --limit-grb-number=-1 \
  --geometry-path="/data/models/massmodel-cosi-smex-detailed/COSISMEX.sim_BGOreading.geo.setup" \
  --path-repository="/cosi-grb/codidl/" \
  --path-data="/cosi-grb/data" \
  --noise="true" \
  --max-job=200
```

4) Dataset for Aitoff plots (nside=32, flux=10, medium spectrum, 10 GRB/pixel)
-------------------------------------------------------------------------------

- **nside**: 32
- **flux**: fixed at 10 (`--random-flux=10`)
- **spectrum**: medium (`--random-spectrum="1"`)
- **GRBs per pixel**: 10 (`--grbs-per-pixel=10`)

Example command:

```bash
RUN="aitoff_n32_flux10_medium_10x"
BASE="$HOME/cosi-grb/data/analysis/$RUN"

python3 simulate_random_dataset.py \
  --path-analysis="$BASE" \
  --run-name="$RUN" \
  --nside=32 \
  --grbs-per-pixel=10 \
  --seed=-1 \
  --random-flux=10 \
  --random-spectrum="1" \
  --limit-grb-number=-1 \
  --geometry-path="/data/models/massmodel-cosi-smex-detailed/COSISMEX.sim_BGOreading.geo.setup" \
  --path-repository="/cosi-grb/codidl/" \
  --path-data="/cosi-grb/data" \
  --noise="true" \
  --max-job=300
```

Tips
----

- Always quote flux ranges (e.g., `"1:30"`) to prevent the shell from interpreting the colon.
- To speed up submission, tune `--max-job` according to your cluster resources.
- For very large datasets, consider using `--limit-grb-number` to generate a controlled subset of directions.

Create datasets with create_dataset.py
======================================

After simulations complete, use `create_dataset.py` to build pickle datasets from the generated `.evt.lc` files. The script supports three modes:

- **files_flat**: iterate over a file glob and produce flat arrays.
- **healpix_flat**: scan simulation folders by HEALPix pixel and produce flat arrays.
- **healpix_matrix**: group by HEALPix pixel and build a fixed-size matrix per pixel.

The script writes two pickles: `<output>` and `<output>_shared`, separating standard vs "shared" ASIC data.

General CLI
-----------

```bash
python3 create_dataset.py \
  --mode <files_flat|healpix_flat|healpix_matrix> \
  --output "/abs/path/to/output.pkl" \
  [--suffix ".evt.lc"] \
  # files_flat only:
  [--glob "/abs/path/**/*.evt.lc"] \
  # healpix_* only:
  [--root-dir "/abs/path/to/run"] [--nside <N>] \
  # healpix_matrix only:
  [--entries-per-pixel 10]
```

Recommended modes per dataset
-----------------------------

- **Training dataset**: use flat mode (`files_flat`).
- **Test dataset**: use HEALPix mode (`healpix_flat`).
- **Look-up tables (chi² fit)**: use HEALPix mode (`healpix_flat`).
- **Aitoff plots dataset**: use HEALPix matrix mode (`healpix_matrix`).

1) Training dataset (flat mode)
--------------------------------

Assuming simulations are under `$HOME/cosi-grb/data/analysis/train_n128`:

```bash
RUN="train_n128"
ROOT="$HOME/cosi-grb/data/analysis/$RUN"
OUT="$HOME/cosi-grb/data/datasets/${RUN}.pkl"

python3 create_dataset.py \
  --mode files_flat \
  --glob "$ROOT/**/*.evt.lc" \
  --output "$OUT"
```

2) Test dataset (HEALPix flat)
------------------------------

Assuming simulations are under `$HOME/cosi-grb/data/analysis/test_n64` with `nside=64`:

```bash
RUN="test_n64"
ROOT="$HOME/cosi-grb/data/analysis/$RUN"
OUT="$HOME/cosi-grb/data/datasets/${RUN}.pkl"

python3 create_dataset.py \
  --mode healpix_flat \
  --root-dir "$ROOT" \
  --nside 64 \
  --output "$OUT"
```

3) Look-up tables (HEALPix flat)
---------------------------------

For each spectral dataset (soft/medium/hard) at `nside=32`:

```bash
RUN="lut_soft_n32";   ROOT="$HOME/cosi-grb/data/analysis/$RUN"; OUT="$HOME/cosi-grb/data/datasets/${RUN}.pkl"
python3 create_dataset.py --mode healpix_flat --root-dir "$ROOT" --nside 32 --output "$OUT"

RUN="lut_medium_n32"; ROOT="$HOME/cosi-grb/data/analysis/$RUN"; OUT="$HOME/cosi-grb/data/datasets/${RUN}.pkl"
python3 create_dataset.py --mode healpix_flat --root-dir "$ROOT" --nside 32 --output "$OUT"

RUN="lut_hard_n32";   ROOT="$HOME/cosi-grb/data/analysis/$RUN"; OUT="$HOME/cosi-grb/data/datasets/${RUN}.pkl"
python3 create_dataset.py --mode healpix_flat --root-dir "$ROOT" --nside 32 --output "$OUT"
```

4) Aitoff plots dataset (HEALPix matrix)
----------------------------------------

Assuming simulations are under `$HOME/cosi-grb/data/analysis/aitoff_n32_flux10_medium_10x` with `nside=32` and 10 GRBs per pixel:

```bash
RUN="aitoff_n32_flux10_medium_10x"
ROOT="$HOME/cosi-grb/data/analysis/$RUN"
OUT="$HOME/cosi-grb/data/datasets/${RUN}.pkl"

python3 create_dataset.py \
  --mode healpix_matrix \
  --root-dir "$ROOT" \
  --nside 32 \
  --entries-per-pixel 10 \
  --output "$OUT"
```

Notes
-----

- The default `--suffix` is `.evt.lc`, matching the light-curve count files produced by `read_sim_files.py`.
- `create_dataset.py` emits two files: `<output>` and `<output>_shared`.
- Ensure you pass absolute paths (or `$HOME/...`) to avoid path resolution issues.


