### Download DC3 background data and extract light curves

This folder contains utilities to download COSI DC3 BGO background CSV files from Wasabi and extract light-curve count files used by the pipeline.

Tools:
- `download_background_from_wasabi.py`: downloads the compressed CSVs from Wasabi (uses `cosipy.util.fetch_wasabi_file`).
- `extract_background_data.sh`: converts the CSVs into light-curve count files by calling `extract_lc_from_csv.py` in parallel.

Prerequisites
-------------
- Run inside the Docker container (recommended), where `/data` is mounted to your host data directory.
- Python environment with `cosipy` available (provided in the container image).

Where files are stored
----------------------
- The download script saves files under the container path `/data/background` by default (see `data_dir` inside `download_background_from_wasabi.py`).
- The extraction script reads and writes in a single directory that you can pass as an argument or via `BACKGROUND_DIR`.

1) Download from Wasabi
-----------------------

Run the downloader; any missing files will be fetched into the chosen directory. You can pass it as a positional argument, with `--data-dir`, or via `BACKGROUND_DIR` (default: `/data/background`):

```bash
#enter inside docker container
docker exec --user cosi -it cosi_grb_bgo_container bash

#activate cosipy env inside docker
source $HOME/deeplearning/bin/activate

#define the path for the code
export CONTAINER_COSI_CODE="/home/cosi/cosi/cosidl"

# default destination
cd $CONTAINER_COSI_CODE/download_background
python3 download_background_from_wasabi.py

# positional directory
python3 download_background_from_wasabi.py /data/background

# flag form
python3 download_background_from_wasabi.py --data-dir /data/background

# environment variable
BACKGROUND_DIR=/data/background python3 download_background_from_wasabi.py
```

Notes:
- If you prefer a different destination, use the CLI argument or `BACKGROUND_DIR` above; editing the script is no longer necessary.
- The script is idempotent: it skips files that already exist.

2) Extract light-curve counts
-----------------------------

Use the shell launcher to convert all downloaded CSVs into LC files. You can provide the base directory explicitly or via `BACKGROUND_DIR`.

Option A: pass the directory as an argument (matches the downloader default):

```bash
cd $CONTAINER_COSI_CODE/download_background
sh extract_background_data.sh /data/background
```

Option B: use an environment variable and run without arguments:

```bash
export BACKGROUND_DIR="/data/background"
sh extract_background_data.sh
```

What gets generated
-------------------
For each input `*.csv.gz`, the extractor writes:
- a CSV at the same base path without the `.gz` extension, and
- a light-curve count file with suffix `.lc`.

Example outputs in `/data/background`:
- `AlbedoPhotons_BGOhit_Total.csv` and `AlbedoPhotons_BGOhit_Total.csv.lc`
- `CosmicPhotons_BGOhit_Total.csv` and `CosmicPhotons_BGOhit_Total.csv.lc`
... and similarly for the other components.

Tips
----
- If running outside Docker, ensure paths are writable and adjust them consistently in both scripts.
- Keep background data under `/data/...` inside the container so other parts of the pipeline can find them.


