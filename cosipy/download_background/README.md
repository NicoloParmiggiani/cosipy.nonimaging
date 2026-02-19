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

Gli script ora richiedono path espliciti. Per il download devi passare obbligatoriamente la cartella di destinazione (BASE_DIR) come argomento posizionale. Il downloader creerà la cartella se non esiste ed eviterà di riscaricare file già presenti.

Esempi minimi:

- DC4:
```bash
python3 cosipy/download_background/download_background_from_wasabi_DC4.py /percorso/ai/dati/dc4
```
  - Nota: per DC4 viene scaricato anche il file `DC4_format_6_month_EarthGalactic_correct_timing.ori.zip` nella stessa cartella.

- DC3 (legacy):
```bash
python3 cosipy/download_background/download_background_from_wasabi_DC3.py /percorso/ai/dati/dc3
```

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


