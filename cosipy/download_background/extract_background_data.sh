#!/usr/bin/env sh

# Background light-curve extraction launcher
# -----------------------------------------
# This script launches parallel conversions of several compressed CSV background
# files into light-curve count files using extract_lc_from_csv.py.
#
# Usage:
#   ./extract_background_data.sh [BASE_DIR]
#
# Where BASE_DIR is the directory containing the input .csv.gz files and where
# the output files will be written. You can also set the environment variable
# BACKGROUND_DIR; precedence is: CLI argument > $BACKGROUND_DIR > $HOME/cosi-grb/data/background
#
# Example:
#   BACKGROUND_DIR="$HOME/cosi-grb/data/background" ./extract_background_data.sh
#   ./extract_background_data.sh /data/background

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BASE_DIR="${1:-${BACKGROUND_DIR:-$HOME/cosi-grb/data/background}}"

ORI_FILE="$BASE_DIR/DC3_final_530km_3_month_with_slew_15sbins_GalacticEarth_SAA.ori"
if [ ! -f "$ORI_FILE" ]; then
    echo "ORI file not found: $ORI_FILE"
    exit 1
fi
nohup python "$SCRIPT_DIR/extract_lc_from_csv.py"  "$BASE_DIR/AlbedoPhotons_BGOhit_Total.csv.gz" "$BASE_DIR/AlbedoPhotons_BGOhit_Total" 1 1 "$ORI_FILE" &
nohup python "$SCRIPT_DIR/extract_lc_from_csv.py"  "$BASE_DIR/CosmicPhotons_BGOhit_Total.csv.gz" "$BASE_DIR/CosmicPhotons_BGOhit_Total" 1 1 "$ORI_FILE" &
nohup python "$SCRIPT_DIR/extract_lc_from_csv.py"  "$BASE_DIR/PrimaryAlphas_BGOhit_Total.csv.gz" "$BASE_DIR/PrimaryAlphas_BGOhit_Total" 1 1 "$ORI_FILE" &
nohup python "$SCRIPT_DIR/extract_lc_from_csv.py"  "$BASE_DIR/PrimaryProtons_BGOhit_Total.csv.gz" "$BASE_DIR/PrimaryProtons_BGOhit_Total" 1 1 "$ORI_FILE" &
nohup python "$SCRIPT_DIR/extract_lc_from_csv.py"  "$BASE_DIR/SAAprotons_BGOhit_Total.csv.gz" "$BASE_DIR/SAAprotons_BGOhit_Total" 1 1 "$ORI_FILE" &
nohup python "$SCRIPT_DIR/extract_lc_from_csv.py"  "$BASE_DIR/SecondaryElectrons_BGOhit_Total.csv.gz" "$BASE_DIR/SecondaryElectrons_BGOhit_Total" 1 1 "$ORI_FILE" &
nohup python "$SCRIPT_DIR/extract_lc_from_csv.py"  "$BASE_DIR/SecondaryPositrons_BGOhit_Total.csv.gz" "$BASE_DIR/SecondaryPositrons_BGOhit_Total" 1 1 "$ORI_FILE" &
