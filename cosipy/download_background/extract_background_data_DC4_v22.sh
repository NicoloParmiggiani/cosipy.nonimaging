#!/usr/bin/env sh

# Background light-curve extraction launcher
# -----------------------------------------
# Usage:
#   ./extract_background_data.sh BASE_DIR ORI_FILE
#
# Example:
#   ./extract_background_data.sh /data/background /data/file.ori DC[3-4-4_v22]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Require exactly three arguments
if [ "$#" -ne 3 ]; then
    echo "Error: Exactly three arguments required."
    echo "Usage: $0 BASE_DIR ORI_FILE"
    exit 1
fi

BASE_DIR="$1"
ORI_FILE="$2"
DATA_CHALLENGE="$3"

# Check BASE_DIR exists
if [ ! -d "$BASE_DIR" ]; then
    echo "BASE_DIR not found: $BASE_DIR"
    exit 1
fi

# Check ORI_FILE exists
if [ ! -f "$ORI_FILE" ]; then
    echo "ORI file not found: $ORI_FILE"
    exit 1
fi




nohup python "$SCRIPT_DIR/extract_lc_from_csv.py" "$BASE_DIR/AlbedoNeutrons_BGOhit_Total.csv.gz"      "$BASE_DIR/AlbedoNeutrons_BGOhit_Total"      1 1 "$ORI_FILE" "$DATA_CHALLENGE" &
nohup python "$SCRIPT_DIR/extract_lc_from_csv.py" "$BASE_DIR/AlbedoPhotons_BGOhit_Total.csv.gz"       "$BASE_DIR/AlbedoPhotons_BGOhit_Total"       1 1 "$ORI_FILE" "$DATA_CHALLENGE" &
nohup python "$SCRIPT_DIR/extract_lc_from_csv.py" "$BASE_DIR/CosmicPhotons_BGOhit_Total.csv.gz"       "$BASE_DIR/CosmicPhotons_BGOhit_Total"       1 1 "$ORI_FILE" "$DATA_CHALLENGE" &
nohup python "$SCRIPT_DIR/extract_lc_from_csv.py" "$BASE_DIR/PrimaryAlphas_BGOhit_Total.csv.gz"       "$BASE_DIR/PrimaryAlphas_BGOhit_Total"       1 1 "$ORI_FILE" "$DATA_CHALLENGE" &
nohup python "$SCRIPT_DIR/extract_lc_from_csv.py" "$BASE_DIR/PrimaryProtons_BGOhit_Total.csv.gz"      "$BASE_DIR/PrimaryProtons_BGOhit_Total"      1 1 "$ORI_FILE" "$DATA_CHALLENGE" &
nohup python "$SCRIPT_DIR/extract_lc_from_csv.py" "$BASE_DIR/SecondaryElectrons_BGOhit_Total.csv.gz"  "$BASE_DIR/SecondaryElectrons_BGOhit_Total"  1 1 "$ORI_FILE" "$DATA_CHALLENGE" &
nohup python "$SCRIPT_DIR/extract_lc_from_csv.py" "$BASE_DIR/SecondaryPositrons_BGOhit_Total.csv.gz"  "$BASE_DIR/SecondaryPositrons_BGOhit_Total"  1 1 "$ORI_FILE" "$DATA_CHALLENGE" &
#nohup python "$SCRIPT_DIR/extract_lc_from_csv.py" "$BASE_DIR/PrimaryElectrons_BGOhit_Total.csv.gz"    "$BASE_DIR/PrimaryElectrons_BGOhit_Total"    1 1 "$ORI_FILE" "$DATA_CHALLENGE" &
nohup python "$SCRIPT_DIR/extract_lc_from_csv.py" "$BASE_DIR/SecondaryProtons_BGOhit_Total.csv.gz"    "$BASE_DIR/SecondaryProtons_BGOhit_Total"    1 1 "$ORI_FILE" "$DATA_CHALLENGE" &
nohup python "$SCRIPT_DIR/extract_lc_from_csv.py" "$BASE_DIR/SAAprotons_BGOhit_Total.csv.gz"    "$BASE_DIR/SAAprotons_BGOhit_Total"    1 1 "$ORI_FILE" "$DATA_CHALLENGE" &
