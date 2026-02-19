from cosipy.util import fetch_wasabi_file
from pathlib import Path
import os
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download COSI DC3 BGO background CSVs from Wasabi into a local directory"
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        help="Base directory where the files will be stored (positional)",
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dir_flag",
        help="Base directory where the files will be stored (optional flag)",
    )
    return parser.parse_args()


def resolve_data_dir(args: argparse.Namespace) -> Path:
    # Precedence: flag > positional > BACKGROUND_DIR env > default
    default_dir = "/data/background"
    chosen_dir = args.data_dir_flag or args.data_dir or os.environ.get("BACKGROUND_DIR") or default_dir
    data_dir = Path(chosen_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir(args)
    
    albedo_photons = data_dir / "AlbedoPhotons_BGOhit_Total.csv.gz"
    if not albedo_photons.exists():
        fetch_wasabi_file(
            "COSI-SMEX/DC3/Data/Backgrounds/BGO/AlbedoPhotons_BGOhit_Total.csv.gz",
            albedo_photons,
        )

    cosmic_photons = data_dir / "CosmicPhotons_BGOhit_Total.csv.gz"
    if not cosmic_photons.exists():
        fetch_wasabi_file(
            "COSI-SMEX/DC3/Data/Backgrounds/BGO/CosmicPhotons_BGOhit_Total.csv.gz",
            cosmic_photons,
        )

    primary_alphas = data_dir / "PrimaryAlphas_BGOhit_Total.csv.gz"
    if not primary_alphas.exists():
        fetch_wasabi_file(
            "COSI-SMEX/DC3/Data/Backgrounds/BGO/PrimaryAlphas_BGOhit_Total.csv.gz",
            primary_alphas,
        )

    primary_protons = data_dir / "PrimaryProtons_BGOhit_Total.csv.gz"
    if not primary_protons.exists():
        fetch_wasabi_file(
            "COSI-SMEX/DC3/Data/Backgrounds/BGO/PrimaryProtons_BGOhit_Total.csv.gz",
            primary_protons,
        )

    saa_protons_1 = data_dir / "SAAprotons_BGOhit_Total.csv.gz"
    if not saa_protons_1.exists():
        fetch_wasabi_file(
            "COSI-SMEX/DC3/Data/Backgrounds/BGO/SAAprotons_BGOhit_Total.csv.gz",
            saa_protons_1,
        )

    secondary_electrons = data_dir / "SecondaryElectrons_BGOhit_Total.csv.gz"
    if not secondary_electrons.exists():
        fetch_wasabi_file(
            "COSI-SMEX/DC3/Data/Backgrounds/BGO/SecondaryElectrons_BGOhit_Total.csv.gz",
            secondary_electrons,
        )

    secondary_positrons = data_dir / "SecondaryPositrons_BGOhit_Total.csv.gz"
    if not secondary_positrons.exists():
        fetch_wasabi_file(
            "COSI-SMEX/DC3/Data/Backgrounds/BGO/SecondaryPositrons_BGOhit_Total.csv.gz",
            secondary_positrons,
        )
        
    ori_file = data_dir / "DC3_final_530km_3_month_with_slew_15sbins_GalacticEarth_SAA.ori"
    if not ori_file.exists():
        fetch_wasabi_file(
            "COSI-SMEX/DC3/Data/Orientation/DC3_final_530km_3_month_with_slew_15sbins_GalacticEarth_SAA.ori",
            ori_file,
        )


if __name__ == "__main__":
    main()