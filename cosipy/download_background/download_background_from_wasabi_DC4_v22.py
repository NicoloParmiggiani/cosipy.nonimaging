from cosipy.util import fetch_wasabi_file
from pathlib import Path
import os
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download COSI DC4 BGO background CSVs from Wasabi into a local directory"
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
    default_dir = "/data/background/dc4/new"
    chosen_dir = args.data_dir_flag or args.data_dir or os.environ.get("BACKGROUND_DIR") or default_dir
    data_dir = Path(chosen_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir(args)
    
    albedo_photons = data_dir / "AlbedoPhotons_BGOhit_Total.csv.gz"
    print(albedo_photons)
    if not albedo_photons.exists():
        fetch_wasabi_file(
            "COSI-SMEX/DC4/Data/Backgrounds/BGO/ASICS_massmodel_corrected/AlbedoPhotons_BGOhit_Total.csv.gz",
            albedo_photons,
        )

    cosmic_photons = data_dir / "CosmicPhotons_BGOhit_Total.csv.gz"
    print(cosmic_photons)
    if not cosmic_photons.exists():
        fetch_wasabi_file(
            "COSI-SMEX/DC4/Data/Backgrounds/BGO/ASICS_massmodel_corrected/CosmicPhotons_BGOhit_Total.csv.gz",
            cosmic_photons,
        )

    primary_alphas = data_dir / "PrimaryAlphas_BGOhit_Total.csv.gz"
    print(primary_alphas)
    if not primary_alphas.exists():
        fetch_wasabi_file(
            "COSI-SMEX/DC4/Data/Backgrounds/BGO/ASICS_massmodel_corrected/PrimaryAlphas_BGOhit_Total.csv.gz",
            primary_alphas,
        )

    primary_protons = data_dir / "PrimaryProtons_BGOhit_Total.csv.gz"
    print(primary_protons)
    if not primary_protons.exists():
        fetch_wasabi_file(
            "COSI-SMEX/DC4/Data/Backgrounds/BGO/ASICS_massmodel_corrected/PrimaryProtons_BGOhit_Total.csv.gz",
            primary_protons,
        )

    secondary_electrons = data_dir / "SecondaryElectrons_BGOhit_Total.csv.gz"
    print(secondary_electrons)
    if not secondary_electrons.exists():
        fetch_wasabi_file(
            "COSI-SMEX/DC4/Data/Backgrounds/BGO/ASICS_massmodel_corrected/SecondaryElectrons_BGOhit_Total.csv.gz",
            secondary_electrons,
        )

    secondary_positrons = data_dir / "SecondaryPositrons_BGOhit_Total.csv.gz"
    print(secondary_positrons)
    if not secondary_positrons.exists():
        fetch_wasabi_file(
            "COSI-SMEX/DC4/Data/Backgrounds/BGO/ASICS_massmodel_corrected/SecondaryPositrons_BGOhit_Total.csv.gz",
            secondary_positrons,
        )
        
    secondary_protons = data_dir / "SecondaryProtons_BGOhit_Total.csv.gz"
    print(secondary_protons)
    if not secondary_protons.exists():
        fetch_wasabi_file(
            "COSI-SMEX/DC4/Data/Backgrounds/BGO/ASICS_massmodel_corrected/SecondaryProtons_BGOhit_Total.csv.gz",
            secondary_protons,
        )
        
    albedo_neutrons = data_dir / "AlbedoNeutrons_BGOhit_Total.csv.gz"
    print(albedo_neutrons)
    if not albedo_neutrons.exists():
        fetch_wasabi_file(
            "COSI-SMEX/DC4/Data/Backgrounds/BGO/ASICS_massmodel_corrected/AlbedoNeutrons_BGOhit_Total.csv.gz",
            albedo_neutrons,
        )
        
    primary_electron = data_dir / "PrimaryElectrons_BGOhit_Total.csv.gz"
    print(primary_electron)
    if not primary_electron.exists():
        fetch_wasabi_file(
            "COSI-SMEX/DC4/Data/Backgrounds/BGO/ASICS_massmodel_corrected/PrimaryElectrons_BGOhit_Total.csv.gz",
            primary_electron,
        )
        
    saa_protons_part1 = data_dir / "SAAprotons_BGOhit_Total.csv.gz"
    print(saa_protons_part1)
    if not saa_protons_part1.exists():
        fetch_wasabi_file(
            "COSI-SMEX/DC4/Data/Backgrounds/BGO/ASICS_massmodel_corrected/SAAprotons_BGOhit_Total.csv.gz",
            saa_protons_part1,
        )
   
        
    ori_file = data_dir / "DC4_final_530km_3_month_with_slew_1sbins_GalacticEarth_SAA.ori"
    print(ori_file)
    if not ori_file.exists():
        fetch_wasabi_file(
            "COSI-SMEX/DC4/Data/Orientation/DC4_final_530km_3_month_with_slew_1sbins_GalacticEarth_SAA.ori",
            ori_file,
        )


if __name__ == "__main__":
    main()