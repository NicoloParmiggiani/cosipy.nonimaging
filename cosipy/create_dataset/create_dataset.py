"""
Dataset creator for COSI simulated GRB data
===========================================

This script builds pickle datasets from simulated GRB outputs, supporting three
processing modes selected via --mode:
- files_flat: iterate over a glob of input files and produce flat arrays
- healpix_flat: scan simulation folders organized by HEALPix pixel and produce flat arrays
- healpix_matrix: group entries by HEALPix pixel and produce a fixed-size matrix per pixel

Each dataset entry contains:
- spectrum: value parsed from GRB.source (.Spectrum line)
- flux: float parsed from GRB.source (.Flux line)
- coord: [theta, phi] parsed from the filename
- counts: six BGO counts in a fixed detector order

Inputs (flags):
- --mode: one of files_flat, healpix_flat, healpix_matrix
- --output: base path for the output pickle (also writes <output>_shared)
- --suffix: file suffix to match (default: .evt.lc)

Mode-specific flags:
- files_flat:
  - --glob: glob pattern to enumerate files (e.g., "/root/**/*.evt.lc")

- healpix_flat:
  - --root-dir: directory containing simulation subfolders named as theta_phi_seed
  - --nside: HEALPix nside value (integer)

- healpix_matrix:
  - --root-dir: directory containing simulation subfolders named as theta_phi_seed
  - --nside: HEALPix nside value (integer)
  - --entries-per-pixel: maximum number of entries per pixel (default: 10)

Notes:
- "shared" files are detected if the basename (without suffix) ends with "_shared" or contains a token "shared".
- Hidden files (starting with a dot) are ignored.
"""

import os
import sys
import glob
import pickle
import argparse
from typing import Dict, List, Tuple

import numpy as np
import healpy as hp


DETECTOR_NAMES: List[str] = [
    "bgo_z1[keV]", "bgo_z0[keV]",
    "bgo_x1[keV]", "bgo_x0[keV]",
    "bgo_y1[keV]", "bgo_y0[keV]",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset creation for COSI GRB simulations")

    parser.add_argument("--mode", choices=["files_flat", "healpix_flat", "healpix_matrix"], required=True,
                        help="Selection of behavior to use")
    parser.add_argument("--output", required=True, help="Base path for output pickle")
    parser.add_argument("--suffix", default=".evt.lc", help="File suffix to filter, e.g., .evt.lc")

    # files_flat
    parser.add_argument("--glob", dest="glob_pattern", help="Glob pattern of files (for files_flat mode)")

    # healpix_* modes
    parser.add_argument("--root-dir", dest="root_dir", help="Root directory with theta_phi_seed subfolders")
    parser.add_argument("--nside", type=int, help="HEALPix nside value (required for healpix modes)")

    # healpix_matrix extras
    parser.add_argument("--entries-per-pixel", type=int, default=10,
                        help="Maximum number of entries per pixel (healpix_matrix)")

    args = parser.parse_args()

    if args.mode == "files_flat":
        if not args.glob_pattern:
            parser.error("--glob is required when --mode=files_flat")
    elif args.mode in ("healpix_flat", "healpix_matrix"):
        if not args.root_dir or args.nside is None:
            parser.error("--root-dir and --nside are required for healpix modes")

    return args


def is_hidden_file(file_path: str) -> bool:
    return os.path.basename(file_path).startswith('.')


def remove_suffix_from_name(filename: str, suffix: str) -> str:
    if filename.endswith(suffix):
        return filename[: -len(suffix)]
    return filename


def detect_is_shared(basename_no_suffix: str) -> bool:
    if basename_no_suffix.endswith("_shared"):
        return True
    tokens = basename_no_suffix.split("_")
    return "shared" in tokens


def extract_theta_phi_from_filename(file_path: str, suffix: str) -> Tuple[str, str]:
    base = os.path.basename(file_path)
    base_no = remove_suffix_from_name(base, suffix)
    tokens = base_no.split("_")
    if len(tokens) < 2:
        return "", ""
    return tokens[0], tokens[1]


def parse_source_file(folder_path: str) -> Tuple[str, float]:
    source_path = os.path.join(folder_path, "GRB.source")
    spectrum_value: str = None
    flux_value: float = None

    try:
        with open(source_path, "r") as f:
            for line in f:
                line_stripped = line.strip()
                if ".Spectrum" in line_stripped:
                    try:
                        _, value = line_stripped.split(maxsplit=1)
                        spectrum_value = value
                    except Exception:
                        pass
                if ".Flux" in line_stripped:
                    try:
                        _, value = line_stripped.split(maxsplit=1)
                        flux_value = float(value)
                    except Exception:
                        pass
    except FileNotFoundError:
        # Keep None defaults; caller can handle
        pass

    return spectrum_value, (flux_value if flux_value is not None else 0.0)


def read_counts_from_file(file_path: str) -> List[int]:
    bgo_counts: List[int] = [0] * len(DETECTOR_NAMES)
    with open(file_path, "r") as f:
        for row in f:
            components = row.split()
            if len(components) >= 2:
                name = components[0]
                try:
                    value = int(components[1])
                except ValueError:
                    continue
                if name in DETECTOR_NAMES:
                    idx = DETECTOR_NAMES.index(name)
                    bgo_counts[idx] = value
    return bgo_counts


def process_one_file(file_path: str, suffix: str) -> Tuple[Dict, bool]:
    filename = os.path.basename(file_path)
    folder = os.path.dirname(file_path)

    base_no_suffix = remove_suffix_from_name(filename, suffix)
    is_shared = detect_is_shared(base_no_suffix)

    theta_str, phi_str = extract_theta_phi_from_filename(file_path, suffix)
    counts = read_counts_from_file(file_path)
    spectrum, flux = parse_source_file(folder)

    return {
        "spectrum": spectrum,
        "flux": float(flux),
        "coord": [theta_str, phi_str],
        "counts": counts,
    }, is_shared


def is_valid_sim_dir(folder: str) -> bool:
    if not os.path.isdir(folder):
        return False
    parts = os.path.basename(folder).split("_")
    if len(parts) != 3:
        return False
    try:
        theta = float(parts[0])
        phi = float(parts[1])
        seed = int(parts[2])
        if not (0.0 <= theta <= 180.0):
            return False
        if not (0.0 <= phi <= 360.0):
            return False
        _ = seed
        return True
    except ValueError:
        return False


def extract_theta_phi_from_folder(folder_name: str) -> Tuple[float, float]:
    try:
        theta_str, phi_str, _ = os.path.basename(folder_name).split("_")
        return float(theta_str), float(phi_str)
    except ValueError:
        return None, None


def run_files_flat(glob_pattern: str, suffix: str) -> Tuple[np.ndarray, np.ndarray]:
    dataset_list: List[Dict] = []
    dataset_shared_list: List[Dict] = []

    for file_path in glob.glob(glob_pattern, recursive=True):
        if is_hidden_file(file_path):
            continue
        entry, is_shared = process_one_file(file_path, suffix)
        if is_shared:
            dataset_shared_list.append(entry)
        else:
            dataset_list.append(entry)

        if len(dataset_list) % 500 == 0 and len(dataset_list) > 0:
            print(f"Processed {len(dataset_list)} non-shared files...")

    dataset = np.array(dataset_list, dtype=object)
    dataset_shared = np.array(dataset_shared_list, dtype=object)
    return dataset, dataset_shared


def run_healpix_flat(root_dir: str, suffix: str, nside: int) -> Tuple[np.ndarray, np.ndarray]:
    all_entries = glob.glob(os.path.join(root_dir, "*"))
    all_folders = [f for f in all_entries if is_valid_sim_dir(f)]

    indexed_folders: List[Tuple[int, str]] = []
    for folder in all_folders:
        theta_deg, phi_deg = extract_theta_phi_from_folder(folder)
        if theta_deg is None or phi_deg is None:
            continue
        try:
            theta = np.radians(theta_deg)
            phi = np.radians(phi_deg)
            ipix = hp.ang2pix(nside, theta, phi, nest=True)
            indexed_folders.append((ipix, folder))
        except Exception as ex:
            print(f"HEALPix index error in {folder}: {ex}")

    indexed_folders.sort(key=lambda x: x[0])
    ordered_folders = [f for _, f in indexed_folders]

    dataset_list: List[Dict] = []
    dataset_shared_list: List[Dict] = []

    for folder in ordered_folders:
        for file_path in glob.glob(os.path.join(folder, f"*{suffix}")):
            if is_hidden_file(file_path):
                continue
            entry, is_shared = process_one_file(file_path, suffix)
            if is_shared:
                dataset_shared_list.append(entry)
            else:
                dataset_list.append(entry)
            if len(dataset_list) % 500 == 0 and len(dataset_list) > 0:
                print(f"Processed {len(dataset_list)} non-shared files...")

    dataset = np.array(dataset_list, dtype=object)
    dataset_shared = np.array(dataset_shared_list, dtype=object)
    return dataset, dataset_shared


def run_healpix_matrix(root_dir: str, suffix: str, nside: int, entries_per_pixel: int) -> Tuple[np.ndarray, np.ndarray]:
    all_entries = glob.glob(os.path.join(root_dir, "*"))
    all_folders = [f for f in all_entries if is_valid_sim_dir(f)]

    pixel_to_folders: Dict[int, List[str]] = {}
    for folder in all_folders:
        theta_deg, phi_deg = extract_theta_phi_from_folder(folder)
        if theta_deg is None or phi_deg is None:
            continue
        try:
            theta = np.radians(theta_deg)
            phi = np.radians(phi_deg)
            ipix = hp.ang2pix(nside, theta, phi, nest=True)
            pixel_to_folders.setdefault(ipix, []).append(folder)
        except Exception as ex:
            print(f"HEALPix index error in {folder}: {ex}")

    ordered_pixels = sorted(pixel_to_folders.keys())
    num_pixels = len(ordered_pixels)
    print(f"Found {num_pixels} unique HEALPix pixels")

    dataset = np.empty((num_pixels, entries_per_pixel), dtype=object)
    dataset_shared = np.empty((num_pixels, entries_per_pixel), dtype=object)

    for pixel_idx, ipix in enumerate(ordered_pixels):
        folders = pixel_to_folders[ipix]
        pixel_entries: List[Dict] = []
        pixel_entries_shared: List[Dict] = []

        for folder in folders:
            for file_path in glob.glob(os.path.join(folder, f"*{suffix}")):
                if is_hidden_file(file_path):
                    continue
                entry, is_shared = process_one_file(file_path, suffix)
                if is_shared:
                    pixel_entries_shared.append(entry)
                else:
                    pixel_entries.append(entry)

        # Fill per-pixel up to entries_per_pixel
        for i in range(entries_per_pixel):
            dataset[pixel_idx, i] = pixel_entries[i] if i < len(pixel_entries) else None
            dataset_shared[pixel_idx, i] = pixel_entries_shared[i] if i < len(pixel_entries_shared) else None

        if pixel_idx % 100 == 0:
            print(f"Processed {pixel_idx} pixels...")

    return dataset, dataset_shared


def main() -> None:
    args = parse_args()

    if args.mode == "files_flat":
        dataset, dataset_shared = run_files_flat(args.glob_pattern, args.suffix)
    elif args.mode == "healpix_flat":
        dataset, dataset_shared = run_healpix_flat(args.root_dir, args.suffix, args.nside)
    elif args.mode == "healpix_matrix":
        dataset, dataset_shared = run_healpix_matrix(args.root_dir, args.suffix, args.nside, args.entries_per_pixel)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    with open(args.output, "wb") as f:
        pickle.dump(dataset, f)
    with open(args.output + "_shared", "wb") as f:
        pickle.dump(dataset_shared, f)

    # Minimal completion logs
    try:
        print(f"Done. Dataset shapes: dataset={getattr(dataset, 'shape', None)}, shared={getattr(dataset_shared, 'shape', None)}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
