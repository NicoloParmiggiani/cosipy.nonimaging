import argparse
import os
import pickle
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.stats import chi2

"""
Aitoff projection utility for HEALPix maps.

If --plot-mask=1, the input filename must end with the following numeric
suffix to encode geometry and chi2 thresholding info:

  _{theta_real}_{phi_real}_{theta_reco}_{phi_reco}_{min_chi2}.pkl

Example filename:
  areas_chi2_run57_mix_mega_shared_2438_51.318_28.828_55.771_28.125_0.5221.pkl

Notes:
- Angles are in degrees. Phi values in [0, 360) are converted to [-180, 180] for plotting.
- Maps are assumed to be in NESTED ordering.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot HEALPix map in Aitoff projection with optional chi2 masking"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input pickle file containing a 1D HEALPix map",
    )
    parser.add_argument(
        "--plot-mask",
        type=int,
        default=0,
        help="If 1, mask values >= min_chi2 + chi2.ppf(0.9,2) parsed from filename",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Figure title; defaults to the input filename",
    )
    parser.add_argument(
        "--downsample-nside",
        type=int,
        default=0,
        help="If >0, downsample the map to this nside and plot a second panel",
    )
    parser.add_argument(
        "--coord",
        default="G",
        help="Coordinate system passed to healpy (e.g., G)",
    )
    parser.add_argument(
        "--cmap",
        default="turbo",
        help="Colormap for the base map (e.g., turbo, viridis)",
    )
    parser.add_argument(
        "--mask-cmap",
        default="viridis",
        help="Colormap for the masked chi2 map",
    )
    parser.add_argument(
        "--cbar-label",
        default="",
        help="Colorbar label for the base map",
    )
    parser.add_argument(
        "--downsample-cbar-label",
        default=None,
        help="Colorbar label for the downsampled map (defaults to --cbar-label)",
    )
    parser.add_argument(
        "--save",
        default=None,
        help=(
            "If set, save figure(s) to this path. If downsampling or mask is requested, "
            "suffixes _nside<N> and _masked are appended to the base name."
        ),
    )
    return parser.parse_args()


def read_map(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        m = pickle.load(f)
    return np.array(m)


def _set_colorbar_label(label: str) -> None:
    if not label:
        return
    fig = plt.gcf()
    axes = fig.get_axes()
    main_ax = plt.gca()
    for ax in axes:
        if ax is not main_ax:
            ax.set_xlabel(label, fontsize=14)
            break


def plot_aitoff(
    m: np.ndarray,
    title: str,
    cmap: str,
    coord: str,
    cbar_label: str,
    badcolor: str = None,
) -> None:
    hp.projview(
        m,
        coord=[coord],
        projection_type="aitoff",
        graticule=True,
        graticule_labels=True,
        longitude_grid_spacing=60,
        latitude_grid_spacing=30,
        title=title,
        cmap=cmap,
        nest=True,
        unit="",
        badcolor=badcolor if badcolor else None,
        fontsize={
            "xlabel": 14,
            "ylabel": 14,
            "title": 16,
            "xtick_label": 14,
            "ytick_label": 14,
            "cbar_label": 14,
            "cbar_tick_label": 14,
        },
        override_plot_properties={
            "cbar_shrink": 0.8,
            "cbar_pad": 0.05,
            "cbar_label_pad": 5,
        },
    )
    # Improve x tick contrast and set colorbar label
    ax = plt.gca()
    ax.tick_params(axis="x", colors="white")
    _set_colorbar_label(cbar_label)


def parse_filename_for_mask(basename: str):
    """
    Parse theta/phi real/reco and min_chi2 from the filename suffix.

    Expected pattern:
        ..._{theta_real}_{phi_real}_{theta_reco}_{phi_reco}_{min_chi2}.pkl
    """
    stem = os.path.basename(basename)
    if not stem.endswith(".pkl"):
        raise ValueError("Input must be a .pkl file when --plot-mask=1")
    tokens = stem[:-4].split("_")
    if len(tokens) < 5:
        raise ValueError(
            "Filename must end with _thetaReal_phiReal_thetaReco_phiReco_minChi2.pkl"
        )
    try:
        min_chi2 = float(tokens[-1])
        theta_real = float(tokens[-5])
        phi_real = float(tokens[-4])
        theta_reco = float(tokens[-3])
        phi_reco = float(tokens[-2])
    except Exception as ex:
        raise ValueError("Could not parse numeric values from filename suffix") from ex

    # Map phi from [0, 360) to [-180, 180] for display
    if phi_real > 180:
        phi_real -= 360
    if phi_reco > 180:
        phi_reco -= 360
    return theta_real, phi_real, theta_reco, phi_reco, min_chi2


def plot_masked(
    m: np.ndarray,
    title: str,
    cmap: str,
    coord: str,
    input_path: str,
) -> None:
    theta_real, phi_real, theta_reco, phi_reco, min_chi2 = parse_filename_for_mask(
        input_path
    )
    # Mask values outside the 90% c.l. region for 2 d.o.f.
    limit = min_chi2 + chi2.ppf(0.9, 2)
    masked = np.ma.masked_where(m >= limit, m)

    plot_aitoff(
        masked,
        title=title,
        cmap=cmap,
        coord=coord,
        cbar_label="$\\chi^2$ value (90% c.l.)",
        badcolor="antiquewhite",
    )

    # Overlay true (magenta star) and reconstructed (red X) directions
    hp.newprojplot(
        theta=np.radians(theta_real),
        phi=np.radians(phi_real),
        marker="*",
        color="magenta",
        markersize=13,
    )
    hp.newprojplot(
        theta=np.radians(theta_reco),
        phi=np.radians(phi_reco),
        marker="X",
        color="red",
        markersize=12,
    )


def main() -> None:
    args = parse_args()

    title = args.title or os.path.basename(args.input)
    m = read_map(args.input)

    # Base map
    plot_aitoff(
        m,
        title=title,
        cmap=args.cmap,
        coord=args.coord,
        cbar_label=args.cbar_label,
    )
    if args.save:
        base, _ = os.path.splitext(args.save)
        plt.savefig(f"{base}.png", dpi=200, bbox_inches="tight")

    # Optional downsampled map
    if args.downsample_nside and args.downsample_nside > 0:
        m_down = hp.ud_grade(
            m,
            nside_out=args.downsample_nside,
            order_in="NESTED",
            order_out="NESTED",
            power=0,
        )
        plot_aitoff(
            m_down,
            title=f"{title} (nside={args.downsample_nside})",
            cmap=args.cmap,
            coord=args.coord,
            cbar_label=(
                args.downsample_cbar_label
                if args.downsample_cbar_label is not None
                else args.cbar_label
            ),
        )
        if args.save:
            base, _ = os.path.splitext(args.save)
            plt.savefig(
                f"{base}_nside{args.downsample_nside}.png",
                dpi=200,
                bbox_inches="tight",
            )

    # Optional masked chi2 map (requires filename suffix pattern)
    if args.plot_mask == 1:
        plot_masked(
            m,
            title=f"{title} (masked 90% c.l.)",
            cmap=args.mask_cmap,
            coord=args.coord,
            input_path=args.input,
        )
        if args.save:
            base, _ = os.path.splitext(args.save)
            plt.savefig(f"{base}_masked.png", dpi=200, bbox_inches="tight")

    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()