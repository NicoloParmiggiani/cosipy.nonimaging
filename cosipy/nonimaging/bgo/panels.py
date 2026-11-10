"""
Names and ordering of the six ACS (BGO) panels.

``PANEL_NAMES`` is the order of the COUNT vector in the ACS_LC FITS
extension (the ASIC order SCBA_A0, SCBA_A1, SCBB_A0, SCBB_A1, SCBC_A0,
SCBC_A1), and the only order used in memory: dictionaries, plots, FITS
read and write.

Counts are passed to the localizers as dicts panel -> value. Each localizer
converts them with ``to_array`` to the panel order of its own IRF or LUT.
"""

import numpy as np

PANEL_NAMES = (
    "ACS-A-Z1",
    "ACS-A-Z0",
    "ACS-B-Y1",
    "ACS-B-Y0",
    "ACS-C-X1",
    "ACS-C-X0",
)

# Short keys still used by background NPZ files and older simulation CSVs
LEGACY_PANEL_KEYS = {
    "ACS-A-Z1": "z1",
    "ACS-A-Z0": "z0",
    "ACS-B-Y1": "y1",
    "ACS-B-Y0": "y0",
    "ACS-C-X1": "x1",
    "ACS-C-X0": "x0",
}

_FROM_LEGACY = {short: name for name, short in LEGACY_PANEL_KEYS.items()}


def panel_name(name):
    """Panel name in PANEL_NAMES, given either that name or the legacy short key."""
    if name in LEGACY_PANEL_KEYS:
        return name
    try:
        return _FROM_LEGACY[name]
    except KeyError:
        raise KeyError(f"Unknown panel: {name!r}") from None


def to_array(counts, order):
    """Values of a dict panel -> value as a float array in ``order``.

    Keys can be the names in PANEL_NAMES or the legacy short keys.
    """
    if not hasattr(counts, "keys"):
        raise TypeError(
            f"Counts must be a dict panel -> value with keys {PANEL_NAMES}, "
            f"got {type(counts).__name__}"
        )
    by_name = {panel_name(k): v for k, v in counts.items()}
    missing = [p for p in order if p not in by_name]
    if missing:
        raise KeyError(f"Missing panels: {missing}")
    return np.array([by_name[p] for p in order], dtype=float)
