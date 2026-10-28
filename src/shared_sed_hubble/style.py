"""Stable scientific figure style for shared-SED distance and Hubble results.

Reference bands are display metadata only. They are never inference priors.
"""
from typing import NamedTuple

import matplotlib.pyplot as plt

INK = "#171717"
GRAY = "#c6c6c6"
ORANGE = "#df9229"
BLUE = "#436ac4"
CYAN = "#28b8ca"
ERROR_COLOR = "#555555"
HOLLOW_SOURCES = frozenset({"Mrk509"})

HUBBLE_FIGSIZE = (12.2, 5.8)
HUBBLE_LAYOUT = dict(width_ratios=[1.18, 1], height_ratios=[3.05, 1.10],
                     left=.073, right=.985, bottom=.12, top=.977,
                     wspace=.25, hspace=.055)
SED_MAX_COLUMNS = 5
SED_PANEL_SIZE = (3.34, 3.3)
SED_LAYOUT = dict(left=.055, right=.99, bottom=.105, top=.975,
                  wspace=.25, hspace=.18)
SAVE_DPI = 250
HUBBLE_MARKER_SIZE = 5
SED_MARKER_SIZE = 4.3
H0_KDE_GRID_SIZE = 800
H0_KDE_BANDWIDTH = "scott"
REDSHIFT_TICKS = (.003, .005, .01, .02, .03, .05)
WAVELENGTH_TICKS = (2000, 3000, 5000)

# Preserve the accepted ten-source annotation placement; new sources use
# DEFAULT_LABEL_OFFSET and do not require additions to this mapping.
SOURCE_LABEL_OFFSETS = {
    "NGC4151_a": (7, 10), "NGC6814": (7, 7), "NGC2617_a": (-7, 10),
    "NGC5548": (7, -15), "Mrk335": (-7, 9), "Mrk279_b": (-7, -15),
    "Mrk817_a": (-7, 10), "Mrk509": (7, 8), "Mrk142": (-7, 10),
    "F9_a": (-2, -16),
}
DEFAULT_LABEL_OFFSET = (7, 8)


class ReferenceBand(NamedTuple):
    name: str
    mean: float
    sigma: float
    color: str
    linestyle: str
    label: str


H0_REFERENCES = (
    ReferenceBand("Planck", 67.4, .5, CYAN, ":", r"Planck: $67.4\pm0.5$"),
    ReferenceBand("SH0ES", 73.04, 1.04, BLUE, "--", r"SH0ES: $73.04\pm1.04$"),
)


def apply_style():
    """Apply the accepted serif, four-sided-axis figure defaults."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 12, "axes.labelsize": 14, "axes.linewidth": 1.1,
        "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 5, "ytick.major.size": 5,
        "xtick.minor.size": 2.8, "ytick.minor.size": 2.8,
        "xtick.major.width": 1, "ytick.major.width": 1,
        "xtick.minor.width": .8, "ytick.minor.width": .8,
        "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def apply_frame(ax):
    """Show all four spines and inward major/minor ticks without a grid."""
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(1.1)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def marker_face(source: str) -> str:
    """Keep the accepted hollow Mrk509 marker for its upstream ESS warning."""
    return "white" if source in HOLLOW_SOURCES else INK


def source_label(source: str) -> str:
    base = source.split("_", 1)[0]
    if base == "F9":
        return "Fairall 9"
    for prefix in ("Mrk", "NGC"):
        if base.startswith(prefix):
            return prefix + " " + base[len(prefix):]
    return base
