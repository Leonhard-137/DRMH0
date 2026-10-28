"""Project-local image-creator style shared by input and recovery figures.

The bundled mplstyle is the canonical base. Explicit project overrides retain
Times New Roman, black total transfer functions, and JPG/PDF delivery. No
personal skill path or user/system matplotlibrc is needed at runtime.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, NullFormatter

STYLE_PATH = Path(__file__).resolve().parent / "assets/image-creator.mplstyle"
COLORS = {
    "INK": "#222222", "PRIMARY": "#1F4E9E", "ALTERNATIVE": "#C43C39",
    "HIGHLIGHT": "#E69F00", "SECONDARY": "#009E73", "REFERENCE": "#4169E1",
    "UNCERTAINTY": "#B8B8B8", "GUIDE": "#D9D9D9",
}
COMPONENT_COLORS = (COLORS["PRIMARY"], COLORS["ALTERNATIVE"])
SHORT_DASH = (0, (5.0, 2.2))
REFERENCE_DASH = (0, (1.2, 1.8))


def configure_style():
    mpl.rcParams.update(mpl.rc_params_from_file(
        str(STYLE_PATH), use_default_template=False, fail_on_error=True))
    # Earlier explicit project choice takes precedence over the skill's STIX.
    for name in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
        path = Path("/mnt/c/Windows/Fonts") / name
        if path.is_file():
            font_manager.fontManager.addfont(path)
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
        "mathtext.cal": "Times New Roman:italic",
    })


def style_axis(ax, *, y_on_right=True, major_bins=5):
    # Keep scale-specific locators on log axes; every linear interval has
    # exactly three unlabelled minor ticks, including corner count axes.
    for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
        if scale == "linear":
            axis.set_minor_locator(AutoMinorLocator(4))
        axis.set_minor_formatter(NullFormatter())
    if ax.get_yscale() == "linear" and major_bins is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=major_bins))
    ax.yaxis.set_label_position("right" if y_on_right else "left")
    ax.tick_params(which="both", direction="in", top=True, bottom=True, left=True, right=True)
    ax.tick_params(axis="y", labelleft=not y_on_right, labelright=y_on_right)


def observations(ax, time, flux, error, *, label=None):
    """Dense gray capless errors behind opaque observed points."""
    ax.errorbar(time, flux, yerr=error, fmt="none", ecolor=COLORS["UNCERTAINTY"],
                elinewidth=.55, alpha=.55, capsize=0, zorder=2)
    ax.plot(time, flux, "o", color=COLORS["INK"], ms=2.3, mew=0,
            linestyle="none", label=label, zorder=4)


def save_figure(fig, output):
    """Keep the explicitly requested formats; vector PDF, 300-dpi JPG."""
    for extension in ("jpg", "pdf"):
        fig.savefig(output.with_suffix(f".{extension}"), dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
