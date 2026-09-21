import matplotlib.pyplot as plt
import numpy as np

COLUMN_WIDTH = 3.5
COLUMN_HEIGHT = 2 * COLUMN_WIDTH / (1 + 5 ** 0.5)

PLOT_PARAMS = {
    'figure.figsize': (COLUMN_WIDTH, COLUMN_HEIGHT),
    'figure.dpi': 72,
    'font.size': 9,
    'font.family': 'serif',
    'font.serif': 'cm',
    'axes.titlesize': 9,
    'axes.linewidth': 0.5,
    'grid.alpha': 0.42,
    'grid.linewidth': 0.5,
    'legend.title_fontsize': 9,
    'legend.fontsize': 7,
    'lines.linewidth': 1.,
    'lines.markersize': 2,
    'text.usetex': False,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
}


AGENT_COLORS = {
    'DCF': '#a6cee3',
    'SR': '#1f78b4',
    'MAB': '#b2df8a',
    'H-MAB': '#33a02c',
    'F-Optimal': '#fb9a99',
    'T-Optimal': '#e31a1c',
}


def set_style() -> None:
    plt.rcParams.update(PLOT_PARAMS)


def get_cmap(n: int) -> plt.cm:
    return plt.cm.viridis(np.linspace(0., 0.8, n))
