"""Plot the exact unit-area BLR kernel recorded for the fast-driver injection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_style import configure_style, style_axis, save_figure, COLORS

HERE = Path(__file__).resolve().parent
CASE = HERE / 'cases_fast_driver/disk_blr_uonly_gp_tauI0p5'
p = json.loads((CASE / 'config/generation.json').read_text())['parameters']
onset, theta = p['blr_onset_days'], p['blr_theta_days']
t = np.linspace(-10, 100, 11001)
x = np.maximum(t - onset, 0)
b = x / theta**2 * np.exp(-x / theta)
configure_style()
fig, ax = plt.subplots(figsize=(7.2, 4.1))
ax.plot(t, b, color=COLORS['REFERENCE'], lw=1.8, label='Input BLR (unit area)')
ax.set(xlim=(-10, 100), ylim=(0, 0.06), xlabel=r'$\tau$ (days)', ylabel=r'$B(\tau)$ (day$^{-1}$)')
ax.legend(frameon=False)
style_axis(ax, y_on_right=False)
fig.tight_layout()
out = CASE / 'results/input_blr_transfer_function'
save_figure(fig, out)
print(out.with_suffix('.jpg'))
print(f'onset={onset:g}, peak={onset+theta:g}, centroid={onset+2*theta:g}, SD={np.sqrt(2)*theta:g}, response_blr_gain={p["response_blr_gain"]:g}')
