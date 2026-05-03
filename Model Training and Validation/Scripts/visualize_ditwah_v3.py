"""
Enhanced Ditwah Visualization v3 — Nov 2025 to Dec 2025
Dynamic Y-axis scaling per hazard so fluctuations are clearly visible.
Separate subplots per model for cleaner comparison.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.ndimage import uniform_filter1d

df = pd.read_csv('Dataset Separation/test_results.csv')
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] >= '2025-11-01'].copy()
dates = sorted(df['date'].unique())
date_nums = np.arange(len(dates))
date_labels = [d.strftime('%b\n%d') for d in dates]
tick_step = max(1, len(dates) // 14)

SEV_NUM = {'Normal': 0, 'Moderate': 1, 'Severe': 2, 'Extreme': 3}
SEV_ORDER = ['Normal', 'Moderate', 'Severe', 'Extreme']
HAZARDS = ['Flood', 'Landslide', 'Drought']

MODEL_CONFIG = [
    ('Actual',   None,        '#26C6DA', '-',  2.8, 0.95),
    ('XGBoost',  'xgboost',   '#AB47BC', '--', 1.8, 0.90),
    ('LightGBM', 'lightgbm',  '#42A5F5', '-.',  1.8, 0.90),
    ('Ensemble', 'ensemble',  '#FFA726', ':',  2.5, 0.95),
]

def get_col(h, key):
    return f'{h}_actual_d1' if key is None else f'{h}_{key}_pred_d1'

def daily_mean_sev(col):
    return np.array([
        df[df['date'] == d][col].map(SEV_NUM).dropna().mean()
        for d in dates
    ])

def daily_crisis_pct(col):
    """% divisions at Severe or Extreme"""
    return np.array([
        ((df[df['date'] == d][col].isin(['Severe', 'Extreme'])).sum()
         / max(1, (df['date'] == d).sum())) * 100
        for d in dates
    ])

def smooth(arr, w=3):
    """Light smoothing to reduce noise while keeping shape"""
    return uniform_filter1d(arr, size=w, mode='nearest')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Mean Severity — one row per hazard, 4 model lines each
# ═══════════════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(3, 1, figsize=(20, 16), facecolor='#0d1117')
fig1.suptitle('Sri Lanka — Mean Daily Disaster Severity: Cyclone Ditwah Period (Nov–Dec 2025)',
              fontsize=16, fontweight='bold', color='white', y=0.99)

for row, hazard in enumerate(HAZARDS):
    h = hazard.lower()
    ax = axes1[row]
    ax.set_facecolor('#161b22')

    all_vals = []
    for label, key, color, ls, lw, alpha in MODEL_CONFIG:
        col = get_col(h, key)
        raw = daily_mean_sev(col)
        vals = smooth(raw, w=2)
        ax.plot(date_nums, vals, color=color, linestyle=ls,
                linewidth=lw, label=label, alpha=alpha, zorder=3 if key is None else 2)
        # Draw raw with very low alpha for reference
        ax.plot(date_nums, raw, color=color, linestyle=ls,
                linewidth=0.5, alpha=0.25, zorder=1)
        all_vals.extend(raw[~np.isnan(raw)])

    # Dynamic Y scaling with padding
    ymin = max(0, np.nanmin(all_vals) - 0.1)
    ymax = min(3.0, np.nanmax(all_vals) + 0.2)
    if ymax - ymin < 0.3:   # if range is tiny, zoom in more
        ypad = 0.15
        ymin = max(0, np.nanmin(all_vals) - ypad)
        ymax = min(3.0, np.nanmax(all_vals) + ypad)
    ax.set_ylim(ymin, ymax)

    # Y ticks: map to severity labels within visible range
    sev_labels = ['Normal', 'Moderate', 'Severe', 'Extreme']
    visible_ticks = [i for i in range(4) if ymin - 0.05 <= i <= ymax + 0.05]
    ax.set_yticks(visible_ticks)
    ax.set_yticklabels([sev_labels[i] for i in visible_ticks],
                       fontsize=9, color='#cccccc', fontweight='bold')

    # Horizontal severity bands (clipped to visible range)
    band_colors = ['#1a3a1a', '#3a3000', '#3a1500', '#3a0000']
    for i, (bmin, bmax) in enumerate([(0, 1), (1, 2), (2, 3), (3, 3.5)]):
        clipped_min = max(bmin, ymin)
        clipped_max = min(bmax, ymax)
        if clipped_max > clipped_min:
            ax.axhspan(clipped_min, clipped_max, color=band_colors[i], alpha=0.35, zorder=0)

    # Annotate peak of actual
    actual_col = get_col(h, None)
    actual_raw = daily_mean_sev(actual_col)
    peak_idx = int(np.nanargmax(actual_raw))
    ax.axvline(peak_idx, color='#ff4444', linewidth=1.5, linestyle='--', alpha=0.7, zorder=4)
    peak_date = dates[peak_idx].strftime('%b %d')
    ax.annotate(f'Peak\n{peak_date}', xy=(peak_idx, actual_raw[peak_idx]),
                xytext=(peak_idx + max(2, len(dates) // 12),
                        actual_raw[peak_idx] - (ymax - ymin) * 0.15),
                color='#ff8888', fontsize=8, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#ff6666', lw=1.3))

    ax.set_xlim(0, len(dates) - 1)
    ax.set_xticks(date_nums[::tick_step])
    ax.set_xticklabels(date_labels[::tick_step], fontsize=8, color='#aaaaaa')
    ax.set_title(f'{hazard}', fontsize=13, fontweight='bold', color='white',
                 loc='left', pad=6)
    ax.set_ylabel('Mean Severity', fontsize=9, color='#aaaaaa')
    ax.legend(fontsize=9, facecolor='#0d1117', edgecolor='#30363d',
              labelcolor='white', loc='upper right', framealpha=0.92,
              ncol=4, columnspacing=1.2, handlelength=2.2)
    ax.grid(axis='y', color='#30363d', linewidth=0.8, zorder=0)
    ax.grid(axis='x', color='#21262d', linewidth=0.5, linestyle=':', zorder=0)
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.tick_params(colors='#555555', length=3)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('Ditwah_MeanSeverity_v3.png', dpi=160, bbox_inches='tight',
            facecolor=fig1.get_facecolor())
plt.close()
print("Saved: Ditwah_MeanSeverity_v3.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: % Crisis Divisions — one row per hazard, 4 model lines each
# ═══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(3, 1, figsize=(20, 16), facecolor='#0d1117')
fig2.suptitle('Sri Lanka — % Divisions at Severe/Extreme: Cyclone Ditwah Period (Nov–Dec 2025)',
              fontsize=16, fontweight='bold', color='white', y=0.99)

for row, hazard in enumerate(HAZARDS):
    h = hazard.lower()
    ax = axes2[row]
    ax.set_facecolor('#161b22')

    all_vals = []
    for label, key, color, ls, lw, alpha in MODEL_CONFIG:
        col = get_col(h, key)
        raw = daily_crisis_pct(col)
        vals = smooth(raw, w=2)
        # Shade under Actual
        if key is None:
            ax.fill_between(date_nums, vals, alpha=0.12, color=color, linewidth=0, zorder=1)
        ax.plot(date_nums, vals, color=color, linestyle=ls,
                linewidth=lw, label=label, alpha=alpha, zorder=3 if key is None else 2)
        ax.plot(date_nums, raw, color=color, linestyle=ls,
                linewidth=0.5, alpha=0.2, zorder=1)
        all_vals.extend(raw[~np.isnan(raw)])

    # Dynamic Y scaling
    ymin = 0
    ymax_data = np.nanmax(all_vals) if all_vals else 1
    ymax = min(100, ymax_data * 1.18 + 1)
    if ymax < 5:   # Very low range — zoom in
        ymax = max(5, ymax_data * 1.5 + 0.5)
    ax.set_ylim(ymin, ymax)

    # Shade crisis peak
    actual_col = get_col(h, None)
    actual_raw = daily_crisis_pct(actual_col)
    thresh = np.nanpercentile(actual_raw, 80)
    ax.fill_between(date_nums, 0, actual_raw,
                    where=actual_raw >= thresh,
                    color='#ff2222', alpha=0.15, linewidth=0, zorder=0,
                    label='_nolegend_')

    # Peak annotation
    peak_idx = int(np.nanargmax(actual_raw))
    peak_date = dates[peak_idx].strftime('%b %d')
    peak_val = actual_raw[peak_idx]
    ax.axvline(peak_idx, color='#ff4444', linewidth=1.5, linestyle='--', alpha=0.7, zorder=4)
    ax.annotate(f'Peak: {peak_val:.1f}%\n{peak_date}',
                xy=(peak_idx, peak_val),
                xytext=(peak_idx + max(2, len(dates) // 12), peak_val - ymax * 0.12),
                color='#ff8888', fontsize=8, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#ff6666', lw=1.3))

    # Y ticks auto-scaled
    tick_count = 5
    ytick_vals = np.linspace(0, ymax, tick_count)
    ax.set_yticks(ytick_vals)
    ax.set_yticklabels([f'{v:.0f}%' for v in ytick_vals],
                       fontsize=9, color='#cccccc')

    ax.set_xlim(0, len(dates) - 1)
    ax.set_xticks(date_nums[::tick_step])
    ax.set_xticklabels(date_labels[::tick_step], fontsize=8, color='#aaaaaa')
    ax.set_title(f'{hazard}', fontsize=13, fontweight='bold', color='white',
                 loc='left', pad=6)
    ax.set_ylabel('% of Divisions in Crisis', fontsize=9, color='#aaaaaa')
    ax.legend(fontsize=9, facecolor='#0d1117', edgecolor='#30363d',
              labelcolor='white', loc='upper right', framealpha=0.92,
              ncol=4, columnspacing=1.2, handlelength=2.2)
    ax.grid(axis='y', color='#30363d', linewidth=0.8, zorder=0)
    ax.grid(axis='x', color='#21262d', linewidth=0.5, linestyle=':', zorder=0)
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.tick_params(colors='#555555', length=3)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('Ditwah_CrisisPct_v3.png', dpi=160, bbox_inches='tight',
            facecolor=fig2.get_facecolor())
plt.close()
print("Saved: Ditwah_CrisisPct_v3.png")
