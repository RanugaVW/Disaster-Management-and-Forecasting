"""
Enhanced Ditwah Visualization — Nov 2025 to Dec 2025
Shows daily % of divisions at SEVERE + EXTREME severity (crisis indicator)
for Actual vs XGBoost vs LightGBM vs Ensemble, per hazard.
Clean dark premium design.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

df = pd.read_csv('Dataset Separation/test_results.csv')
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] >= '2025-11-01'].copy()
dates = sorted(df['date'].unique())
date_nums = np.arange(len(dates))
date_labels = [d.strftime('%b %d') for d in dates]
tick_step = max(1, len(dates) // 10)

SEV_NUM = {'Normal': 0, 'Moderate': 1, 'Severe': 2, 'Extreme': 3}
SEV_ORDER = ['Normal', 'Moderate', 'Severe', 'Extreme']

HAZARDS = ['Flood', 'Landslide', 'Drought']

MODEL_CONFIG = [
    ('Actual',   None,        '#80DEEA', '-',  2.2),
    ('XGBoost',  'xgboost',   '#CE93D8', '--', 1.8),
    ('LightGBM', 'lightgbm',  '#90CAF9', '-.',  1.8),
    ('Ensemble', 'ensemble',  '#FFD54F', ':',  2.2),
]

SEV_COLORS = {
    'Normal':   '#4CAF50',
    'Moderate': '#FFC107',
    'Severe':   '#FF5722',
    'Extreme':  '#B71C1C',
}

def get_daily_mean_sev(df, col):
    vals = []
    for d in dates:
        nums = df[df['date'] == d][col].map(SEV_NUM).dropna()
        vals.append(nums.mean() if len(nums) > 0 else np.nan)
    return np.array(vals)

def get_daily_crisis_pct(df, col):
    """% divisions at Severe or Extreme"""
    vals = []
    for d in dates:
        subset = df[df['date'] == d][col]
        total = len(subset)
        crisis = ((subset == 'Severe') | (subset == 'Extreme')).sum()
        vals.append(crisis / total * 100 if total > 0 else np.nan)
    return np.array(vals)

def get_daily_dist(df, col):
    dist = {s: [] for s in SEV_ORDER}
    for d in dates:
        subset = df[df['date'] == d][col]
        total = len(subset)
        for s in SEV_ORDER:
            dist[s].append((subset == s).sum() / total * 100 if total > 0 else 0)
    return dist

# ── FIGURE ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 20), facecolor='#0d1117')
fig.suptitle(
    '🌀  Sri Lanka Disaster Severity — Cyclone Ditwah Period  (Nov–Dec 2025)',
    fontsize=18, fontweight='bold', color='white', y=0.99,
    fontfamily='DejaVu Sans'
)

gs = GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.30,
              left=0.07, right=0.97, top=0.94, bottom=0.08)

for row, hazard in enumerate(HAZARDS):
    h = hazard.lower()

    # ── Left panel: Mean severity line comparison ─────────────────────────────
    ax1 = fig.add_subplot(gs[row, 0])
    ax1.set_facecolor('#161b22')

    for label, key, color, ls, lw in MODEL_CONFIG:
        col = f'{h}_actual_d1' if key is None else f'{h}_{key}_pred_d1'
        vals = get_daily_mean_sev(df, col)
        ax1.plot(date_nums, vals, color=color, linestyle=ls,
                 linewidth=lw, label=label, alpha=0.92)

    # Annotate peak
    actual_col = f'{h}_actual_d1'
    actual_vals = get_daily_mean_sev(df, actual_col)
    peak_idx = np.nanargmax(actual_vals)
    ax1.axvline(peak_idx, color='#ff4444', linewidth=1.2, linestyle='--', alpha=0.6)
    ax1.annotate('Peak\nSeverity', xy=(peak_idx, actual_vals[peak_idx]),
                 xytext=(peak_idx + max(1, len(dates)//15), actual_vals[peak_idx] - 0.3),
                 color='#ff7777', fontsize=8,
                 arrowprops=dict(arrowstyle='->', color='#ff7777', lw=1.2))

    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(['Normal', 'Moderate', 'Severe', 'Extreme'],
                         fontsize=8, color='#aaaaaa')
    ax1.set_ylim(-0.2, 3.5)
    ax1.set_xlim(0, len(dates)-1)
    ax1.set_xticks(date_nums[::tick_step])
    ax1.set_xticklabels(date_labels[::tick_step], rotation=40, ha='right',
                         fontsize=8, color='#aaaaaa')
    ax1.set_title(f'{hazard} — Mean Daily Severity (Actual vs Models)',
                  fontsize=11, fontweight='bold', color='white', pad=8)
    ax1.set_ylabel('Mean Severity Level', fontsize=9, color='#aaaaaa')
    ax1.legend(fontsize=8, facecolor='#0d1117', edgecolor='#333355',
               labelcolor='white', loc='upper right', framealpha=0.85)
    ax1.grid(axis='y', color='#2d333b', linewidth=0.7)
    ax1.grid(axis='x', color='#2d333b', linewidth=0.4, linestyle=':')
    for spine in ax1.spines.values():
        spine.set_color('#333355')
    ax1.tick_params(colors='#555555')

    # ── Right panel: % Crisis Divisions stacked area ──────────────────────────
    ax2 = fig.add_subplot(gs[row, 1])
    ax2.set_facecolor('#161b22')

    actual_crisis = get_daily_crisis_pct(df, f'{h}_actual_d1')
    ax2.fill_between(date_nums, actual_crisis, alpha=0.25, color='#80DEEA', linewidth=0)
    ax2.plot(date_nums, actual_crisis, color='#80DEEA', linewidth=2.2,
             label='Actual', linestyle='-')

    for label, key, color, ls, lw in MODEL_CONFIG[1:]:  # skip Actual
        col = f'{h}_{key}_pred_d1'
        crisis_vals = get_daily_crisis_pct(df, col)
        ax2.plot(date_nums, crisis_vals, color=color, linestyle=ls,
                 linewidth=lw, label=label, alpha=0.88)

    # Shade the known peak crisis window (top 10% worst days)
    thresh = np.nanpercentile(actual_crisis, 85)
    ax2.fill_between(date_nums, 0, actual_crisis,
                     where=actual_crisis >= thresh,
                     color='#ff4444', alpha=0.18, linewidth=0, label='_nolegend_')

    ax2.set_xlim(0, len(dates)-1)
    ax2.set_ylim(0, 105)
    ax2.set_xticks(date_nums[::tick_step])
    ax2.set_xticklabels(date_labels[::tick_step], rotation=40, ha='right',
                         fontsize=8, color='#aaaaaa')
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'],
                         fontsize=8, color='#aaaaaa')
    ax2.set_title(f'{hazard} — % Divisions at Severe/Extreme (Crisis Alert)',
                  fontsize=11, fontweight='bold', color='white', pad=8)
    ax2.set_ylabel('% of 121 Divisions in Crisis', fontsize=9, color='#aaaaaa')
    ax2.legend(fontsize=8, facecolor='#0d1117', edgecolor='#333355',
               labelcolor='white', loc='upper right', framealpha=0.85)
    ax2.grid(axis='y', color='#2d333b', linewidth=0.7)
    ax2.grid(axis='x', color='#2d333b', linewidth=0.4, linestyle=':')
    for spine in ax2.spines.values():
        spine.set_color('#333355')
    ax2.tick_params(colors='#555555')

# Footer annotation
fig.text(0.5, 0.01,
         'Left: Mean national severity score (0=Normal → 3=Extreme)  |  '
         'Right: % of 121 Sri Lankan divisions flagged as Severe or Extreme  |  '
         'Red shading = top 15% worst days during the Ditwah event window',
         ha='center', fontsize=8, color='#666677', style='italic')

out_path = 'Ditwah_Disaster_Forecast_Nov2025_v2.png'
plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out_path}")
