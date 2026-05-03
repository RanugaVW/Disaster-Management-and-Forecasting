"""
Visualize Predicted vs Actual for Nov 2025+ (Ditwah period)
for Flood, Landslide, and Drought using XGBoost, LightGBM, Ensemble.
Shows national-level daily severity distribution (all 121 divisions aggregated).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import Counter

df = pd.read_csv('Dataset Separation/test_results.csv')
df['date'] = pd.to_datetime(df['date'])

# Filter: Nov 2025 onwards
df = df[df['date'] >= '2025-11-01'].copy()
dates = sorted(df['date'].unique())

SEV_ORDER = ['Normal', 'Moderate', 'Severe', 'Extreme']
SEV_COLORS = {
    'Normal':   '#4CAF50',
    'Moderate': '#FFC107',
    'Severe':   '#FF5722',
    'Extreme':  '#B71C1C',
}
SEV_NUM = {s: i for i, s in enumerate(SEV_ORDER)}

HAZARDS = ['Flood', 'Landslide', 'Drought']
MODELS  = [
    ('Actual',   None),
    ('XGBoost',  'xgboost'),
    ('LightGBM', 'lightgbm'),
    ('Ensemble', 'ensemble'),
]

def get_daily_severity_dist(df, hazard, col_suffix):
    """Return a dict of date -> weighted average severity (0-3 scale)"""
    daily_mean = []
    for d in dates:
        subset = df[df['date'] == d][col_suffix]
        nums = subset.map(SEV_NUM).dropna()
        daily_mean.append(nums.mean() if len(nums) > 0 else np.nan)
    return daily_mean

def get_daily_max_severity(df, hazard, col_suffix):
    """Return daily % of divisions at each severity level"""
    dist = {s: [] for s in SEV_ORDER}
    for d in dates:
        subset = df[df['date'] == d][col_suffix]
        counts = subset.value_counts()
        total = len(subset)
        for s in SEV_ORDER:
            dist[s].append(counts.get(s, 0) / total * 100)
    return dist

# ── FIGURE SETUP ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 28), facecolor='#0f0f1a')
fig.suptitle(
    'Sri Lanka Disaster Severity Forecast — November 2025 Onwards\n(Cyclone / Ditwah Event Period)',
    fontsize=20, fontweight='bold', color='white', y=0.98
)

# 3 rows (hazards) × 4 columns (actual + 3 models)
gs = GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.28,
              left=0.06, right=0.97, top=0.93, bottom=0.05)

date_nums = np.arange(len(dates))
date_labels = [d.strftime('%b %d') for d in dates]
tick_step = max(1, len(dates) // 8)

for row, hazard in enumerate(HAZARDS):
    h = hazard.lower()
    
    for col, (model_label, model_key) in enumerate(MODELS):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('#1a1a2e')
        
        if model_key is None:
            col_suffix = f'{h}_actual_d1'
        else:
            col_suffix = f'{h}_{model_key}_pred_d1'
        
        dist = get_daily_max_severity(df, hazard, col_suffix)
        
        # Stacked area chart
        bottom = np.zeros(len(dates))
        for sev in SEV_ORDER:
            vals = np.array(dist[sev])
            ax.fill_between(date_nums, bottom, bottom + vals,
                            color=SEV_COLORS[sev], alpha=0.85, linewidth=0)
            bottom += vals
        
        # Overlay daily mean severity line
        mean_sev = get_daily_severity_dist(df, hazard, col_suffix)
        ax2 = ax.twinx()
        ax2.plot(date_nums, mean_sev, color='white', linewidth=1.5,
                 linestyle='--', alpha=0.7, label='Mean severity')
        ax2.set_ylim(-0.2, 3.5)
        ax2.set_yticks([0, 1, 2, 3])
        ax2.set_yticklabels(['N', 'M', 'S', 'X'], color='#aaaaaa', fontsize=7)
        ax2.tick_params(axis='y', colors='#555555', length=2)
        ax2.spines['right'].set_color('#333355')
        ax2.spines['top'].set_color('#333355')
        ax2.spines['left'].set_color('#333355')
        ax2.spines['bottom'].set_color('#333355')
        
        # Styling
        title_color = {'Actual': '#80CBC4', 'XGBoost': '#CE93D8',
                       'LightGBM': '#90CAF9', 'Ensemble': '#FFD54F'}
        ax.set_title(f'{hazard} — {model_label}',
                     fontsize=10, fontweight='bold',
                     color=title_color.get(model_label, 'white'), pad=6)
        
        ax.set_xlim(0, len(dates)-1)
        ax.set_ylim(0, 100)
        ax.set_xticks(date_nums[::tick_step])
        ax.set_xticklabels(date_labels[::tick_step], rotation=45, ha='right',
                           fontsize=7, color='#aaaaaa')
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'],
                            fontsize=7, color='#aaaaaa')
        ax.tick_params(axis='both', colors='#555555', length=2)
        for spine in ax.spines.values():
            spine.set_color('#333355')
        
        if col == 0:
            ax.set_ylabel(f'% of 121 Divisions', fontsize=8, color='#aaaaaa')

# Legend
legend_patches = [
    mpatches.Patch(color=SEV_COLORS[s], label=s) for s in SEV_ORDER
]
line_patch = plt.Line2D([0], [0], color='white', linestyle='--',
                        linewidth=1.5, label='Mean Severity (right axis)')
fig.legend(handles=legend_patches + [line_patch],
           loc='lower center', ncol=5, fontsize=9,
           facecolor='#1a1a2e', edgecolor='#333355',
           labelcolor='white', framealpha=0.9,
           bbox_to_anchor=(0.5, 0.01))

out_path = 'Ditwah_Disaster_Forecast_Nov2025.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out_path}")
