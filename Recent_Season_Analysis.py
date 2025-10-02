# Purpose: Apply historical learning to 2024 data and produce multi-dimensional analysis


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

# Load 2024 data
print("Loading 2024 red zone data...")
df_2024 = pd.read_csv('redzone_DTDP_2024.csv')
print(f"✓ 2024 data loaded. Shape: {df_2024.shape}")

# Load historical references from Script 1
print("\nLoading historical references from Script 1...")
benchmarks = pd.read_csv('historical_benchmarks.csv')
optimal_plays_basic = pd.read_csv('optimal_play_lookup_basic.csv')
optimal_plays_detailed = pd.read_csv('optimal_play_lookup_detailed.csv')
df_historical = pd.read_csv('redzone_DTDP_2015_2023_enhanced.csv')

print(f" Historical benchmarks loaded")
print(f" Optimal play lookups loaded")
print(f" Historical enhanced data loaded for model training")

print("\n" + "="*80)
print("FEATURE ENGINEERING - 2024 DATA")
print("="*80)

def engineer_features(df):
    df['success'] = (df['epa'] > 0).astype(int)
    df['drive_touchdown'] = df.groupby(['game_id', 'drive'])['touchdown'].transform('max')
    df['is_goal_to_go'] = df['goal_to_go'].fillna(0).astype(int)
    df['yards_to_goal'] = df['yardline_100']
    df['down_distance_ratio'] = df['ydstogo'] / (df['down'] + 1)
    df['under_2min_half'] = (df['half_seconds_remaining'] <= 120).astype(int)
    df['under_2min_game'] = (df['game_seconds_remaining'] <= 120).astype(int)
    
    df['score_differential_binned'] = pd.cut(
        df['score_differential'], 
        bins=[-100, -14, -7, -3, 0, 3, 7, 14, 100],
        labels=['down_14plus', 'down_8to14', 'down_4to7', 'down_1to3', 
                'tied_up3', 'up_4to7', 'up_8to14', 'up_14plus']
    )
    
    df['field_position_bin'] = pd.cut(
        df['yardline_100'],
        bins=[0, 5, 10, 15, 20],
        labels=['inside_5', 'yards_6to10', 'yards_11to15', 'yards_16to20'],
        include_lowest=True
    )
    
    df['is_4th_quarter'] = (df['qtr'] == 4).astype(int)
    df['play_type_clean'] = df['play_type'].fillna('no_play')
    
    # Detailed play types
    df['pass_length_clean'] = df['pass_length'].fillna('none')
    df['pass_location_clean'] = df['pass_location'].fillna('none')
    df['run_location_clean'] = df['run_location'].fillna('none')
    df['run_gap_clean'] = df['run_gap'].fillna('none')
    
    def create_detailed_play_type(row):
        if row['play_type_clean'] == 'pass':
            length = row['pass_length_clean'] if row['pass_length_clean'] != 'none' else 'unknown'
            location = row['pass_location_clean'] if row['pass_location_clean'] != 'none' else 'unknown'
            return f"pass_{length}_{location}"
        elif row['play_type_clean'] == 'run':
            location = row['run_location_clean'] if row['run_location_clean'] != 'none' else 'unknown'
            gap = row['run_gap_clean'] if row['run_gap_clean'] != 'none' else 'unknown'
            return f"run_{location}_{gap}"
        else:
            return row['play_type_clean']
    
    df['detailed_play_type'] = df.apply(create_detailed_play_type, axis=1)
    df['posteam_timeouts'] = df['posteam_timeouts_remaining'].fillna(3)
    df['defteam_timeouts'] = df['defteam_timeouts_remaining'].fillna(3)
    
    return df

df_2024 = engineer_features(df_2024)
print(f"Features engineered for 2024 data")

# CALCULATE DTDP FOR 2024
print("\n" + "="*80)
print("CALCULATING DTDP FOR 2024")
print("="*80)

feature_cols = [
    'down', 'ydstogo', 'yardline_100', 'qtr',
    'score_differential', 'half_seconds_remaining', 'game_seconds_remaining',
    'is_goal_to_go', 'down_distance_ratio', 'under_2min_half', 'under_2min_game',
    'is_4th_quarter', 'wp', 'posteam_timeouts', 'defteam_timeouts'
]

# Train model on historical data
drive_first_plays_hist = df_historical.groupby(['game_id', 'drive']).first().reset_index()
modeling_data_hist = drive_first_plays_hist[feature_cols + ['drive_touchdown']].dropna()

X_hist = modeling_data_hist[feature_cols]
y_hist = modeling_data_hist['drive_touchdown']

dtdp_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_split=50,
    min_samples_leaf=25, random_state=42, n_jobs=-1
)
dtdp_model.fit(X_hist, y_hist)

# Predict DTDP for 2024
drive_first_plays_2024 = df_2024.groupby(['game_id', 'drive']).first().reset_index()
drive_first_plays_2024_clean = drive_first_plays_2024.dropna(subset=feature_cols)
drive_first_plays_2024_clean['DTDP'] = dtdp_model.predict_proba(
    drive_first_plays_2024_clean[feature_cols]
)[:, 1]

df_2024 = df_2024.merge(
    drive_first_plays_2024_clean[['game_id', 'drive', 'DTDP']],
    on=['game_id', 'drive'], how='left'
)

print(f"DTDP calculated for 2024")

print("\n" + "="*80)
print("CALCULATING CDEM FOR 2024")
print("="*80)

situation_cols = ['down', 'field_position_bin', 'score_differential_binned']

# Merge optimal play recommendations
df_2024 = df_2024.merge(optimal_plays_basic, on=situation_cols, how='left')
df_2024 = df_2024.merge(optimal_plays_detailed, on=situation_cols, how='left')

# Decision accuracy
df_2024['chose_optimal_play_basic'] = (
    df_2024['play_type_clean'] == df_2024['optimal_play_type_basic']
).astype(int)

df_2024['chose_optimal_play_detailed'] = (
    df_2024['detailed_play_type'] == df_2024['optimal_play_type_detailed']
).astype(int)

# Get expected EPA from historical data
situation_epa_basic = df_historical.groupby(
    situation_cols + ['play_type_clean']
)['epa'].mean().reset_index()
situation_epa_basic.columns = situation_cols + ['play_type_clean', 'expected_epa']

df_2024 = df_2024.merge(
    situation_epa_basic.rename(columns={'play_type_clean': 'optimal_play_type_basic',
                                         'expected_epa': 'optimal_expected_epa_basic'}),
    on=situation_cols + ['optimal_play_type_basic'], how='left'
)

situation_epa_detailed = df_historical.groupby(
    situation_cols + ['detailed_play_type']
)['epa'].mean().reset_index()
situation_epa_detailed.columns = situation_cols + ['detailed_play_type', 'expected_epa']

df_2024 = df_2024.merge(
    situation_epa_detailed.rename(columns={'detailed_play_type': 'optimal_play_type_detailed',
                                            'expected_epa': 'optimal_expected_epa_detailed'}),
    on=situation_cols + ['optimal_play_type_detailed'], how='left'
)

# Calculate CDEM components
df_2024['actual_play_epa'] = df_2024['epa'].fillna(0)
df_2024['epa_differential_basic'] = df_2024['actual_play_epa'] - df_2024['optimal_expected_epa_basic'].fillna(0)
df_2024['epa_differential_detailed'] = df_2024['actual_play_epa'] - df_2024['optimal_expected_epa_detailed'].fillna(0)

# Normalize the data using historical distribution
hist_mean_basic = df_historical['epa_differential_basic'].mean()
hist_std_basic = df_historical['epa_differential_basic'].std()
df_2024['epa_diff_norm_basic'] = (
    (df_2024['epa_differential_basic'] - hist_mean_basic) / hist_std_basic
).clip(-3, 3)

hist_mean_detailed = df_historical['epa_differential_detailed'].mean()
hist_std_detailed = df_historical['epa_differential_detailed'].std()
df_2024['epa_diff_norm_detailed'] = (
    (df_2024['epa_differential_detailed'] - hist_mean_detailed) / hist_std_detailed
).clip(-3, 3)

# Calculate CDEM scores
df_2024['CDEM_basic'] = (
    (df_2024['chose_optimal_play_basic'] * 40) + 
    ((df_2024['epa_diff_norm_basic'] + 3) / 6 * 60)
)

df_2024['CDEM_detailed'] = (
    (df_2024['chose_optimal_play_detailed'] * 40) + 
    ((df_2024['epa_diff_norm_detailed'] + 3) / 6 * 60)
)

df_2024['CDEM'] = 0.4 * df_2024['CDEM_basic'] + 0.6 * df_2024['CDEM_detailed']

print(f"CDEM calculated for 2024")


# Team Ranks

print("\n" + "="*80)
print("GENERATING TEAM RANKINGS")
print("="*80)

team_stats_2024 = df_2024.groupby('posteam').agg({
    'DTDP': 'mean',
    'CDEM': 'mean',
    'CDEM_basic': 'mean',
    'CDEM_detailed': 'mean',
    'chose_optimal_play_basic': 'mean',
    'chose_optimal_play_detailed': 'mean',
    'touchdown': 'sum',
    'play_id': 'count',
    'epa': 'mean',
    'success': 'mean',
    'yards_gained': 'mean'
}).reset_index()

team_stats_2024.columns = [
    'team', 'avg_DTDP', 'avg_CDEM', 'avg_CDEM_basic', 'avg_CDEM_detailed',
    'optimal_play_pct_basic', 'optimal_play_pct_detailed',
    'total_touchdowns', 'total_plays', 'avg_epa', 'success_rate', 'avg_yards_gained'
]

# Calculate deviations from historical benchmarks
team_stats_2024['DTDP_vs_hist'] = team_stats_2024['avg_DTDP'] - benchmarks['avg_DTDP'].values[0]
team_stats_2024['CDEM_vs_hist'] = team_stats_2024['avg_CDEM'] - benchmarks['avg_CDEM'].values[0]
team_stats_2024['optimal_basic_vs_hist'] = team_stats_2024['optimal_play_pct_basic'] - benchmarks['optimal_play_rate_basic'].values[0]
team_stats_2024['optimal_detailed_vs_hist'] = team_stats_2024['optimal_play_pct_detailed'] - benchmarks['optimal_play_rate_detailed'].values[0]

# Calculate efficiency index (composite metric)
team_stats_2024['efficiency_index'] = (
    0.35 * (team_stats_2024['avg_CDEM'] / 100) +
    0.25 * (team_stats_2024['optimal_play_pct_basic']) +
    0.25 * (team_stats_2024['optimal_play_pct_detailed']) +
    0.15 * ((team_stats_2024['avg_epa'] - team_stats_2024['avg_epa'].min()) / 
            (team_stats_2024['avg_epa'].max() - team_stats_2024['avg_epa'].min()))
)

# Rankings
team_stats_2024 = team_stats_2024.sort_values('efficiency_index', ascending=False).reset_index(drop=True)
team_stats_2024['efficiency_rank'] = range(1, len(team_stats_2024) + 1)
team_stats_2024['CDEM_rank'] = team_stats_2024['avg_CDEM'].rank(ascending=False).astype(int)
team_stats_2024['DTDP_rank'] = team_stats_2024['avg_DTDP'].rank(ascending=False).astype(int)

print("\nTop 10 Teams by Efficiency Index:")
print(team_stats_2024[['efficiency_rank', 'team', 'efficiency_index', 'avg_CDEM', 
                        'optimal_play_pct_basic', 'optimal_play_pct_detailed']].head(10).to_string(index=False))

print("\nBottom 5 Teams by Efficiency Index:")
print(team_stats_2024[['efficiency_rank', 'team', 'efficiency_index', 'avg_CDEM',
                        'optimal_play_pct_basic', 'optimal_play_pct_detailed']].tail(5).to_string(index=False))

# Save team rankings
team_stats_2024.to_csv('team_rankings_2024_comprehensive.csv', index=False)
print(f"\n✓ Team rankings saved: team_rankings_2024_comprehensive.csv")
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Create figure directory if needed
import os
os.makedirs('figures_2024', exist_ok=True)

# Historical averages for reference lines
hist_avg_cdem = benchmarks['avg_CDEM'].values[0]
hist_avg_dtdp = benchmarks['avg_DTDP'].values[0]
hist_optimal_basic = benchmarks['optimal_play_rate_basic'].values[0]
hist_optimal_detailed = benchmarks['optimal_play_rate_detailed'].values[0]

# ============== VISUALIZATION 1: CDEM vs DTDP (Decision Quality vs Opportunity) ==============
fig, ax = plt.subplots(figsize=(14, 10))

scatter = ax.scatter(
    team_stats_2024['avg_DTDP'],
    team_stats_2024['avg_CDEM'],
    s=team_stats_2024['total_plays'] * 2,
    c=team_stats_2024['efficiency_index'],
    cmap='RdYlGn',
    alpha=0.7,
    edgecolors='black',
    linewidth=1.5
)

# Add team labels
for idx, row in team_stats_2024.iterrows():
    ax.annotate(
        row['team'],
        (row['avg_DTDP'], row['avg_CDEM']),
        fontsize=9,
        fontweight='bold',
        ha='center'
    )

# Reference lines
ax.axhline(y=hist_avg_cdem, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Historical Avg CDEM ({hist_avg_cdem:.1f})')
ax.axvline(x=hist_avg_dtdp, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Historical Avg DTDP ({hist_avg_dtdp:.3f})')

# Quadrant labels
ax.text(0.95, 0.95, 'High Opportunity\nGood Decisions', transform=ax.transAxes,
        fontsize=11, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax.text(0.05, 0.95, 'Low Opportunity\nGood Decisions', transform=ax.transAxes,
        fontsize=11, verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.text(0.95, 0.05, 'High Opportunity\nPoor Decisions', transform=ax.transAxes,
        fontsize=11, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
ax.text(0.05, 0.05, 'Low Opportunity\nPoor Decisions', transform=ax.transAxes,
        fontsize=11, verticalalignment='bottom', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

ax.set_xlabel('Average DTDP (Drive Touchdown Probability)', fontsize=13, fontweight='bold')
ax.set_ylabel('Average CDEM (Decision Efficiency)', fontsize=13, fontweight='bold')
ax.set_title('NFL Red Zone Coaching Efficiency: Decision Quality vs Scoring Opportunity (2024)\nBubble size = Total plays', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.colorbar(scatter, ax=ax, label='Efficiency Index')
plt.tight_layout()
plt.savefig('figures_2024/01_CDEM_vs_DTDP.png', dpi=300, bbox_inches='tight')
print("✓ Visualization 1 saved: CDEM vs DTDP")
plt.close()

# ============== 2: Basic vs Detailed Decision Making ==============
fig, ax = plt.subplots(figsize=(14, 10))

# Scatter plot
scatter = ax.scatter(
    team_stats_2024['optimal_play_pct_basic'],
    team_stats_2024['optimal_play_pct_detailed'],
    s=team_stats_2024['total_plays'] * 2,
    c=team_stats_2024['avg_CDEM'],
    cmap='viridis',
    alpha=0.7,
    edgecolors='black',
    linewidth=1.5
)

# Team labels
for idx, row in team_stats_2024.iterrows():
    ax.annotate(
        row['team'],
        (row['optimal_play_pct_basic'], row['optimal_play_pct_detailed']),
        fontsize=9,
        fontweight='bold',
        ha='center'
    )

# Historical average lines
ax.axhline(y=hist_optimal_detailed, color='red', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Historical Avg Detailed ({hist_optimal_detailed:.1%})')
ax.axvline(x=hist_optimal_basic, color='blue', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Historical Avg Basic ({hist_optimal_basic:.1%})')

# Axis limits (zoomed region)
ax.set_xlim(0.2, 0.6)   # 20% to 60%
ax.set_ylim(0.0, 0.2)   # 0% to 20%

# Diagonal line through intersection of red & blue averages
x_intersect = hist_optimal_basic
y_intersect = hist_optimal_detailed
slope = y_intersect / x_intersect if x_intersect != 0 else 0

# Diagonal from (0,0) to far corner of zoomed x-axis
x_vals = [0, 0.6]  # use zoomed max x
y_vals = [0, slope * 0.6]
ax.plot(x_vals, y_vals, 'k--', alpha=0.7, linewidth=1.5)

# Labels and title
ax.set_xlabel('Optimal Play % - Basic (Run vs Pass)', fontsize=13, fontweight='bold')
ax.set_ylabel('Optimal Play % - Detailed (Direction & Depth)', fontsize=13, fontweight='bold')
ax.set_title('Strategic Decision Making: Basic vs Detailed Play Selection (2024)\nBubble size = Total plays',
             fontsize=15, fontweight='bold', pad=20)

# Format axes as percentages
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

# Legend only, no colorbar
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures_2024/02_Basic_vs_Detailed_Decisions_zoomed.png', dpi=300, bbox_inches='tight')
print("Visualization 2 saved: Basic vs Detailed Decisions (Zoomed)")
plt.close()

# ============== VISUALIZATION 3: Efficiency Index Rankings (Bar Chart) ==============
fig, ax = plt.subplots(figsize=(16, 12))

colors = ['darkgreen' if x > 0.5 else 'darkred' if x < 0.4 else 'goldenrod' 
          for x in team_stats_2024['efficiency_index']]

bars = ax.barh(
    range(len(team_stats_2024)),
    team_stats_2024['efficiency_index'],
    color=colors,
    alpha=0.7,
    edgecolor='black',
    linewidth=1.5
)

ax.set_yticks(range(len(team_stats_2024)))
ax.set_yticklabels(team_stats_2024['team'], fontsize=10)
ax.set_xlabel('Efficiency Index', fontsize=13, fontweight='bold')
ax.set_title('2024 NFL Red Zone Coaching Efficiency Rankings\nComposite of CDEM, Optimal Play Selection, and EPA',
             fontsize=15, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

# Add values on bars
for i, (idx, row) in enumerate(team_stats_2024.iterrows()):
    ax.text(
        row['efficiency_index'] + 0.01,
        i,
        f"{row['efficiency_index']:.3f}",
        va='center',
        fontsize=9,
        fontweight='bold'
    )

plt.tight_layout()
plt.savefig('figures_2024/03_Efficiency_Rankings.png', dpi=300, bbox_inches='tight')
print("✓ Visualization 3 saved: Efficiency Rankings")
plt.close()

# ============== VISUALIZATION 4: Multi-Metric Dashboard (4-Panel) ==============
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('2024 Red Zone Performance Dashboard: Multi-Dimensional Analysis', 
             fontsize=16, fontweight='bold', y=0.995)

# Panel 1: CDEM Distribution
ax1 = axes[0, 0]
ax1.hist(team_stats_2024['avg_CDEM'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
ax1.axvline(hist_avg_cdem, color='red', linestyle='--', linewidth=2, label='Historical Avg')
ax1.axvline(team_stats_2024['avg_CDEM'].median(), color='green', linestyle='-', linewidth=2, label='2024 Median')
ax1.set_xlabel('Average CDEM', fontsize=11, fontweight='bold')
ax1.set_ylabel('Number of Teams', fontsize=11, fontweight='bold')
ax1.set_title('Distribution of Decision Efficiency Scores', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Panel 2: EPA vs Success Rate
ax2 = axes[0, 1]
scatter2 = ax2.scatter(
    team_stats_2024['avg_epa'],
    team_stats_2024['success_rate'],
    s=team_stats_2024['total_plays'] * 2,
    c=team_stats_2024['avg_CDEM'],
    cmap='plasma',
    alpha=0.7,
    edgecolors='black',
    linewidth=1.5
)
for idx, row in team_stats_2024.head(10).iterrows():
    ax2.annotate(row['team'], (row['avg_epa'], row['success_rate']),
                 fontsize=8, ha='center')
ax2.set_xlabel('Average EPA per Play', fontsize=11, fontweight='bold')
ax2.set_ylabel('Success Rate', fontsize=11, fontweight='bold')
ax2.set_title('Offensive Efficiency: EPA vs Success Rate', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='CDEM')

# Panel 3: Touchdown Efficiency
ax3 = axes[1, 0]
td_rate = team_stats_2024['total_touchdowns'] / team_stats_2024['total_plays']
team_stats_2024['td_rate'] = td_rate
scatter3 = ax3.scatter(
    team_stats_2024['avg_DTDP'],
    team_stats_2024['td_rate'],
    s=team_stats_2024['total_plays'] * 2,
    c=team_stats_2024['avg_CDEM'],
    cmap='coolwarm',
    alpha=0.7,
    edgecolors='black',
    linewidth=1.5
)
ax3.axhline(y=benchmarks['td_rate'].values[0], color='red', linestyle='--', linewidth=2, alpha=0.7)
ax3.axvline(x=hist_avg_dtdp, color='blue', linestyle='--', linewidth=2, alpha=0.7)
for idx, row in team_stats_2024.head(10).iterrows():
    ax3.annotate(row['team'], (row['avg_DTDP'], row['td_rate']),
                 fontsize=8, ha='center')
ax3.set_xlabel('Average DTDP (Predicted TD Prob)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Actual TD Rate', fontsize=11, fontweight='bold')
ax3.set_title('Expected vs Actual Touchdown Conversion', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter3, ax=ax3, label='CDEM')

# Panel 4: Optimal Play Selection by Level
ax4 = axes[1, 1]
x = np.arange(len(team_stats_2024.head(15)))
width = 0.35
bars1 = ax4.barh(x - width/2, team_stats_2024.head(15)['optimal_play_pct_basic'], 
                 width, label='Basic (Run/Pass)', color='steelblue', alpha=0.8)
bars2 = ax4.barh(x + width/2, team_stats_2024.head(15)['optimal_play_pct_detailed'],
                 width, label='Detailed (Direction/Depth)', color='coral', alpha=0.8)
ax4.set_yticks(x)
ax4.set_yticklabels(team_stats_2024.head(15)['team'], fontsize=9)
ax4.set_xlabel('Optimal Play Selection Rate', fontsize=11, fontweight='bold')
ax4.set_title('Top 15 Teams: Decision Accuracy by Complexity', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(axis='x', alpha=0.3)
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax4.invert_yaxis()

plt.tight_layout()
plt.savefig('figures_2024/04_Multi_Metric_Dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Visualization 4 saved: Multi-Metric Dashboard")
plt.close()

# ============== VISUALIZATION 5: Deviation from Historical Norms ==============
fig, ax = plt.subplots(figsize=(16, 10))

x_pos = np.arange(len(team_stats_2024))
colors_dev = ['green' if x > 0 else 'red' for x in team_stats_2024['CDEM_vs_hist']]

bars = ax.barh(x_pos, team_stats_2024['CDEM_vs_hist'], color=colors_dev, alpha=0.7, edgecolor='black')

ax.set_yticks(x_pos)
ax.set_yticklabels(team_stats_2024['team'], fontsize=10)
ax.set_xlabel('CDEM Deviation from Historical Average (2015-2023)', fontsize=13, fontweight='bold')
ax.set_title('2024 Teams: Above or Below Historical Decision-Making Standards', 
             fontsize=15, fontweight='bold', pad=20)
ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

# Add value labels
for i, (idx, row) in enumerate(team_stats_2024.iterrows()):
    value = row['CDEM_vs_hist']
    ax.text(
        value + (0.5 if value > 0 else -0.5),
        i,
        f"{value:+.1f}",
        va='center',
        ha='left' if value > 0 else 'right',
        fontsize=9,
        fontweight='bold'
    )

plt.tight_layout()
plt.savefig('figures_2024/05_Historical_Deviation.png', dpi=300, bbox_inches='tight')
print("✓ Visualization 5 saved: Historical Deviation")
plt.close()

# ============== VISUALIZATION 6: Situational Analysis (Down & Distance) ==============
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Situational Decision-Making Analysis: Performance by Game Context', 
             fontsize=16, fontweight='bold', y=0.995)

# By Down
down_analysis = df_2024.groupby('down').agg({
    'CDEM': 'mean',
    'chose_optimal_play_basic': 'mean',
    'chose_optimal_play_detailed': 'mean',
    'epa': 'mean',
    'play_id': 'count'
}).reset_index()

ax1 = axes[0, 0]
x_downs = down_analysis['down']
width = 0.25
x_pos = np.arange(len(x_downs))
ax1.bar(x_pos - width, down_analysis['CDEM'], width, label='CDEM', color='steelblue', alpha=0.8)
ax1.bar(x_pos, down_analysis['chose_optimal_play_basic']*100, width, label='Optimal Basic %', color='coral', alpha=0.8)
ax1.bar(x_pos + width, down_analysis['chose_optimal_play_detailed']*100, width, label='Optimal Detailed %', color='lightgreen', alpha=0.8)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f"{int(d)}" for d in x_downs])
ax1.set_xlabel('Down', fontsize=11, fontweight='bold')
ax1.set_ylabel('Score / Percentage', fontsize=11, fontweight='bold')
ax1.set_title('Decision Quality by Down', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# By Field Position
field_pos_analysis = df_2024.groupby('field_position_bin').agg({
    'CDEM': 'mean',
    'chose_optimal_play_basic': 'mean',
    'chose_optimal_play_detailed': 'mean',
    'touchdown': 'mean',
    'play_id': 'count'
}).reset_index()

ax2 = axes[0, 1]
x_pos2 = np.arange(len(field_pos_analysis))
ax2.bar(x_pos2 - width, field_pos_analysis['CDEM'], width, label='CDEM', color='steelblue', alpha=0.8)
ax2.bar(x_pos2, field_pos_analysis['chose_optimal_play_basic']*100, width, label='Optimal Basic %', color='coral', alpha=0.8)
ax2.bar(x_pos2 + width, field_pos_analysis['touchdown']*100, width, label='TD Rate %', color='gold', alpha=0.8)
ax2.set_xticks(x_pos2)
ax2.set_xticklabels(field_pos_analysis['field_position_bin'], rotation=45, ha='right')
ax2.set_xlabel('Field Position', fontsize=11, fontweight='bold')
ax2.set_ylabel('Score / Percentage', fontsize=11, fontweight='bold')
ax2.set_title('Decision Quality by Field Position', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# By Score Differential
score_analysis = df_2024.groupby('score_differential_binned').agg({
    'CDEM': 'mean',
    'chose_optimal_play_basic': 'mean',
    'chose_optimal_play_detailed': 'mean',
    'play_id': 'count'
}).reset_index()

ax3 = axes[1, 0]
x_pos3 = np.arange(len(score_analysis))
ax3.plot(x_pos3, score_analysis['CDEM'], marker='o', linewidth=2, markersize=8, label='CDEM', color='darkblue')
ax3.plot(x_pos3, score_analysis['chose_optimal_play_basic']*100, marker='s', linewidth=2, markersize=8, label='Optimal Basic %', color='darkred')
ax3.plot(x_pos3, score_analysis['chose_optimal_play_detailed']*100, marker='^', linewidth=2, markersize=8, label='Optimal Detailed %', color='darkgreen')
ax3.set_xticks(x_pos3)
ax3.set_xticklabels(score_analysis['score_differential_binned'], rotation=45, ha='right', fontsize=9)
ax3.set_xlabel('Score Differential', fontsize=11, fontweight='bold')
ax3.set_ylabel('Score / Percentage', fontsize=11, fontweight='bold')
ax3.set_title('Decision Quality by Game Score Situation', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# By Quarter
quarter_analysis = df_2024.groupby('qtr').agg({
    'CDEM': 'mean',
    'chose_optimal_play_basic': 'mean',
    'chose_optimal_play_detailed': 'mean',
    'epa': 'mean',
    'play_id': 'count'
}).reset_index()

ax4 = axes[1, 1]
x_quarters = quarter_analysis['qtr']
x_pos4 = np.arange(len(x_quarters))
ax4.bar(x_pos4 - width, quarter_analysis['CDEM'], width, label='CDEM', color='purple', alpha=0.8)
ax4.bar(x_pos4, quarter_analysis['chose_optimal_play_basic']*100, width, label='Optimal Basic %', color='orange', alpha=0.8)
ax4.bar(x_pos4 + width, quarter_analysis['epa']*10, width, label='EPA x10', color='teal', alpha=0.8)
ax4.set_xticks(x_pos4)
ax4.set_xticklabels([f"Q{int(q)}" for q in x_quarters])
ax4.set_xlabel('Quarter', fontsize=11, fontweight='bold')
ax4.set_ylabel('Score / Metric', fontsize=11, fontweight='bold')
ax4.set_title('Decision Quality by Quarter', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures_2024/06_Situational_Analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualization 6 saved: Situational Analysis")
plt.close()

# ============== VISUALIZATION 7: Play Type Effectiveness Heatmap ==============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Pass effectiveness by length and location
pass_data = df_2024[df_2024['play_type_clean'] == 'pass'].copy()
pass_heatmap = pass_data.groupby(['pass_length_clean', 'pass_location_clean']).agg({
    'epa': 'mean',
    'play_id': 'count'
}).reset_index()
pass_pivot = pass_heatmap[pass_heatmap['play_id'] >= 10].pivot(
    index='pass_length_clean', columns='pass_location_clean', values='epa'
)

sns.heatmap(pass_pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
            ax=ax1, cbar_kws={'label': 'Average EPA'}, linewidths=1, linecolor='black')
ax1.set_title('Pass Effectiveness by Length and Location\n(EPA per play, min 10 attempts)',
              fontsize=13, fontweight='bold', pad=15)
ax1.set_xlabel('Pass Location', fontsize=11, fontweight='bold')
ax1.set_ylabel('Pass Length', fontsize=11, fontweight='bold')

# Run effectiveness by location and gap
run_data = df_2024[df_2024['play_type_clean'] == 'run'].copy()
run_heatmap = run_data.groupby(['run_location_clean', 'run_gap_clean']).agg({
    'epa': 'mean',
    'play_id': 'count'
}).reset_index()
run_pivot = run_heatmap[run_heatmap['play_id'] >= 10].pivot(
    index='run_location_clean', columns='run_gap_clean', values='epa'
)

sns.heatmap(run_pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
            ax=ax2, cbar_kws={'label': 'Average EPA'}, linewidths=1, linecolor='black')
ax2.set_title('Run Effectiveness by Location and Gap\n(EPA per play, min 10 attempts)',
              fontsize=13, fontweight='bold', pad=15)
ax2.set_xlabel('Run Gap', fontsize=11, fontweight='bold')
ax2.set_ylabel('Run Location', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('figures_2024/07_Play_Type_Effectiveness.png', dpi=300, bbox_inches='tight')
print("✓ Visualization 7 saved: Play Type Effectiveness Heatmaps")
plt.close()

print("\n" + "="*80)
print("GENERATING SUMMARY REPORT")
print("="*80)

summary_data = {
    'Metric': [
        'Average DTDP',
        'Average CDEM (Overall)',
        'Average CDEM (Basic)',
        'Average CDEM (Detailed)',
        'Optimal Play Rate (Basic)',
        'Optimal Play Rate (Detailed)',
        'Touchdown Rate',
        'Average EPA',
        'Success Rate'
    ],
    'Historical (2015-2023)': [
        f"{benchmarks['avg_DTDP'].values[0]:.3f}",
        f"{benchmarks['avg_CDEM'].values[0]:.2f}",
        f"{benchmarks['avg_CDEM_basic'].values[0]:.2f}",
        f"{benchmarks['avg_CDEM_detailed'].values[0]:.2f}",
        f"{benchmarks['optimal_play_rate_basic'].values[0]:.1%}",
        f"{benchmarks['optimal_play_rate_detailed'].values[0]:.1%}",
        f"{benchmarks['td_rate'].values[0]:.1%}",
        f"{benchmarks['avg_epa'].values[0]:.3f}",
        f"{benchmarks['success_rate'].values[0]:.1%}"
    ],
    '2024 Season': [
        f"{df_2024['DTDP'].mean():.3f}",
        f"{df_2024['CDEM'].mean():.2f}",
        f"{df_2024['CDEM_basic'].mean():.2f}",
        f"{df_2024['CDEM_detailed'].mean():.2f}",
        f"{df_2024['chose_optimal_play_basic'].mean():.1%}",
        f"{df_2024['chose_optimal_play_detailed'].mean():.1%}",
        f"{df_2024['touchdown'].mean():.1%}",
        f"{df_2024['epa'].mean():.3f}",
        f"{df_2024['success'].mean():.1%}"
    ],
    'Difference': [
        f"{df_2024['DTDP'].mean() - benchmarks['avg_DTDP'].values[0]:+.3f}",
        f"{df_2024['CDEM'].mean() - benchmarks['avg_CDEM'].values[0]:+.2f}",
        f"{df_2024['CDEM_basic'].mean() - benchmarks['avg_CDEM_basic'].values[0]:+.2f}",
        f"{df_2024['CDEM_detailed'].mean() - benchmarks['avg_CDEM_detailed'].values[0]:+.2f}",
        f"{(df_2024['chose_optimal_play_basic'].mean() - benchmarks['optimal_play_rate_basic'].values[0])*100:+.1f}pp",
        f"{(df_2024['chose_optimal_play_detailed'].mean() - benchmarks['optimal_play_rate_detailed'].values[0])*100:+.1f}pp",
        f"{(df_2024['touchdown'].mean() - benchmarks['td_rate'].values[0])*100:+.1f}pp",
        f"{df_2024['epa'].mean() - benchmarks['avg_epa'].values[0]:+.3f}",
        f"{(df_2024['success'].mean() - benchmarks['success_rate'].values[0])*100:+.1f}pp"
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('summary_report_2024.csv', index=False)

print("\n" + "="*80)
print("2024 vs HISTORICAL COMPARISON")
print("="*80)
print(summary_df.to_string(index=False))

# Save enhanced 2024 dataset
df_2024.to_csv('redzone_DTDP_2024_enhanced.csv', index=False)
print(f"\n✓ Enhanced 2024 dataset saved: redzone_DTDP_2024_enhanced.csv")

print("\n" + "="*80)
print("KEY INSIGHTS & FINDINGS")
print("="*80)

# Top performers
best_team = team_stats_2024.iloc[0]
worst_team = team_stats_2024.iloc[-1]

print(f"\nMOST EFFICIENT TEAM: {best_team['team']}")
print(f"   Efficiency Index: {best_team['efficiency_index']:.3f}")
print(f"   CDEM: {best_team['avg_CDEM']:.2f} (Rank #{best_team['CDEM_rank']})")
print(f"   Optimal Basic Play %: {best_team['optimal_play_pct_basic']:.1%}")
print(f"   Optimal Detailed Play %: {best_team['optimal_play_pct_detailed']:.1%}")

print(f"\nLEAST EFFICIENT TEAM: {worst_team['team']}")
print(f"   Efficiency Index: {worst_team['efficiency_index']:.3f}")
print(f"   CDEM: {worst_team['avg_CDEM']:.2f} (Rank #{worst_team['CDEM_rank']})")
print(f"   Optimal Basic Play %: {worst_team['optimal_play_pct_basic']:.1%}")
print(f"   Optimal Detailed Play %: {worst_team['optimal_play_pct_detailed']:.1%}")

# League-wide trends
print(f"\nLEAGUE-WIDE TRENDS:")
print(f"   2024 avg CDEM vs Historical: {df_2024['CDEM'].mean() - benchmarks['avg_CDEM'].values[0]:+.2f}")
print(f"   Teams above historical avg: {(team_stats_2024['CDEM_vs_hist'] > 0).sum()}/{len(team_stats_2024)}")
print(f"   Optimal basic play selection improved: {(df_2024['chose_optimal_play_basic'].mean() - benchmarks['optimal_play_rate_basic'].values[0])*100:+.1f} percentage points")
print(f"   Optimal detailed play selection improved: {(df_2024['chose_optimal_play_detailed'].mean() - benchmarks['optimal_play_rate_detailed'].values[0])*100:+.1f} percentage points")

# Correlations
corr_cdem_wins = df_2024.groupby('posteam').agg({
    'CDEM': 'mean',
    'touchdown': 'sum'
}).corr().iloc[0, 1]

print(f"\n🔗 CORRELATIONS:")
print(f"   CDEM vs Touchdowns: {corr_cdem_wins:.3f}")
print(f"   Efficiency Index vs EPA: {team_stats_2024[['efficiency_index', 'avg_epa']].corr().iloc[0, 1]:.3f}")

print("\n" + "="*80)
print("SCRIPT 2 COMPLETE - ALL OUTPUTS SAVED")
print("="*80)
print("\nGenerated Files:")
print("   1. redzone_DTDP_2024_enhanced.csv")
print("   2. team_rankings_2024_comprehensive.csv")
print("   3. summary_report_2024.csv")
print("\nGenerated Visualizations (figures_2024/):")
print("   1. 01_CDEM_vs_DTDP.png - Decision quality vs opportunity")
print("   2. 02_Basic_vs_Detailed_Decisions.png - Strategic complexity analysis")
print("   3. 03_Efficiency_Rankings.png - Complete team rankings")
print("   4. 04_Multi_Metric_Dashboard.png - 4-panel comprehensive view")
print("   5. 05_Historical_Deviation.png - Comparison to historical standards")
print("   6. 06_Situational_Analysis.png - Performance by game context")
print("   7. 07_Play_Type_Effectiveness.png - Pass/run effectiveness heatmaps")
print("\n" + "="*80)