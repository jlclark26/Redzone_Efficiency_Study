
# Script 1: NFL Red Zone Analysis - Historical Data (2015-2023)
# Purpose: Build foundational dataset for DTDP and CDEM modeling

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SCRIPT 1: HISTORICAL RED ZONE ANALYSIS (2015-2023)")
print("="*80)

print("\nLoading historical red zone data (2015-2023)...")
df = pd.read_csv('redzone_DTDP_2015_2023.csv')

print(f"Data loaded successfully. Shape: {df.shape}")
print(f"Seasons included: {sorted(df['season'].unique())}")

# Machine learning: FEATURE ENGINEERING

print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

# Create success metric (EPA > 0 is considered successful)
df['success'] = (df['epa'] > 0).astype(int)

# Drive-level touchdown indicator
df['drive_touchdown'] = df.groupby(['game_id', 'drive'])['touchdown'].transform('max')

# Situational features
df['is_goal_to_go'] = df['goal_to_go'].fillna(0).astype(int)
df['yards_to_goal'] = df['yardline_100']
df['down_distance_ratio'] = df['ydstogo'] / (df['down'] + 1)

# Time pressure features
df['under_2min_half'] = (df['half_seconds_remaining'] <= 120).astype(int)
df['under_2min_game'] = (df['game_seconds_remaining'] <= 120).astype(int)

# Score situation bins
df['score_differential_binned'] = pd.cut(
    df['score_differential'], 
    bins=[-100, -14, -7, -3, 0, 3, 7, 14, 100],
    labels=['down_14plus', 'down_8to14', 'down_4to7', 'down_1to3', 
            'tied_up3', 'up_4to7', 'up_8to14', 'up_14plus']
)

# Field position bins
df['field_position_bin'] = pd.cut(
    df['yardline_100'],
    bins=[0, 5, 10, 15, 20],
    labels=['inside_5', 'yards_6to10', 'yards_11to15', 'yards_16to20'],
    include_lowest=True
)

# Quarter context
df['is_4th_quarter'] = (df['qtr'] == 4).astype(int)

# Play type encoding
df['play_type_clean'] = df['play_type'].fillna('no_play')

# Create detailed play type with direction and depth
df['pass_length_clean'] = df['pass_length'].fillna('none')
df['pass_location_clean'] = df['pass_location'].fillna('none')
df['run_location_clean'] = df['run_location'].fillna('none')
df['run_gap_clean'] = df['run_gap'].fillna('none')

# Detailed play descriptor combining type, direction, and depth
def create_detailed_play_type(row):
    if row['play_type_clean'] == 'pass':
        # For passes: pass_length_location (e.g., "pass_short_left")
        length = row['pass_length_clean'] if row['pass_length_clean'] != 'none' else 'unknown'
        location = row['pass_location_clean'] if row['pass_location_clean'] != 'none' else 'unknown'
        return f"pass_{length}_{location}"
    elif row['play_type_clean'] == 'run':
        # For runs: run_location_gap (e.g., "run_left_end")
        location = row['run_location_clean'] if row['run_location_clean'] != 'none' else 'unknown'
        gap = row['run_gap_clean'] if row['run_gap_clean'] != 'none' else 'unknown'
        return f"run_{location}_{gap}"
    else:
        return row['play_type_clean']

df['detailed_play_type'] = df.apply(create_detailed_play_type, axis=1)

# Timeout situation
df['posteam_timeouts'] = df['posteam_timeouts_remaining'].fillna(3)
df['defteam_timeouts'] = df['defteam_timeouts_remaining'].fillna(3)

print(f"✓ Created drive-level touchdown indicators")
print(f"✓ Engineered situational features")
print(f"✓ Created time pressure indicators")
print(f"✓ Created success metric (EPA > 0)")
print(f"✓ Created detailed play types (direction + depth)")

# Display sample of detailed play types
print(f"\nSample of detailed play types:")
print(df['detailed_play_type'].value_counts().head(10))

#BUILD DTDP Mod
print("\n" + "="*80)
print("BUILDING DTDP MODEL")
print("="*80)

# Features for DTDP modeling
feature_cols = [
    'down', 'ydstogo', 'yardline_100', 'qtr',
    'score_differential', 'half_seconds_remaining', 'game_seconds_remaining',
    'is_goal_to_go', 'down_distance_ratio', 'under_2min_half', 'under_2min_game',
    'is_4th_quarter', 'wp', 'posteam_timeouts', 'defteam_timeouts'
]

# Get first play of each drive
drive_first_plays = df.groupby(['game_id', 'drive']).first().reset_index()

# Prepare modeling dataset
modeling_data = drive_first_plays[feature_cols + ['drive_touchdown']].dropna()

print(f"Modeling dataset: {modeling_data.shape[0]} drives")
print(f"Touchdown rate: {modeling_data['drive_touchdown'].mean():.3f}")

# Split data
X = modeling_data[feature_cols]
y = modeling_data['drive_touchdown']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train DTDP model
print("\nTraining Random Forest for DTDP...")
dtdp_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=50,
    min_samples_leaf=25,
    random_state=42,
    n_jobs=-1
)

dtdp_model.fit(X_train, y_train)

train_score = dtdp_model.score(X_train, y_train)
test_score = dtdp_model.score(X_test, y_test)

print(f"✓ DTDP Model trained")
print(f"  Train accuracy: {train_score:.3f}")
print(f"  Test accuracy: {test_score:.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': dtdp_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 5 Most Important Features for DTDP:")
print(feature_importance.head().to_string(index=False))

# Generate DTDP predictions
drive_first_plays_clean = drive_first_plays.dropna(subset=feature_cols)
drive_first_plays_clean['DTDP'] = dtdp_model.predict_proba(
    drive_first_plays_clean[feature_cols]
)[:, 1]

# Merge DTDP back to main dataframe
df = df.merge(
    drive_first_plays_clean[['game_id', 'drive', 'DTDP']],
    on=['game_id', 'drive'],
    how='left'
)

print(f"✓ DTDP calculated for {df['DTDP'].notna().sum()} plays")


# ANALYZE OPTIMAL PLAY TYPES (DETAILED)
print("\n" + "="*80)
print("ANALYZING OPTIMAL PLAY TYPES (WITH DIRECTION & DEPTH)")
print("="*80)

situation_cols = ['down', 'field_position_bin', 'score_differential_binned']

# Analyze at TWO levels: basic and detailed

# LEVEL 1: Basic play type (run vs pass)
print("\n--- LEVEL 1: Basic Play Type Analysis ---")
play_type_basic = df.groupby(situation_cols + ['play_type_clean']).agg({
    'touchdown': 'sum',
    'play_id': 'count',
    'epa': 'mean',
    'success': 'mean'
}).reset_index()

play_type_basic.columns = situation_cols + ['play_type_clean', 'td_count', 
                                              'play_count', 'avg_epa', 'success_rate']

play_type_basic['td_probability'] = play_type_basic['td_count'] / play_type_basic['play_count']

# LEVEL 2: Detailed play type (with direction and depth)
print("--- LEVEL 2: Detailed Play Type Analysis (Direction + Depth) ---")
play_type_detailed = df.groupby(situation_cols + ['detailed_play_type']).agg({
    'touchdown': 'sum',
    'play_id': 'count',
    'epa': 'mean',
    'success': 'mean'
}).reset_index()

play_type_detailed.columns = situation_cols + ['detailed_play_type', 'td_count', 
                                                 'play_count', 'avg_epa', 'success_rate']

play_type_detailed['td_probability'] = play_type_detailed['td_count'] / play_type_detailed['play_count']

# Composite score for both levels
for pt_df in [play_type_basic, play_type_detailed]:
    epa_min = pt_df['avg_epa'].min()
    epa_max = pt_df['avg_epa'].max()
    pt_df['epa_normalized'] = (pt_df['avg_epa'] - epa_min) / (epa_max - epa_min)
    pt_df['composite_score'] = (
        0.5 * pt_df['td_probability'] + 
        0.3 * pt_df['epa_normalized'] +
        0.2 * pt_df['success_rate']
    )

# Filter for minimum sample size
play_type_basic_filtered = play_type_basic[play_type_basic['play_count'] >= 10].copy()
play_type_detailed_filtered = play_type_detailed[play_type_detailed['play_count'] >= 5].copy()

# Identify optimal basic play type
optimal_plays_basic = play_type_basic_filtered.loc[
    play_type_basic_filtered.groupby(situation_cols)['composite_score'].idxmax()
][situation_cols + ['play_type_clean', 'td_probability', 'avg_epa', 'composite_score']].rename(
    columns={
        'play_type_clean': 'optimal_play_type_basic', 
        'td_probability': 'optimal_td_prob_basic',
        'avg_epa': 'optimal_epa_basic',
        'composite_score': 'optimal_score_basic'
    }
)

# Identify optimal detailed play type
optimal_plays_detailed = play_type_detailed_filtered.loc[
    play_type_detailed_filtered.groupby(situation_cols)['composite_score'].idxmax()
][situation_cols + ['detailed_play_type', 'td_probability', 'avg_epa', 'composite_score']].rename(
    columns={
        'detailed_play_type': 'optimal_play_type_detailed', 
        'td_probability': 'optimal_td_prob_detailed',
        'avg_epa': 'optimal_epa_detailed',
        'composite_score': 'optimal_score_detailed'
    }
)

print(f"✓ Identified optimal BASIC play types for {len(optimal_plays_basic)} situations")
print(f"✓ Identified optimal DETAILED play types for {len(optimal_plays_detailed)} situations")

# Merge both levels
df = df.merge(optimal_plays_basic, on=situation_cols, how='left')
df = df.merge(optimal_plays_detailed, on=situation_cols, how='left')

# Show example of detailed recommendations
print("\nExample: 3rd down, inside 5 yards, tied game")
example = optimal_plays_detailed[
    (optimal_plays_detailed['down'] == 3) & 
    (optimal_plays_detailed['field_position_bin'] == 'inside_5') &
    (optimal_plays_detailed['score_differential_binned'] == 'tied_up3')
]
if len(example) > 0:
    print(example[['optimal_play_type_detailed', 'optimal_td_prob_detailed', 'optimal_epa_detailed']].to_string(index=False))

# CALCULATE CDEM (TWO LEVELS)
print("\n" + "="*80)
print("CALCULATING CDEM (BASIC & DETAILED)")
print("="*80)

# Actual play performance
df['actual_play_epa'] = df['epa'].fillna(0)

# LEVEL 1: Basic CDEM (did coach choose run vs pass correctly?)
df['chose_optimal_play_basic'] = (
    df['play_type_clean'] == df['optimal_play_type_basic']
).astype(int)

# LEVEL 2: Detailed CDEM (did coach choose correct direction/depth?)
df['chose_optimal_play_detailed'] = (
    df['detailed_play_type'] == df['optimal_play_type_detailed']
).astype(int)

# Get expected EPA for basic play types
situation_epa_basic = df.groupby(situation_cols + ['play_type_clean'])['epa'].mean().reset_index()
situation_epa_basic.columns = situation_cols + ['play_type_clean', 'expected_epa']

df = df.merge(
    situation_epa_basic.rename(columns={'play_type_clean': 'optimal_play_type_basic', 
                                         'expected_epa': 'optimal_expected_epa_basic'}),
    on=situation_cols + ['optimal_play_type_basic'],
    how='left'
)

# Get expected EPA for detailed play types
situation_epa_detailed = df.groupby(situation_cols + ['detailed_play_type'])['epa'].mean().reset_index()
situation_epa_detailed.columns = situation_cols + ['detailed_play_type', 'expected_epa']

df = df.merge(
    situation_epa_detailed.rename(columns={'detailed_play_type': 'optimal_play_type_detailed', 
                                            'expected_epa': 'optimal_expected_epa_detailed'}),
    on=situation_cols + ['optimal_play_type_detailed'],
    how='left'
)

# EPA differentials
df['epa_differential_basic'] = df['actual_play_epa'] - df['optimal_expected_epa_basic'].fillna(0)
df['epa_differential_detailed'] = df['actual_play_epa'] - df['optimal_expected_epa_detailed'].fillna(0)

# Normalize EPA differentials
epa_mean_basic = df['epa_differential_basic'].mean()
epa_std_basic = df['epa_differential_basic'].std()
df['epa_diff_norm_basic'] = (
    (df['epa_differential_basic'] - epa_mean_basic) / epa_std_basic
).clip(-3, 3)

epa_mean_detailed = df['epa_differential_detailed'].mean()
epa_std_detailed = df['epa_differential_detailed'].std()
df['epa_diff_norm_detailed'] = (
    (df['epa_differential_detailed'] - epa_mean_detailed) / epa_std_detailed
).clip(-3, 3)

# Calculate CDEM scores (0-100 scale)
# Basic CDEM: Did they choose run vs pass correctly?
df['CDEM_basic'] = (
    (df['chose_optimal_play_basic'] * 40) + 
    ((df['epa_diff_norm_basic'] + 3) / 6 * 60)
)

# Detailed CDEM: Did they choose the right direction and depth?
df['CDEM_detailed'] = (
    (df['chose_optimal_play_detailed'] * 40) + 
    ((df['epa_diff_norm_detailed'] + 3) / 6 * 60)
)

# Overall CDEM: Average of both levels (weighted 40% basic, 60% detailed)
df['CDEM'] = 0.4 * df['CDEM_basic'] + 0.6 * df['CDEM_detailed']

print(f"✓ CDEM calculated for {df['CDEM'].notna().sum()} plays")
print(f"\n  Basic CDEM (run/pass choice): {df['CDEM_basic'].mean():.2f}")
print(f"    Optimal basic play chosen: {df['chose_optimal_play_basic'].mean():.1%}")
print(f"\n  Detailed CDEM (direction/depth): {df['CDEM_detailed'].mean():.2f}")
print(f"    Optimal detailed play chosen: {df['chose_optimal_play_detailed'].mean():.1%}")
print(f"\n  Overall CDEM: {df['CDEM'].mean():.2f}")
print("\n" + "="*80)
print("GENERATING BENCHMARKS")
print("="*80)

# Team-level statistics
team_stats = df.groupby(['season', 'posteam']).agg({
    'DTDP': 'mean',
    'CDEM': 'mean',
    'CDEM_basic': 'mean',
    'CDEM_detailed': 'mean',
    'chose_optimal_play_basic': 'mean',
    'chose_optimal_play_detailed': 'mean',
    'touchdown': 'sum',
    'play_id': 'count',
    'epa': 'mean',
    'success': 'mean'
}).reset_index()

team_stats.columns = ['season', 'team', 'avg_DTDP', 'avg_CDEM', 
                       'avg_CDEM_basic', 'avg_CDEM_detailed',
                       'optimal_play_pct_basic', 'optimal_play_pct_detailed',
                       'total_touchdowns', 'total_plays', 'avg_epa', 'success_rate']

# Overall benchmarks
overall_benchmarks = {
    'avg_DTDP': df['DTDP'].mean(),
    'avg_CDEM': df['CDEM'].mean(),
    'avg_CDEM_basic': df['CDEM_basic'].mean(),
    'avg_CDEM_detailed': df['CDEM_detailed'].mean(),
    'optimal_play_rate_basic': df['chose_optimal_play_basic'].mean(),
    'optimal_play_rate_detailed': df['chose_optimal_play_detailed'].mean(),
    'td_rate': df['touchdown'].mean(),
    'avg_epa': df['epa'].mean(),
    'success_rate': df['success'].mean()
}

print("\nHistorical Benchmarks (2015-2023):")
print(f"  Average DTDP: {overall_benchmarks['avg_DTDP']:.3f}")
print(f"  Average CDEM (Overall): {overall_benchmarks['avg_CDEM']:.2f}")
print(f"  Average CDEM (Basic - run/pass): {overall_benchmarks['avg_CDEM_basic']:.2f}")
print(f"  Average CDEM (Detailed - direction/depth): {overall_benchmarks['avg_CDEM_detailed']:.2f}")
print(f"  Optimal Basic Play Rate: {overall_benchmarks['optimal_play_rate_basic']:.1%}")
print(f"  Optimal Detailed Play Rate: {overall_benchmarks['optimal_play_rate_detailed']:.1%}")
print(f"  TD Rate: {overall_benchmarks['td_rate']:.1%}")
print(f"  Success Rate: {overall_benchmarks['success_rate']:.1%}")

print("\n" + "="*80)
print("SAVING OUTPUTS")
print("="*80)

# Save enhanced dataset
output_file = 'redzone_DTDP_2015_2023_enhanced.csv'
df.to_csv(output_file, index=False)
print(f"✓ Enhanced dataset saved: {output_file}")

# Save team statistics
team_stats.to_csv('team_redzone_stats_2015_2023.csv', index=False)
print(f"✓ Team statistics saved: team_redzone_stats_2015_2023.csv")

# Save benchmarks
pd.DataFrame([overall_benchmarks]).to_csv('historical_benchmarks.csv', index=False)
print(f"✓ Benchmarks saved: historical_benchmarks.csv")

# Save optimal play lookup
optimal_plays_basic.to_csv('optimal_play_lookup_basic.csv', index=False)
print(f"✓ Optimal play lookup (basic) saved: optimal_play_lookup_basic.csv")

optimal_plays_detailed.to_csv('optimal_play_lookup_detailed.csv', index=False)
print(f"✓ Optimal play lookup (detailed) saved: optimal_play_lookup_detailed.csv")

# Save feature importance
feature_importance.to_csv('dtdp_feature_importance.csv', index=False)
print(f"✓ Feature importance saved: dtdp_feature_importance.csv")

print("\n" + "="*80)
print("SCRIPT 1 COMPLETE")
print("="*80)
print("\nOutputs generated:")
print("  1. redzone_DTDP_2015_2023_enhanced.csv - Main dataset with DTDP & CDEM")
print("  2. team_redzone_stats_2015_2023.csv - Team-level aggregations")
print("  3. historical_benchmarks.csv - Overall benchmarks for comparison")
print("  4. optimal_play_lookup_basic.csv - Optimal play types (run vs pass)")
print("  5. optimal_play_lookup_detailed.csv - Optimal play details (direction & depth)")
print("  6. dtdp_feature_importance.csv - Feature importance rankings")
print("\nCDEM now includes:")
print("  • Basic level: Run vs Pass decision quality")
print("  • Detailed level: Specific direction (left/center/right) & pass depth (short/deep)")
