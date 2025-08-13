import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv(r"C:\Users\USER\Downloads\new_company_data.csv")

# Feature Engineering
df['capacity_fade'] = df['batteryDetails.full_charge_capacity_ah'] / (df['batteryDetails.total_battery_capacity_ah'] + 1e-6)
df['battery_age_months'] = (pd.to_datetime(df['timestamp']).dt.year - df['batteryDetails.battery_installation_year']) * 12 + \
                           (pd.to_datetime(df['timestamp']).dt.month - df['batteryDetails.battery_installation_month'])
df['battery_age_months'] = df['battery_age_months'].clip(lower=0)
df['recent_warning'] = ((df['batteryDetails.warning'] != 0) | (df['batteryDetails.protection'] != 0) | (df['batteryDetails.error_code'] != 0)).astype(int)

# Choose & Normalize Features
features = [
    'batteryDetails.soh',                      
    'capacity_fade',                           
    'batteryDetails.cell_level_voltage_difference', # Lower = better
    'batteryDetails.maximum_cell_temperature',  # Lower = better
    'batteryDetails.internal_resistance_of_battery', # Lower = better
    'battery_age_months',                      # Lower = better
    'batteryDetails.no_of_battery_cycles',      # Lower = better
    'batteryDetails.remaining_battery_cycles',  # Higher = better
    'recent_warning'                           # Lower = better
]

# Prepare for normalization (reverse for "lower is better")
df['imbalance_norm'] = MinMaxScaler().fit_transform(df[['batteryDetails.cell_level_voltage_difference']])
df['max_temp_norm'] = MinMaxScaler().fit_transform(df[['batteryDetails.maximum_cell_temperature']])
df['internal_res_norm'] = MinMaxScaler().fit_transform(df[['batteryDetails.internal_resistance_of_battery']])
df['age_norm'] = MinMaxScaler().fit_transform(df[['battery_age_months']])
df['cycles_norm'] = MinMaxScaler().fit_transform(df[['batteryDetails.no_of_battery_cycles']])
df['soh_norm'] = MinMaxScaler().fit_transform(df[['batteryDetails.soh']])
df['cap_fade_norm'] = MinMaxScaler().fit_transform(df[['capacity_fade']])
df['rem_cycles_norm'] = MinMaxScaler().fit_transform(df[['batteryDetails.remaining_battery_cycles']])
# Warning is already 0/1

# Invert lower-better metrics
df['imbalance_norm_inv'] = 1 - df['imbalance_norm']
df['max_temp_norm_inv'] = 1 - df['max_temp_norm']
df['internal_res_norm_inv'] = 1 - df['internal_res_norm']
df['age_norm_inv'] = 1 - df['age_norm']
df['cycles_norm_inv'] = 1 - df['cycles_norm']
df['recent_warning_inv'] = 1 - df['recent_warning']

# Final weighted health score (out of 100)
df['health_score'] = (
    0.25 * df['soh_norm'] +
    0.20 * df['cap_fade_norm'] +
    0.15 * df['imbalance_norm_inv'] +
    0.10 * df['max_temp_norm_inv'] +
    0.10 * df['internal_res_norm_inv'] +
    0.05 * df['age_norm_inv'] +
    0.05 * df['cycles_norm_inv'] +
    0.05 * df['rem_cycles_norm'] +
    0.05 * df['recent_warning_inv']
) * 100

# Bucket health status
df['health_status'] = pd.cut(df['health_score'], bins=[0, 50, 80, 100], labels=['Critical', 'Moderate', 'Healthy'])

# Save and Visualize
df[['batteryDetails.battery_pack_serial_no', 'health_score', 'health_status']].to_csv('battery_health_scores.csv', index=False)

import matplotlib.pyplot as plt
plt.hist(df['health_score'], bins=30, color='dodgerblue', alpha=0.7)
plt.xlabel('Battery Health Score')
plt.ylabel('Number of Packs')
plt.title('Battery Pack Health Score Distribution')
plt.show()

