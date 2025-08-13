import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv(r"C:\Users\USER\Downloads\new_company_data.csv")

df['capacity_fade'] = df['batteryDetails.full_charge_capacity_ah'] / (df['batteryDetails.total_battery_capacity_ah'] + 1e-6)
df['battery_age_months'] = (pd.to_datetime(df['timestamp']).dt.year - df['batteryDetails.battery_installation_year']) * 12 + \
                           (pd.to_datetime(df['timestamp']).dt.month - df['batteryDetails.battery_installation_month'])
df['battery_age_months'] = df['battery_age_months'].clip(lower=0)
df['recent_warning'] = ((df['batteryDetails.warning'] != 0) | (df['batteryDetails.protection'] != 0) | (df['batteryDetails.error_code'] != 0)).astype(int)

features = [
    'batteryDetails.soh',                      
    'capacity_fade',                           
    'batteryDetails.cell_level_voltage_difference', 
    'batteryDetails.maximum_cell_temperature',  
    'batteryDetails.internal_resistance_of_battery', 
    'battery_age_months',                      
    'batteryDetails.no_of_battery_cycles',
    'batteryDetails.remaining_battery_cycles', 
    'recent_warning'                          


df['imbalance_norm'] = MinMaxScaler().fit_transform(df[['batteryDetails.cell_level_voltage_difference']])
df['max_temp_norm'] = MinMaxScaler().fit_transform(df[['batteryDetails.maximum_cell_temperature']])
df['internal_res_norm'] = MinMaxScaler().fit_transform(df[['batteryDetails.internal_resistance_of_battery']])
df['age_norm'] = MinMaxScaler().fit_transform(df[['battery_age_months']])
df['cycles_norm'] = MinMaxScaler().fit_transform(df[['batteryDetails.no_of_battery_cycles']])
df['soh_norm'] = MinMaxScaler().fit_transform(df[['batteryDetails.soh']])
df['cap_fade_norm'] = MinMaxScaler().fit_transform(df[['capacity_fade']])
df['rem_cycles_norm'] = MinMaxScaler().fit_transform(df[['batteryDetails.remaining_battery_cycles']])



df['imbalance_norm_inv'] = 1 - df['imbalance_norm']
df['max_temp_norm_inv'] = 1 - df['max_temp_norm']
df['internal_res_norm_inv'] = 1 - df['internal_res_norm']
df['age_norm_inv'] = 1 - df['age_norm']
df['cycles_norm_inv'] = 1 - df['cycles_norm']
df['recent_warning_inv'] = 1 - df['recent_warning']


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


df['health_status'] = pd.cut(df['health_score'], bins=[0, 50, 80, 100], labels=['Critical', 'Moderate', 'Healthy'])


df[['batteryDetails.battery_pack_serial_no', 'health_score', 'health_status']].to_csv('battery_health_scores.csv', index=False)

import matplotlib.pyplot as plt
plt.hist(df['health_score'], bins=30, color='dodgerblue', alpha=0.7)
plt.xlabel('Battery Health Score')
plt.ylabel('Number of Packs')
plt.title('Battery Pack Health Score Distribution')
plt.show()

