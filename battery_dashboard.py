import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('battery_health_scores.csv') 
raw_df = pd.read_csv(r"C:\Users\USER\Downloads\new_company_data.csv") 
st.title(' Battery Health Score Dashboard')

# ---- Summary stats ----
st.header('Fleet Health Overview')
healthy = (df['health_status'] == 'Healthy').sum()
moderate = (df['health_status'] == 'Moderate').sum()
critical = (df['health_status'] == 'Critical').sum()
st.write(f"**Healthy:** {healthy} | **Moderate:** {moderate} | **Critical:** {critical}")

# ---- Histogram ----
st.subheader('Health Score Distribution')
fig, ax = plt.subplots()
ax.hist(df['health_score'], bins=30, color='dodgerblue', alpha=0.7)
ax.set_xlabel('Health Score')
ax.set_ylabel('Number of Battery Packs')
st.pyplot(fig)

# ---- Filter/Search ----
st.subheader('Battery Health Table')
serial_filter = st.text_input('Filter by Serial Number (partial or full):')
if serial_filter:
    filtered_df = df[df['batteryDetails.battery_pack_serial_no'].astype(str).str.contains(serial_filter)]
else:
    filtered_df = df

st.dataframe(
    filtered_df.sort_values('health_score').reset_index(drop=True),
    use_container_width=True,
    height=300
)

# ---- Worst Packs ----
st.subheader('Worst Performing Packs')
top_n = st.slider('Show N worst packs:', min_value=1, max_value=20, value=5)
worst_packs = df.nsmallest(top_n, 'health_score')
st.table(worst_packs[['batteryDetails.battery_pack_serial_no', 'health_score', 'health_status']])

# ---- Drilldown ----
st.subheader('Detailed Metrics')
selected_serial = st.selectbox('Select a pack for details:', worst_packs['batteryDetails.battery_pack_serial_no'].unique())
if selected_serial:
    details = raw_df[raw_df['batteryDetails.battery_pack_serial_no'] == selected_serial]
    st.write('**Most recent data for this pack:**')
    st.dataframe(details.sort_values('timestamp', ascending=False).head(1).T)

    # Optionally show trend/history for this pack
    st.write('**Health score over time (if available):**')
    if 'health_score' in details.columns:
        fig2, ax2 = plt.subplots()
        ax2.plot(pd.to_datetime(details['timestamp']), details['health_score'])
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Health Score')
        st.pyplot(fig2)
    else:
        st.info("Health score history not available (only snapshot).")

st.caption("Made by Bhavish with Streamlit.")