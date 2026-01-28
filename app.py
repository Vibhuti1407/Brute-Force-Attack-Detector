import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Universal Brute Force Detector", layout="wide")

# --- Load Model ---
@st.cache_resource
def load_model():
    # Ensure you have trained the model using the XGBoost script provided earlier
    return joblib.load('brute_force_model.pkl')

model = load_model()

st.title("🛡️ Brute Force Attack Detector")
st.info("Upload any CSV. The file should atleast contain timestamp, ip, username, status as columns.")

uploaded_file = st.file_uploader("Upload Log File (CSV)", type="csv")

if uploaded_file:
    # Read a sample to show columns
    df_raw = pd.read_csv(uploaded_file)
    
    st.sidebar.header("Settings: Column Mapping")
    st.sidebar.write("Map your file columns to the detector fields:")
    
    # User selects which column is which
    all_cols = df_raw.columns.tolist()
    col_time = st.sidebar.selectbox("Timestamp Column", all_cols, index=0)
    col_ip = st.sidebar.selectbox("IP Address Column", all_cols, index=1 if len(all_cols)>1 else 0)
    col_status = st.sidebar.selectbox("Login Status Column", all_cols, index=2 if len(all_cols)>2 else 0)
    col_user = st.sidebar.selectbox("Username Column (Optional)", ["None"] + all_cols)

    # Threshold Adjustment
    sensitivity = st.sidebar.slider("Detection Sensitivity (Lower = stricter)", 0.1, 0.9, 0.5)

    if st.button("Analyze Logs Now"):
        # 1. Standardize the data
        df = df_raw.copy()
        
        # Convert Time
        df['std_timestamp'] = pd.to_datetime(df[col_time], errors='coerce')
        df = df.dropna(subset=['std_timestamp'])
        
        # Standardize Status (Search for 'fail' or 'err' keywords)
        # This makes it work even if the CSV says "Failure", "Login Failed", or "Error"
        df['is_failed'] = df[col_status].astype(str).str.lower().str.contains('fail|err|block|deny')

        # 2. Feature Engineering
        df['minute'] = df['std_timestamp'].dt.floor('T')
        
        analysis = df.groupby([col_ip, 'minute']).agg(
            attempts=(col_ip, 'count'),
            failed_attempts=('is_failed', 'sum'),
            unique_users=(col_user, 'nunique') if col_user != "None" else ('is_failed', lambda x: 1)
        ).reset_index()

        # 3. Prediction
        # Ensure feature order matches the trained model: [attempts, failed_attempts, unique_users]
        features = analysis[['attempts', 'failed_attempts', 'unique_users']]
        
        analysis['prob'] = model.predict_proba(features)[:, 1]
        analysis['is_attack'] = (analysis['prob'] >= sensitivity).astype(int)

        # 4. Display Results
        alerts = analysis[analysis['is_attack'] == 1].copy()
        
        st.subheader("Results")
        m1, m2 = st.columns(2)
        m1.metric("Lines Scanned", len(df))
        m2.metric("Threats Detected", len(alerts))

        if not alerts.empty:
            st.error(f"Detected {len(alerts)} brute force attempts!")
            
            # Visualization
            fig = px.scatter(alerts, x='minute', y='attempts', color='prob', 
                             title="Attack Intensity Over Time",
                             labels={'prob': 'Threat Score', 'minute': 'Time'})
            fig.update_traces(marker={'size': 30}) 
            st.plotly_chart(fig, width='stretch')
            
            st.subheader("🚨 Threat Intelligence Report")
            st.dataframe(alerts.sort_values('prob', ascending=False), width='stretch')
            
            # Download Button
            csv = alerts.to_csv(index=False).encode('utf-8')
            st.download_button("Download Alert Report", csv, "brute_force_alerts.csv", "text/csv")

            
        else:
            st.success("No attacks detected based on current settings.")