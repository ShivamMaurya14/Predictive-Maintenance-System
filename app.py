
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import random
import datetime

# --- Setup & Configuration ---
st.set_page_config(
    page_title="PMS - Predictive Maintenance System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Professional & Aesthetic Look ---
def load_css():
    st.markdown("""
    <style>
        /* Global Styling */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
            font-family: 'Inter', sans-serif;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #161B22;
            border-right: 1px solid #30363D;
        }
        
        /* Header Styling */
        h1 {
            font-weight: 700;
            color: #58A6FF;
            letter-spacing: -0.5px;
            margin-bottom: 0.5rem;
        }
        h2, h3 {
            font-weight: 600;
            color: #E6EDF3;
        }
        
        /* Metric Card Styling */
        div[data-testid="stMetric"] {
            background-color: #21262D;
            border: 1px solid #30363D;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: #58A6FF;
        }
        div[data-testid="stMetricLabel"] {
            color: #8B949E;
            font-size: 0.9rem;
        }
        div[data-testid="stMetricValue"] {
            color: #E6EDF3;
            
            font-weight: 700;
        }

        /* Button Styling */
        .stButton > button {
            background: linear-gradient(90deg, #238636 0%, #2EA043 100%);
            color: white;
            font-weight: 700;
            font-size: 1.1rem !important;
            border: none;
            padding: 0.8rem 2rem !important;
            border-radius: 6px;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 4px 14px rgba(46, 160, 67, 0.4);
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #2EA043 0%, #3FB950 100%);
            transform: scale(1.02);
            box-shadow: 0 6px 20px rgba(46, 160, 67, 0.6);
            padding: 0.9rem 2.1rem !important;
        }

        /* Input Fields */
        .stNumberInput, .stSelectbox {
            margin-bottom: 1rem;
        }
        
        /* Success/Error Message Containers */
        .stAlert {
            border-radius: 8px;
            padding: 1rem;
            border: none;
        }

        /* DataFrame Styling */
        div[data-testid="stDataFrame"] {
            border: 1px solid #30363D;
            border-radius: 8px;
            overflow: hidden;
        }

        /* Custom Divider */
        hr {
            border-color: #30363D;
            margin: 2rem 0;
        }
        
        /* Machine Card Grid */
        .machine-card {
            background-color: #161B22;
            border: 1px solid #30363D;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .status-ok {
            color: #3fb950;
            font-weight: bold;
        }
        .status-fail {
            color: #ff6b6b;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# --- App Logic ---

@st.cache_resource
def load_model():
    try:
        return joblib.load('model.joblib')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Mode Selection ---
# Create a sidebar navigation menu to switch between modes
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/factory.png", width=64)
    st.title("Choose Version")
    app_mode = st.radio("Select Mode", ["Manual Diagnostics (v1)", "Automatic Diagnostics (v2)"])
    st.markdown("---")

# Main Page Title (Common for both modes)
st.title("🏭 Predictive Maintenance System")

# ==============================================================================
# MODE 1: MANUAL DIAGNOSIS (Single Machine) - The original "Version 1"
# ==============================================================================
if app_mode == "Manual Diagnostics (v1)":
    
    st.markdown("### 🛠️ Manual Diagnostics Tool (Single Unit)")
    
    # Main Area Inputs
    st.header("⚙️ Parameter Configuration")
    
    # Create 3 equal columns for better spacing
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("#### Machine Specs")
        machine_type = st.selectbox("Quality Class", ['L (Low)', 'M (Medium)', 'H (High)'], index=1)
        type_code = machine_type.split()[0]
        tool_wear = st.number_input("Tool Wear [min]", 0, 300, 100, 1)

    with c2:
        st.markdown("#### Operation")
        rot_speed = st.number_input("Rotational Speed [rpm]", 1000, 3000, 1500, 10)
        torque = st.number_input("Torque [Nm]", 10.0, 100.0, 40.0, 0.1)

    with c3:
        st.markdown("#### Environment")
        air_temp = st.number_input("Air Temp [K]", 290.0, 310.0, 300.0, 0.1)
        process_temp = st.number_input("Process Temp [K]", 300.0, 320.0, 310.0, 0.1)

    st.markdown("---")

    # Main Area Calculation
    temp_diff = process_temp - air_temp
    power = torque * rot_speed * 2 * np.pi / 60

    if model:
        # Prepare Input
        type_mapping = {'H': 0, 'L': 1, 'M': 2}
        type_encoded = type_mapping[type_code]
        
        input_df = pd.DataFrame([{
            'Type': type_encoded,
            'Air temperature [K]': air_temp,
            'Process temperature [K]': process_temp,
            'Rotational speed [rpm]': rot_speed,
            'Torque [Nm]': torque,
            'Tool wear [min]': tool_wear,
            'temperature_difference': temp_diff,
            'Mechanical Power [W]': round(power, 4)
        }])

        # Centralized Prediction Button
        predict_btn = st.button("RUN DIAGNOSTICS", use_container_width=True, type="primary")
        
        if predict_btn:
            with st.spinner("Analyzing..."):
                time.sleep(0.3)
                try:
                    prediction = model.predict(input_df)[0]
                    probs = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else [0, 0]
                    
                    # Clean, centered result display
                    st.markdown("### Diagnosis Report")
                    
                    if prediction == 1:
                        st.error(f"⚠️ **FAILURE PREDICTED** (Confidence: {probs[1]:.1%})")
                        st.markdown(f"**Reasons:** High failure probability detected based on current parameters.")
                    else:
                        st.success(f"✅ **NORMAL** (Confidence: {probs[0]:.1%})")
                        st.markdown(f"**Status:** System parameters are within safe operating limits.")
                        
                except Exception as e:
                    st.error(f"Error: {e}")


# ==============================================================================
# MODE 2: AUTOMATIC DIAGNOSTICS (Fleet Tracking) - The new "Version 2"
# ==============================================================================
elif app_mode == "Automatic Diagnostics (v2)":
    
    st.markdown("### 🏭 Real-Time Fleet Monitoring (v2)")

    # --- Utility Functions for v2 ---
    def generate_machine_data(machine_id):
        """Simulates real-time sensor data for a machine"""
        types = ['L', 'M', 'H']
        return {
            'Machine ID': machine_id,
            'Type': random.choice(types),
            'Air temperature [K]': round(random.uniform(295, 305), 1),
            'Process temperature [K]': round(random.uniform(305, 315), 1),
            'Rotational speed [rpm]': random.randint(1200, 2000),
            'Torque [Nm]': round(random.uniform(20, 60), 1),
            'Tool wear [min]': random.randint(0, 250)
        }


    def process_and_predict(data_dict):
        """Processes raw data and returns prediction"""
        if not model:
            return None, 0, {}


        type_mapping = {'H': 0, 'L': 1, 'M': 2}
        type_encoded = type_mapping[data_dict['Type']]
        

        temp_diff = data_dict['Process temperature [K]'] - data_dict['Air temperature [K]']
        power = data_dict['Torque [Nm]'] * data_dict['Rotational speed [rpm]'] * 2 * np.pi / 60
        
        
        input_df = pd.DataFrame([{
            'Type': type_encoded,
            'Air temperature [K]': data_dict['Air temperature [K]'],
            'Process temperature [K]': data_dict['Process temperature [K]'],
            'Rotational speed [rpm]': data_dict['Rotational speed [rpm]'],
            'Torque [Nm]': data_dict['Torque [Nm]'],
            'Tool wear [min]': data_dict['Tool wear [min]'],
            'temperature_difference': temp_diff,
            'Mechanical Power [W]': round(power, 4)
        }])
        
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else 0
        
        derived_metrics = {
            'Temp Diff': f"{temp_diff:.1f} K",
            'Power': f"{power:.0f} W"
        }
        
        return prediction, prob, derived_metrics

    # --- Session State for Machine Fleet ---
    if 'machines' not in st.session_state:
        st.session_state.machines = [f"M-{i:03d}" for i in range(1, 10)] # 9 machines start from 1 to 9 (range(1, 10))

    if 'machine_data' not in st.session_state:
        st.session_state.machine_data = {m_id: generate_machine_data(m_id) for m_id in st.session_state.machines}

    # Sidebar Controls for v2
    with st.sidebar:
        st.markdown("### Fleet Controls")
        
        # Add Machine
        with st.expander("Add New Machine"):
            new_machine_id = st.text_input("New Machine ID", placeholder="e.g. M-010")
            if st.button("Add Unit"):
                if new_machine_id and new_machine_id not in st.session_state.machines:
                    st.session_state.machines.append(new_machine_id)
                    st.session_state.machine_data[new_machine_id] = generate_machine_data(new_machine_id)
                    st.success(f"Added {new_machine_id}")
                    st.rerun()

        # Remove Machine
        with st.expander("Remove Machine"):
            machine_to_remove = st.selectbox("Select Unit to Remove", st.session_state.machines)
            if st.button("Delete Unit"):
                if machine_to_remove in st.session_state.machines:
                    st.session_state.machines.remove(machine_to_remove)
                    del st.session_state.machine_data[machine_to_remove]
                    st.success(f"Removed {machine_to_remove}")
                    st.rerun()
        
        st.markdown("---")
        
        # Simulation Controls
        if st.button("🔄 Refresh Data Feed"):
            for m_id in st.session_state.machines:
                 st.session_state.machine_data[m_id] = generate_machine_data(m_id)
            st.toast("Updated sensor data from all units", icon="📡")
        
        st.caption(f"Last Update: {datetime.datetime.now().strftime('%H:%M:%S')}")

    # Dashboard Summary
    total_machines = len(st.session_state.machines)
    failures_detected = 0

    machine_status = {}
    for m_id, data in st.session_state.machine_data.items():
        pred, prob, metrics = process_and_predict(data)
        machine_status[m_id] = {
            'pred': pred,
            'prob': prob,
            'metrics': metrics,
            'data': data
        }
        if pred == 1:
            failures_detected += 1

    # Top Level Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Active Units", total_machines)
    m2.metric("Healthy Units", total_machines - failures_detected)
    
    # Conditional color for Critical Alerts
    alert_color = "inverse" if failures_detected > 0 else "normal" # inverse usually red/dark in Streamlit, normal green/neutral
    # Streamlit metrics don't support custom colors directly, but 'inverse' often signifies attention.
    # To force red/green, we use delta colors.
    
    delta_val = f"{failures_detected} detected" if failures_detected > 0 else "All systems go"
    delta_color = "inverse" if failures_detected > 0 else "off" # 'off' is grey, 'normal' is green, 'inverse' is red
    
    m3.metric("Critical Alerts", failures_detected, delta=delta_val, delta_color=delta_color)

    st.markdown("---")

    # Machine Grid Display
    st.subheader("Unit Status Overview")
    cols = st.columns(3)

    for idx, m_id in enumerate(st.session_state.machines):
        status = machine_status[m_id]
        data = status['data']
        col_idx = idx % 3
        
        with cols[col_idx]:
            border_color = "#d03030" if status['pred'] == 1 else "#2ea043"
            status_text = "CRITICAL FAIL" if status['pred'] == 1 else "OPERATIONAL"
            status_icon = "🚨" if status['pred'] == 1 else "✅"
            
            with st.container(border=True):
                st.markdown(f"#### {status_icon} Unit: {m_id}")
                st.markdown(f"**Type:** {data['Type']} | **Status:** <span style='color:{border_color}'>{status_text}</span>", unsafe_allow_html=True)
                
                st.progress(status['prob'], text=f"Failure Probability: {status['prob']:.1%}")
                
                c1, c2 = st.columns(2)
                c1.markdown(f"**Temp:** {data['Process temperature [K]']} K")
                c2.markdown(f"**RPM:** {data['Rotational speed [rpm]']}")
                
                c3, c4 = st.columns(2)
                c3.markdown(f"**Torque:** {data['Torque [Nm]']} Nm")
                c4.markdown(f"**Wear:** {data['Tool wear [min]']} min")
                
                with st.expander("Detailed Telemetry"):
                    st.write("Derived Metrics:")
                    st.json(status['metrics'])
                    st.write("Raw Sensor Data:")
                    st.json(data)
