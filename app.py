import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from config import SimConfig
from simulation import SimulationEngine, AnalyticsEngine

# --- Page Config ---
st.set_page_config(page_title="Bus Traffic Sim", layout="wide")
st.title("🚌 Traffic Simulation: Behavioral Analysis")

# --- Sidebar Controls ---
st.sidebar.header("🧪 Experiment Settings")

# 1. Logic & Scenario
st.sidebar.subheader("Scenario")
scenario_type = st.sidebar.radio("Bus Behavior", ["A: Bus Stops in Lane", "B: Bus Bay (Layby)"])
scenario_code = "A" if "A:" in scenario_type else "B"

# 2. Traffic Settings
st.sidebar.subheader("Traffic Conditions")
traffic_density = st.sidebar.slider("Inflow (cars/sec)", 0.1, 1.0, 0.4)
batch_size = st.sidebar.slider("Total Cars to Sim", 20, 100, 50)

# 3. Bus Settings
st.sidebar.subheader("Bus Parameters")
bus_stop_duration = st.sidebar.slider("Stop Duration (s)", 5, 30, 15)
pax_car = st.sidebar.number_input("Pax per Car", 1.0, 5.0, 1.2)
pax_bus = st.sidebar.number_input("Pax per Bus", 10.0, 100.0, 50.0)

# --- Simulation Execution ---
config = SimConfig(
    total_cars_to_spawn=batch_size, 
    traffic_rate=traffic_density,
    pax_per_car=pax_car,
    pax_per_bus=pax_bus,
    bus_dwell_time=bus_stop_duration
)

if st.button("Run Simulation"):
    with st.spinner(f"Simulating {batch_size} cars..."):
        sim = SimulationEngine(scenario=scenario_code, config=config)
        sim.init_scenario()
        
        # --- BATCH LOOP ---
        frame_count = 0
        max_sim_time = 300 
        progress_bar = st.progress(0)
        
        while not sim.all_vehicles_finished():
            sim.step()
            frame_count += 1
            if frame_count * sim.dt >= max_sim_time: break
            if frame_count % 10 == 0:
                 progress_bar.progress(min(sim.cars_finished_count / (batch_size + 1), 1.0))
                 
        progress_bar.empty()
        df = sim.get_results()       
        
        # --- Analytics ---
        analytics = AnalyticsEngine(df, config)
        kpis = analytics.compute_kpis()
        
        # --- 1. DATA PREPARATION (Smart Sampling) ---
        st.subheader("Traffic Visualization")

        # A. Strict Sorting
        df = df.sort_values(by=['id', 'time'])

        # B. SMART SAMPLING (The Fix)
        # We want to reduce data size BUT keep every "State Change" event.
        
        # 1. Identify rows where state changes compared to previous row for same car
        df['prev_state'] = df.groupby('id')['state'].shift(1)
        df['state_change'] = df['state'] != df['prev_state']
        
        # 2. Identify "Boring" rows (State is same as before)
        # We will keep only 20% of these to save space
        np.random.seed(42)
        df['keep_random'] = np.random.rand(len(df)) < 0.2
        
        # 3. Combine: Keep if State Changed OR Random Sample OR It's the Bus
        # We always keep the Bus data to ensure its line is perfect.
        keep_mask = (df['state_change']) | (df['keep_random']) | (df['type'] == 'Bus')
        chart_data = df[keep_mask].copy()

        # C. COLOR LOGIC (Refined)
        def get_inner_color(row):
            state_str = str(row['state'])

            # BUS LOGIC
            if row['type'] == 'Bus':
                if "SHOCKED" in state_str: return 'black'   # Panic
                if state_str == 'IN_BAY': return '#333333'  # Dark Charcoal
                if state_str == 'WAITING_TO_MERGE': return '#FFD700' # Gold
                return 'white'       # Moving
            
            # CAR LOGIC
            else:
                if state_str == 'Squeezing!': return 'purple' 
                if "Reacting" in state_str: return '#FF69B4' # Hot Pink
                return '#aec7e8'     # Light Blue

        chart_data['inner_color'] = chart_data.apply(get_inner_color, axis=1)

        # --- 2. VISUALIZATION (Clean & Direct) ---
        
        # Define Selection: Select a specific ID on hover
        hover = alt.selection_point(on='mouseover', nearest=False, fields=['id'], empty=False)

        # Base Chart
        base = alt.Chart(chart_data).encode(
            x=alt.X('time', title='Time (s)'),
            y=alt.Y('x', title='Position (m)'),
            detail='id',
            tooltip=[
                alt.Tooltip('id', title='Vehicle ID'),
                alt.Tooltip('type', title='Type'),
                alt.Tooltip('v', format='.1f', title='Speed (m/s)'),
                alt.Tooltip('state', title='Status'),
                alt.Tooltip('presence', format='.2f', title='Bus Visibility')
            ]
        )

        # LAYER 1: The "Outer" Line (Vehicle Color)
        # FIX: Opacity changes. Normal = 0.3. Hovered = 1.0 (Focus Mode)
        outer_line = base.mark_line(strokeWidth=4).encode(
            color=alt.Color('type', scale=alt.Scale(domain=['Car', 'Bus'], range=['#1f77b4', '#d62728']), legend=None),
            opacity=alt.condition(hover, alt.value(1.0), alt.value(0.8)) 
        ).add_params(hover)

        # LAYER 2: The "Inner" Line (State Color)
        inner_line = base.mark_line(strokeWidth=2).encode(
            color=alt.Color('inner_color', scale=None, legend=None),
            opacity=alt.condition(hover, alt.value(1.0), alt.value(0.8))
        )

        # Combine
        chart = (outer_line + inner_line).properties(
            width=800, height=500,
            title=f"Scenario {scenario_code}: Trajectories (Hover line to Focus)"
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

        
        # --- 3. Results ---
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Time", f"{df['time'].max():.1f} s")
        c2.metric("Avg Speed", f"{kpis['avg_person_speed']:.1f} km/h")
        c3.metric("Throughput", f"{kpis['throughput_people']} pax")


# import streamlit as st
# import pandas as pd
# import altair as alt
# import numpy as np
# from config import SimConfig
# from simulation import SimulationEngine, AnalyticsEngine

# # --- Page Config ---
# st.set_page_config(page_title="Bus Traffic Sim", layout="wide")
# st.title("🚌 Traffic Simulation: Behavioral Analysis")

# # --- Sidebar Controls ---
# st.sidebar.header("🧪 Experiment Settings")

# # 1. Logic & Scenario
# st.sidebar.subheader("Scenario")
# scenario_type = st.sidebar.radio("Bus Behavior", ["A: Bus Stops in Lane", "B: Bus Bay (Layby)"])
# scenario_code = "A" if "A:" in scenario_type else "B"

# # 2. Traffic Settings
# st.sidebar.subheader("Traffic Conditions")
# traffic_density = st.sidebar.slider("Inflow (cars/sec)", 0.1, 1.0, 0.4)
# batch_size = st.sidebar.slider("Total Cars to Sim", 20, 100, 50)

# # 3. Bus Settings
# st.sidebar.subheader("Bus Parameters")
# bus_stop_duration = st.sidebar.slider("Stop Duration (s)", 5, 30, 15)
# pax_car = st.sidebar.number_input("Pax per Car", 1.0, 5.0, 1.2)
# pax_bus = st.sidebar.number_input("Pax per Bus", 10.0, 100.0, 50.0)

# # --- Simulation Execution ---
# config = SimConfig(
#     total_cars_to_spawn=batch_size, 
#     traffic_rate=traffic_density,
#     pax_per_car=pax_car,
#     pax_per_bus=pax_bus,
#     bus_dwell_time=bus_stop_duration
# )

# if st.button("Run Simulation"):
#     with st.spinner(f"Simulating {batch_size} cars..."):
#         sim = SimulationEngine(scenario=scenario_code, config=config)
#         sim.init_scenario()
        
#         # --- BATCH LOOP ---
#         frame_count = 0
#         max_sim_time = 300 
#         progress_bar = st.progress(0)
        
#         while not sim.all_vehicles_finished():
#             sim.step()
#             frame_count += 1
#             if frame_count * sim.dt >= max_sim_time: break
#             if frame_count % 10 == 0:
#                  progress_bar.progress(min(sim.cars_finished_count / (batch_size + 1), 1.0))
                 
#         progress_bar.empty()
#         df = sim.get_results()       
        
#         # --- Analytics ---
#         analytics = AnalyticsEngine(df, config)
#         kpis = analytics.compute_kpis()
        
#         # --- 1. DATA PREPARATION ---
#         st.subheader("Traffic Visualization")

#         # A. Strict Sorting: Prevents "Zig-Zags"
#         df = df.sort_values(by=['id', 'time'])

#         # B. Smart Downsampling
#         all_times = df['time'].unique()
#         target_frames = 300 
#         if len(all_times) > target_frames:
#             indices = np.linspace(0, len(all_times)-1, target_frames, dtype=int)
#             keep_times = all_times[indices]
#             chart_data = df[df['time'].isin(keep_times)].copy()
#         else:
#             chart_data = df.copy()

#         # C. COLOR LOGIC 
#         # We calculate the specific colors here in Python, not in Altair.
#         # This prevents the "Multiple values for keyword argument" error.
        
#         def get_inner_color(row):
#             state_str = str(row['state']) # Ensure string for "contains" check

#             # BUS LOGIC
#             if row['type'] == 'Bus':
#                 # Priority 1: Panic/Shock (Overrides everything)
#                 if "SHOCKED" in state_str: 
#                     return 'black' # The bus froze!
                
#                 # Priority 2: Infrastructure State
#                 if state_str == 'IN_BAY': 
#                     return '#333333' # Dark Grey (Clearly visible inside Red)
                
#                 if state_str == 'WAITING_TO_MERGE': 
#                     return '#FFD700' # Gold/Yellow (Warning color)
                
#                 return 'white' # Normal moving bus (Red outer, White inner)
            
#             # CAR LOGIC
#             else:
#                 if state_str == 'Squeezing!': 
#                     return 'purple' # Aggressive
                
#                 if "Reacting" in state_str:
#                     return '#FF69B4' # Hot Pink (Human Latency / Distraction) <-- CHANGED  

#                 return '#aec7e8' # Light Blue (Normal car)

#         chart_data['inner_color'] = chart_data.apply(get_inner_color, axis=1)
#         # --- 2. VISUALIZATION (Double Layer) ---
        
#         # Base Chart
#         base = alt.Chart(chart_data).encode(
#             x=alt.X('time', title='Time (s)'),
#             y=alt.Y('x', title='Position (m)'),
#             detail='id', # Group by ID to connect lines
#             tooltip=[
#                 alt.Tooltip('id', title='Vehicle ID'),
#                 alt.Tooltip('type', title='Type'),
#                 alt.Tooltip('v', format='.1f', title='Speed (m/s)'),
#                 alt.Tooltip('state', title='Status'),
#                 alt.Tooltip('presence', format='.2f', title='Bus Visibility')
#             ]
#         )

#         # LAYER 1: The "Outer" Line (Thick, Vehicle Color)
#         outer_line = base.mark_line(
#             strokeWidth=4, 
#             opacity=0.6
#         ).encode(
#             color=alt.Color('type', scale=alt.Scale(domain=['Car', 'Bus'], range=['#1f77b4', '#d62728']), legend=None)
#         )

#         # LAYER 2: The "Inner" Line (Thin, State Color)
#         # Now we just read the column we created above. Safe and fast.
#         inner_line = base.mark_line(
#             strokeWidth=2
#         ).encode(
#             color=alt.Color('inner_color', scale=None, legend=None)
#         )

#         # Combine Layers
#         chart = (outer_line + inner_line).properties(
#             width=800, height=500,
#             title=f"Scenario {scenario_code}: Trajectories (Inner Color = State)"
#         ).interactive()

#         st.altair_chart(chart, use_container_width=True)

#         # --- 3. Results ---
#         st.divider()
#         c1, c2, c3 = st.columns(3)
#         c1.metric("Total Time", f"{df['time'].max():.1f} s")
#         c2.metric("Avg Speed", f"{kpis['avg_person_speed']:.1f} km/h")
#         c3.metric("Throughput", f"{kpis['throughput_people']} pax")
