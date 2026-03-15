import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import time 
from config import SimConfig
from simulation import SimulationEngine, AnalyticsEngine
from graphics.renderer import SimulationRenderer 

# --- Page Config ---
st.set_page_config(page_title="Bus Traffic Sim", layout="wide")
st.title("🚌 Traffic Simulation: Behavioral Analysis")

# --- INITIALIZE MEMORY ---
# This prevents Streamlit from restarting when you click play
if 'sim_complete' not in st.session_state:
    st.session_state.sim_complete = False

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
        
        # Save to memory
        st.session_state.df = sim.get_results()       
        st.session_state.sim_complete = True

# --- DASHBOARD & ANIMATION (Runs outside the Run button) ---
if st.session_state.sim_complete:
    df = st.session_state.df
    
    # --- Analytics ---
    analytics = AnalyticsEngine(df, config)
    kpis = analytics.compute_kpis()
    
    # --- 1. DATA PREPARATION (Smart Sampling) ---
    st.subheader("Traffic Visualization")

    # A. Strict Sorting
    df = df.sort_values(by=['id', 'time'])

    # B. SMART SAMPLING (The Fix)
    df['prev_state'] = df.groupby('id')['state'].shift(1)
    df['state_change'] = df['state'] != df['prev_state']
    
    np.random.seed(42)
    df['keep_random'] = np.random.rand(len(df)) < 0.2
    
    keep_mask = (df['state_change']) | (df['keep_random']) | (df['type'] == 'Bus')
    chart_data = df[keep_mask].copy()

    
    # C. COLOR LOGIC (Refined)
    def get_inner_color(row):
        state_str = str(row['state'])

        # BUS LOGIC
        if row['type'] == 'Bus':
            if "SHOCKED" in state_str: 
                return 'black'   # Panic
            if state_str == 'IN_BAY': 
                return '#333333'  # Dark Charcoal
            if state_str == 'WAITING_TO_MERGE': 
                presence = float(row['presence'])
                if presence <= 0.85:
                    return '#FFD700' # Gold (Creeping)
                else:
                    return '#FF0000' # Red (Dominance)
            
            # If the bus is anything else (like MOVING), it is white.
            # Make sure this is indented directly under "if row['type'] == 'Bus':"
            return 'white'       
        
        # CAR LOGIC
        else:
            if state_str == 'Squeezing!': return 'purple' 
            if "Reacting" in state_str: return '#FF69B4' # Hot Pink
            return '#aec7e8'     # Light Blue

    chart_data['inner_color'] = chart_data.apply(get_inner_color, axis=1)

    # --- 2. VISUALIZATION (Clean & Direct) ---
    hover = alt.selection_point(on='mouseover', fields=['id'], empty=False)
    click = alt.selection_point(fields=['id'], toggle=True)

    base = alt.Chart(chart_data).encode(
        x=alt.X('time', title='Time (s)'),
        y=alt.Y('x', title='Position (m)'),
        detail='id',
        tooltip=[
            alt.Tooltip('id', title='Vehicle ID'),
            alt.Tooltip('type', title='Type'),
            alt.Tooltip('x', format='.1f', title='Absolute Pos (m)'),
            alt.Tooltip('v', format='.1f', title='Speed (m/s)'),
            alt.Tooltip('state', title='Status'),
            alt.Tooltip('presence', format='.2f', title='Own Solidity'),
            alt.Tooltip('leader_info', title='Following'),
            alt.Tooltip('bus_visibility', title='Perceives Bus As')
        ]
    )

    outer_line = base.mark_line().encode(
        color=alt.Color('type', scale=alt.Scale(domain=['Car', 'Bus'], range=['#1f77b4', '#d62728']), legend=None),
        opacity=alt.condition(click, alt.value(0.8), alt.value(0.1)),
        strokeWidth=alt.condition(hover, alt.value(7), alt.value(4))
    ).add_params(hover, click)

    inner_line = base.mark_line().encode(
        color=alt.Color('inner_color', scale=None, legend=None),
        opacity=alt.condition(click, alt.value(1.0), alt.value(0.1)),
        strokeWidth=alt.condition(hover, alt.value(3), alt.value(2))
    )

    chart = (outer_line + inner_line).properties(
        width=800, height=500,
        title=f"Scenario {scenario_code}: Trajectories (Hover = Thicker, Shift+Click = Isolate)"
    ).interactive()

    st.altair_chart(chart, use_container_width=True)
    
    # --- 3. Results ---
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Time", f"{df['time'].max():.1f} s")
    c2.metric("Avg Speed", f"{kpis['avg_person_speed']:.1f} km/h")
    c3.metric("Throughput", f"{kpis['throughput_people']} pax")

    # --- 4. 2D ANIMATION VISUALIZER ---
    st.divider()
    st.subheader("🎬 Live 2D Playback")
    
    # We add columns to put the Play button and Download button side-by-side
    btn_col1, btn_col2 = st.columns([1, 4])
    
    with btn_col1:
        play_btn = st.button("Render Smooth Video")
        
    if play_btn:
        with st.spinner("Rendering Video (Takes ~5 seconds)..."):
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            
            renderer = SimulationRenderer(df, config)
            # Take every 4th frame. 
            time_steps = sorted(df['time'].unique())[::4] 
            
            fig = plt.figure(figsize=(12, 3))
            ax = fig.add_axes([0, 0, 1, 1]) 
            
            def update(t):
                renderer.draw_frame(ax, t)
                return ax,
                
            anim = animation.FuncAnimation(fig, update, frames=time_steps, blit=False)
            
            # SLOWED DOWN: fps changed from 10 to 5 for half-speed playback
            anim.save("animation.gif", writer='pillow', fps=5)
            plt.close(fig)
            
            st.image("animation.gif", use_container_width=True)
            
            # THE NEW DOWNLOAD BUTTON
            with open("animation.gif", "rb") as file:
                btn_col2.download_button(
                    label="💾 Download GIF",
                    data=file,
                    file_name="bus_simulation.gif",
                    mime="image/gif"
                )

                
# import streamlit as st
# import pandas as pd
# import altair as alt
# import numpy as np
# from config import SimConfig
# from simulation import SimulationEngine, AnalyticsEngine
# import time # Needed to control the frame rate
# from graphics.renderer import SimulationRenderer # NEW

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
        
#         # --- 1. DATA PREPARATION (Smart Sampling) ---
#         st.subheader("Traffic Visualization")

#         # A. Strict Sorting
#         df = df.sort_values(by=['id', 'time'])

#         # B. SMART SAMPLING (The Fix)
#         # We want to reduce data size BUT keep every "State Change" event.
        
#         # 1. Identify rows where state changes compared to previous row for same car
#         df['prev_state'] = df.groupby('id')['state'].shift(1)
#         df['state_change'] = df['state'] != df['prev_state']
        
#         # 2. Identify "Boring" rows (State is same as before)
#         # We will keep only 20% of these to save space
#         np.random.seed(42)
#         df['keep_random'] = np.random.rand(len(df)) < 0.2
        
#         # 3. Combine: Keep if State Changed OR Random Sample OR It's the Bus
#         # We always keep the Bus data to ensure its line is perfect.
#         keep_mask = (df['state_change']) | (df['keep_random']) | (df['type'] == 'Bus')
#         chart_data = df[keep_mask].copy()

#         # C. COLOR LOGIC (Refined)
#         def get_inner_color(row):
#             state_str = str(row['state'])

#             # BUS LOGIC
#             if row['type'] == 'Bus':
#                 if "SHOCKED" in state_str: return 'black'   # Panic
#                 if state_str == 'IN_BAY': return '#333333'  # Dark Charcoal
#                 # New Dynamic Merge Colors
#                 if state_str == 'WAITING_TO_MERGE': 
#                     presence = float(row['presence'])
#                     if presence <= 0.85:
#                         return '#FFD700' # Gold (Creeping - Cars can still squeeze)
#                     else:
#                         return '#FF0000' # Red (Dominance - Cars MUST yield)                return 'white'       # Moving
            
#             # CAR LOGIC
#             else:
#                 if state_str == 'Squeezing!': return 'purple' 
#                 if "Reacting" in state_str: return '#FF69B4' # Hot Pink
#                 return '#aec7e8'     # Light Blue

#         chart_data['inner_color'] = chart_data.apply(get_inner_color, axis=1)

#         # --- 2. VISUALIZATION (Clean & Direct) ---
        
#         # Define Selections
#         # empty=False: Nothing is thick unless hovered
#         hover = alt.selection_point(on='mouseover', fields=['id'], empty=False)
#         # toggle=True: Hold shift to select multiple. (Default empty=True means all are visible initially)
#         click = alt.selection_point(fields=['id'], toggle=True)

#         # Base Chart
#         base = alt.Chart(chart_data).encode(
#             x=alt.X('time', title='Time (s)'),
#             y=alt.Y('x', title='Position (m)'),
#             detail='id',
#             tooltip=[
#                 alt.Tooltip('id', title='Vehicle ID'),
#                 alt.Tooltip('type', title='Type'),
#                 alt.Tooltip('x', format='.1f', title='Absolute Pos (m)'),
#                 alt.Tooltip('v', format='.1f', title='Speed (m/s)'),
#                 alt.Tooltip('state', title='Status'),
#                 alt.Tooltip('presence', format='.2f', title='Own Solidity'),
#                 alt.Tooltip('leader_info', title='Following'),
#                 alt.Tooltip('bus_visibility', title='Perceives Bus As')
#             ]
#         )

#         # LAYER 1: The "Outer" Line (Vehicle Color)
#         outer_line = base.mark_line().encode(
#             color=alt.Color('type', scale=alt.Scale(domain=['Car', 'Bus'], range=['#1f77b4', '#d62728']), legend=None),
#             # Click controls visibility (Focus Mode)
#             opacity=alt.condition(click, alt.value(0.8), alt.value(0.1)),
#             # Hover controls thickness
#             strokeWidth=alt.condition(hover, alt.value(7), alt.value(4))
#         ).add_params(hover, click)

#         # LAYER 2: The "Inner" Line (State Color)
#         inner_line = base.mark_line().encode(
#             color=alt.Color('inner_color', scale=None, legend=None),
#             opacity=alt.condition(click, alt.value(1.0), alt.value(0.1)),
#             strokeWidth=alt.condition(hover, alt.value(3), alt.value(2))
#         )

#         # Combine
#         chart = (outer_line + inner_line).properties(
#             width=800, height=500,
#             title=f"Scenario {scenario_code}: Trajectories (Hover = Thicker, Shift+Click = Isolate)"
#         ).interactive()

#         st.altair_chart(chart, use_container_width=True)
        
#         # --- 3. Results ---
#         st.divider()
#         c1, c2, c3 = st.columns(3)
#         c1.metric("Total Time", f"{df['time'].max():.1f} s")
#         c2.metric("Avg Speed", f"{kpis['avg_person_speed']:.1f} km/h")
#         c3.metric("Throughput", f"{kpis['throughput_people']} pax")


#         # --- 4. 2D ANIMATION VISUALIZER ---
#         st.divider()
#         st.subheader("🎬 Live 2D Playback")
        
#         # We only want to run the animation if the user clicks "Play" 
#         # because rendering hundreds of images takes a few seconds.
#         if st.button("Play Animation"):
#             # Create a placeholder on the screen where the images will flash
#             animation_placeholder = st.empty()
            
#             # Initialize the Renderer
#             renderer = SimulationRenderer(df, config)
            
#             # Get a sorted list of every unique time step (0.1, 0.2, 0.3...)
#             time_steps = sorted(df['time'].unique())
            
#             # Start the Movie Loop
#             for t in time_steps:
#                 # 1. Ask the renderer to draw the frame for this exact millisecond
#                 fig = renderer.get_frame(t)
                
#                 # 2. Display the image in the Streamlit placeholder
#                 animation_placeholder.pyplot(fig)
                
#                 # 3. Close the figure from memory so your computer doesn't crash!
#                 plt.close(fig)
                
#                 # 4. Wait a tiny fraction of a second to control the playback speed (FPS)
#                 time.sleep(0.01) 
            
#             st.success("Playback Complete!")

