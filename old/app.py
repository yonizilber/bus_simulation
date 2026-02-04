import streamlit as st
from simulation_engine import TrafficSimulation
from visualizer import plot_cost_comparison
# Import the new animation function
from traffic_animation import traffic_animation 

st.set_page_config(page_title="Bus Sim Animated", layout="wide")
st.title("🚌 Traffic Simulation: The Animation")

# --- Inputs ---
with st.sidebar:
    st.header("Settings")
    passengers = st.slider("Bus Passengers", 0, 100, 40)
    stop_time = st.slider("Stop Time (sec)", 10, 120, 40)
    merge_time = st.slider("Merge Time (sec)", 5, 60, 20)
    traffic = st.slider("Traffic (cars/min)", 5, 60, 30)

# --- Logic ---
sim = TrafficSimulation(traffic, stop_time, merge_time, passengers)
res_lane = sim.run_lane_stop_scenario()
res_bay = sim.run_bus_bay_scenario()

# --- Visualization (The Animated Part) ---
col1, col2 = st.columns(2)


with col1:
    st.subheader("Scenario A: Stop in Lane")
    traffic_animation("lane", stop_time, traffic, merge_time)
    
    # Check for gridlock warning from the engine
    if res_lane['total_cost'] > 50000:
        st.error("🚨 GRIDLOCK ALERT: Traffic volume is too high!")
    else:
        st.error(f"Time Wasted by Cars: {int(res_lane['cost_cars'])}s")
        st.caption(res_lane.get('note', '')) # Shows "Queue clears in X seconds"

with col2:
    st.subheader("Scenario B: Bus Bay")
    # Call the animation component
    traffic_animation("bay", stop_time, traffic, merge_time)
    
    st.warning(f"Time Wasted by Bus Passengers: {int(res_bay['cost_bus'])}s")
    st.caption("Notice cars flow freely, but the Bus (Orange) waits to merge.")

# --- Results ---
st.divider()
total_diff = res_lane['total_cost'] - res_bay['total_cost']

if total_diff > 0:
    st.success(f"🏆 Recommendation: **Build a Bus Bay!** (Saves {int(total_diff)} seconds total)")
else:
    st.info(f"🏆 Recommendation: **Keep the Lane!** (Bay is not worth the merge time)")

plot_cost_comparison(res_lane, res_bay)