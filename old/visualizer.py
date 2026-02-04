# visualizer.py
import streamlit as st
import pandas as pd
import altair as alt

def plot_cost_comparison(result_a, result_b):
    """Draws the bar chart comparing costs."""
    data = pd.DataFrame([
        {"Scenario": "Stop in Lane", "Cost": result_a['total_cost'], "Type": "Car Delay"},
        {"Scenario": "Bus Bay", "Cost": result_b['total_cost'], "Type": "Passenger Delay"}
    ])
    
    chart = alt.Chart(data).mark_bar().encode(
        x='Scenario',
        y='Cost',
        color='Type',
        tooltip=['Scenario', 'Cost']
    ).properties(height=300)
    
    st.altair_chart(chart, use_container_width=True)

def draw_road_snapshot(scenario_name, cars_queued):
    """
    Visualizes the road state using HTML/CSS.
    """
    st.markdown(f"#### Visual Snapshot: {scenario_name}")
    
    # CSS Styles for our little "game" view
    st.markdown("""
    <style>
    .road { background-color: #333; padding: 20px; border-radius: 10px; display: flex; align-items: center; gap: 5px; overflow-x: auto;}
    .car { width: 40px; height: 25px; background-color: #ff4b4b; border-radius: 4px; border: 1px solid #fff; }
    .bus { width: 100px; height: 35px; background-color: #4CAF50; border-radius: 5px; border: 2px solid #fff; display: flex; justify-content: center; align-items: center; color: white; font-weight: bold;}
    .lane-marker { border-bottom: 2px dashed #fff; flex-grow: 1; margin: 0 10px;}
    .bay { border: 2px solid yellow; padding: 5px; border-radius: 5px; margin-left: 10px;}
    </style>
    """, unsafe_allow_html=True)

    # Logic to build the HTML string
    html = '<div class="road">'
    
    if scenario_name == "Lane Stop":
        # Draw cars stuck behind bus
        for _ in range(int(cars_queued)):
            html += '<div class="car" title="Stuck Car">🚗</div>'
        html += '<div class="bus">BUS</div>' # Bus blocking the way
        html += '<div class="lane-marker"></div>' # Empty road ahead
        
    elif scenario_name == "Bus Bay":
        # Draw flowing traffic (just a few cars passing by)
        html += '<div class="car">🚗</div>'
        html += '<div class="car">🚗</div>'
        html += '<div class="car">🚗</div>'
        # Bus inside a special "bay" container
        html += '<div class="bay"><div class="bus">BUS</div></div>'
        
    html += '</div>'
    
    st.markdown(html, unsafe_allow_html=True)
    if cars_queued > 0:
        st.caption(f"🛑 Traffic Jam: {cars_queued} cars stuck behind the bus.")
    else:
        st.caption("✅ Traffic flowing freely.")