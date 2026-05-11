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

    # --- NEW: ALTAIR BUG FIX (Create unique segments) ---
    # Every time the state changes, we increment a segment counter. 
    # This stops Altair from connecting identical colors across time gaps.
    chart_data['segment_id'] = chart_data.groupby('id')['state_change'].cumsum()
    
    # C. COLOR LOGIC (Refined)
    def get_inner_color(row):
        state_str = str(row['state'])

        # BUS LOGIC
        if row['type'] == 'Bus':
            if "SHOCKED" in state_str: return 'black'   
            if state_str == 'IN_BAY': return '#333333'  
            
            if state_str == 'WAITING_TO_MERGE': 
                presence = float(row['presence'])
                
                # DYNAMIC COLOR CALCULATION (Matches Physics)
                lane_width = 3.5
                safety_gap = 0.5
                bus_width = 2.5
                car_width = 1.9
                
                space_left = lane_width - (presence * bus_width)
                space_needed = car_width + safety_gap
                
                if space_left >= space_needed:
                    return '#FFD700' # Gold (Creeping, car can still pass)
                else:
                    return '#FF0000' # Red (Lane Blocked!)
                    
            return '#ffb3b3' # FIX: Light red instead of white for MOVING   
        
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
        detail=['id', 'segment_id'], # <--- FIX: Added segment_id here!
        tooltip=[
            alt.Tooltip('id', title='Vehicle ID'),
            alt.Tooltip('type', title='Type'),
            alt.Tooltip('class', title='Class'),
            alt.Tooltip('x', format='.1f', title='Absolute Pos (m)'),
            alt.Tooltip('v', format='.1f', title='Speed (m/s)'),
            alt.Tooltip('state', title='Status'),
            alt.Tooltip('presence', format='.2f', title='Own Solidity'),
            alt.Tooltip('front_gap', format='.2f', title='Front-Rear Gap (m)'),
            alt.Tooltip('lateral_gap', format='.2f', title='Lateral Gap (m)'),
            alt.Tooltip('physical_collision', title='⚠️ Physical Collision (both gaps <0)'),
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

    # --- Merge Zone 2D View (lateral safety check) ---
    st.divider()
    st.subheader("📐 Merge Zone: Top-Down 2D View")
    st.caption(
        "X = road position (m). Y = lateral offset from lane centre (m). "
        "Vehicle rectangles are **rotated by their real heading angle**. "
        "**Overlapping polygons = physical collision.**"
    )

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.transforms as mtransforms
    import io

    # ── Geometry constants (must match simulation.py / vehicle.py) ───────────
    MERGE_X_MIN   = 255.0    # window start
    MERGE_X_MAX   = 345.0    # window end
    BUS_STOP_X    = config.bus_stop_position   # 300.0
    LANE_W        = 3.5
    # Bus bay: flat section from (BUS_STOP_X - 15) to (BUS_STOP_X + 15),
    # with a 20 m entry taper and a 15 m exit taper.
    # All tapers are triangles joining the main-lane kerb (y = -LANE_W/2)
    # to the bay outer kerb (y = -LANE_W - LANE_W/2 = -LANE_W*1.5 ... actually
    # the bay centre is at y = -LANE_W, so outer edge at y = -LANE_W - LANE_W/2)
    BAY_OUTER_Y   = -LANE_W - LANE_W / 2   # outer kerb of bay = −5.25 m
    LANE_INNER_Y  = -LANE_W / 2            #  inner kerb of main lane = −1.75 m
    BAY_FLAT_X0   = BUS_STOP_X - 15.0      # 285
    BAY_FLAT_X1   = BUS_STOP_X + 15.0      # 315  (bus sits here when stopped)
    ENTRY_TAPER_L = 20.0                   # entry taper length (before flat)
    EXIT_TAPER_L  = 15.0                   # exit taper length  (after flat)
    # taper start/end x coords
    ENTRY_TAPER_X0 = BAY_FLAT_X0 - ENTRY_TAPER_L   # 265
    EXIT_TAPER_X1  = BAY_FLAT_X1 + EXIT_TAPER_L    # 330

    merge_df_all = df[(df['x'] >= MERGE_X_MIN) & (df['x'] <= MERGE_X_MAX)].copy()

    if merge_df_all.empty:
        st.info("No vehicles entered the merge zone in this run.")
    else:
        # ── Per-frame lateral centre ─────────────────────────────────────────
        def lateral_y(row):
            if row['type'] == 'Bus':
                # Use lateral_offset directly — it is synced for all states:
                # DECELERATING via bicycle model, WAITING_TO_MERGE via Phase-D progress sync.
                return float(row['lateral_offset'])
            total_lat = row.get('total_lat', 0.0)
            return float(total_lat) if not pd.isna(total_lat) else 0.0

        merge_df_all = merge_df_all.copy()
        merge_df_all['y_centre']  = merge_df_all.apply(lateral_y, axis=1)
        merge_df_all['half_w']    = merge_df_all['width']  / 2.0
        merge_df_all['half_l']    = merge_df_all['length'] / 2.0
        merge_df_all['heading']   = merge_df_all['heading_deg'].fillna(0.0) if 'heading_deg' in merge_df_all.columns else 0.0
        if 'heading_anomaly' not in merge_df_all.columns:
            merge_df_all['heading_anomaly'] = False

        # ── Rotated-polygon collision detection (per frame) ──────────────────
        def rect_corners(cx, cy, half_l, half_w, angle_deg):
            """Return 4 corners of a rectangle rotated by angle_deg around (cx,cy)."""
            θ = np.radians(angle_deg)
            c, s = np.cos(θ), np.sin(θ)
            dx = np.array([ half_l,  half_l, -half_l, -half_l])
            dy = np.array([ half_w, -half_w, -half_w,  half_w])
            xs = cx + dx * c - dy * s
            ys = cy + dx * s + dy * c
            return np.stack([xs, ys], axis=1)   # (4,2)

        def separating_axis_overlap(poly_a, poly_b):
            """SAT check for two convex polygons (4 corners each). True = overlap."""
            def axes(poly):
                n = len(poly)
                return [(poly[(i+1)%n] - poly[i]) for i in range(n)]
            def project(poly, ax):
                dots = poly @ ax
                return dots.min(), dots.max()
            for poly in (poly_a, poly_b):
                for edge in axes(poly):
                    normal = np.array([-edge[1], edge[0]])
                    minA, maxA = project(poly_a, normal)
                    minB, maxB = project(poly_b, normal)
                    if maxA < minB or maxB < minA:
                        return False
            return True

        collision_times = set()
        for t_val, grp in merge_df_all.groupby('time'):
            bus_rows = grp[grp['type'] == 'Bus']
            car_rows = grp[grp['type'] == 'Car']
            if bus_rows.empty or car_rows.empty:
                continue
            br = bus_rows.iloc[0]
            bus_poly = rect_corners(br['x'], br['y_centre'], br['half_l'], br['half_w'],
                                    float(br['heading']) if not pd.isna(br['heading']) else 0.0)
            for _, cr in car_rows.iterrows():
                car_poly = rect_corners(cr['x'], cr['y_centre'], cr['half_l'], cr['half_w'],
                                        float(cr['heading']) if not pd.isna(cr['heading']) else 0.0)
                if separating_axis_overlap(bus_poly, car_poly):
                    collision_times.add(t_val)
                    break

        merge_df_all['collision_frame'] = merge_df_all['time'].isin(collision_times)

        n_collision_times = len(collision_times)
        if n_collision_times:
            st.warning(f"⚠️ {n_collision_times} collision frame(s) detected — orange cars below.")
        else:
            st.success("✅ No physical overlaps detected in merge zone.")

        # ── Pre-render all frames to PNG bytes (fast scrubbing) ─────────────
        def _draw_bay(ax):
            """Draw road + accurate trapezoidal bus bay."""
            # Main traffic lane
            ax.axhspan(-LANE_W/2, LANE_W/2, color='#555555', zorder=0)
            ax.axhline( LANE_W/2,   color='white', ls='--', lw=1.5, zorder=1)
            ax.axhline(-LANE_W/2,   color='white', ls='-',  lw=2.0, zorder=1)

            # Trapezoidal bus bay (entry taper + flat section + exit taper)
            # Polygon vertices going clockwise from inner-lane kerb:
            bay_poly = np.array([
                [ENTRY_TAPER_X0,  LANE_INNER_Y],   # entry taper start (on main lane kerb)
                [BAY_FLAT_X0,     BAY_OUTER_Y ],   # entry taper end   (bay outer kerb)
                [BAY_FLAT_X1,     BAY_OUTER_Y ],   # flat section end
                [EXIT_TAPER_X1,   LANE_INNER_Y],   # exit taper end    (back to main)
                [EXIT_TAPER_X1,  -LANE_W/2    ],   # follow inner kerb back
                [ENTRY_TAPER_X0, -LANE_W/2    ],
            ])
            from matplotlib.patches import Polygon as MplPolygon
            bay_patch = MplPolygon(bay_poly, closed=True, facecolor='#444444',
                                   edgecolor='white', linewidth=0.8, zorder=0)
            ax.add_patch(bay_patch)
            # Outer bay kerb line
            ax.plot([BAY_FLAT_X0, BAY_FLAT_X1], [BAY_OUTER_Y, BAY_OUTER_Y],
                    color='white', ls='-', lw=1.0, zorder=1)

            # Bus stop marker
            ax.axvline(BUS_STOP_X, color='yellow', ls=':', lw=1.2, alpha=0.7, zorder=1)
            ax.text(BUS_STOP_X, BAY_OUTER_Y + 0.2, 'STOP', color='yellow',
                    ha='center', va='bottom', fontsize=7, zorder=2)

        def _render_frame(t_val):
            frame = merge_df_all[merge_df_all['time'] == t_val]
            fig, ax = plt.subplots(figsize=(16, 5))
            _draw_bay(ax)

            is_collision = t_val in collision_times
            for _, row in frame.iterrows():
                angle = float(row['heading']) if not pd.isna(row['heading']) else 0.0
                color = ('#e74c3c' if row['type'] == 'Bus'
                         else ('#FF8C00' if row['collision_frame'] else '#3498db'))
                corners = rect_corners(row['x'], row['y_centre'],
                                       row['half_l'], row['half_w'], angle)
                poly_patch = mpatches.Polygon(corners, closed=True,
                                              facecolor=color, edgecolor='white',
                                              linewidth=1.4, zorder=5)
                ax.add_patch(poly_patch)
                if row['type'] == 'Bus':
                    hdg = float(row['heading']) if not pd.isna(row['heading']) else 0.0
                    anomaly = bool(row.get('heading_anomaly', False))
                    label = f"{'⚠ ' if anomaly else ''}BUS\n{row.get('v', 0.0):.1f}m/s\n{hdg:+.0f}°"
                    # Red edge border when a heading-snap anomaly has been detected
                    if anomaly:
                        poly_patch.set_edgecolor('yellow')
                        poly_patch.set_linewidth(3.0)
                else:
                    label = f"{int(row['id'])}\n{row.get('v', 0.0):.1f}m/s"
                ax.text(row['x'], row['y_centre'], label,
                        color='white', ha='center', va='center',
                        fontsize=6.5, zorder=6)

            ax.set_xlim(MERGE_X_MIN, MERGE_X_MAX)
            ax.set_ylim(BAY_OUTER_Y - 0.4, LANE_W / 2 + 0.5)
            ax.set_xlabel("Road position (m)")
            ax.set_ylabel("Lateral (m)")
            ax.set_title(
                f"Merge Zone Top-Down  |  t = {t_val:.1f} s"
                + ("  ⚠️ COLLISION FRAME" if is_collision else ""),
                fontsize=11, color='red' if is_collision else 'black'
            )
            ax.set_aspect('equal')
            ax.legend(handles=[
                mpatches.Patch(color='#e74c3c', label='Bus'),
                mpatches.Patch(color='#3498db', label='Car'),
                mpatches.Patch(color='#FF8C00', label='Collision'),
            ], loc='upper right', fontsize=9)
            ax.grid(axis='x', color='white', alpha=0.15, zorder=0)

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return buf.read()

        all_times_full = sorted(merge_df_all['time'].unique())

        # ── Narrow render window to the relevant merge phase ─────────────────
        # Start: 3 s before bus first transitions to WAITING_TO_MERGE
        # End:   bus front position exits MERGE_X_MAX
        bus_merge_rows = merge_df_all[
            (merge_df_all['type'] == 'Bus') & (merge_df_all['state'] == 'WAITING_TO_MERGE')]
        bus_exit_rows  = merge_df_all[
            (merge_df_all['type'] == 'Bus') & (merge_df_all['x'] > MERGE_X_MAX - merge_df_all['length'] / 2)]
        if not bus_merge_rows.empty:
            t_window_start = max(merge_df_all['time'].min(),
                                 bus_merge_rows['time'].min() - 3.0)
        else:
            t_window_start = merge_df_all['time'].min()
        t_window_end = bus_exit_rows['time'].min() if not bus_exit_rows.empty \
                       else merge_df_all['time'].max()
        all_times = [t for t in all_times_full if t_window_start <= t <= t_window_end]
        if not all_times:
            all_times = all_times_full

        # Cache rendered frames in session state keyed by run hash
        run_hash = str(hash(str(df['time'].max()) + str(len(df))))
        cache_key = f"merge_frames_{run_hash}"
        if cache_key not in st.session_state:
            with st.spinner(f"Pre-rendering {len(all_times)} merge frames…"):
                st.session_state[cache_key] = {t: _render_frame(t) for t in all_times}
        frame_images = st.session_state[cache_key]

        # ── Speed-controllable animated GIF ──────────────────────────────────
        # Encoding is fast (PIL only, no ffmpeg). Cached per fps value.
        fps_options = [1, 2, 3, 5, 8, 10, 15, 20]
        fps = st.select_slider(
            "Playback speed (frames/sec)",
            options=fps_options,
            value=5,
            key=f"fps_{run_hash}",
            format_func=lambda v: f"{v} fps  ({len(all_times)/v:.0f} s video)"
        )
        gif_key = f"merge_gif_{run_hash}_{fps}"
        if gif_key not in st.session_state:
            from PIL import Image as _PILImg
            with st.spinner("Encoding animation…"):
                pil_frames = [_PILImg.open(io.BytesIO(frame_images[t])).convert('RGBA')
                              for t in all_times]
                gif_buf = io.BytesIO()
                pil_frames[0].save(
                    gif_buf, format='GIF', save_all=True,
                    append_images=pil_frames[1:],
                    duration=int(1000 / fps),
                    loop=0, optimize=False,
                )
                st.session_state[gif_key] = gif_buf.getvalue()

        st.image(st.session_state[gif_key], use_container_width=True)
        st.caption(
            f"Window: t = {all_times[0]:.1f} s → {all_times[-1]:.1f} s  |  "
            f"{len(all_times)} frames  |  1 frame = {df['time'].diff().median():.2g} s simulation time"
        )

        # ── Single-frame inspector (step-through, inside expander) ───────────
        with st.expander("🔍 Single-frame inspector (step through frames)"):
            slider_key = f"tslider_{run_hash}"
            if slider_key not in st.session_state:
                first_ct = min((t for t in collision_times if t in all_times), default=all_times[0])
                st.session_state[slider_key] = first_ct
            if st.session_state[slider_key] not in all_times:
                st.session_state[slider_key] = all_times[0]

            def _cur_idx():
                return all_times.index(st.session_state[slider_key])

            c1, c2, c3, c4 = st.columns([1, 1, 1, 6])
            with c3:
                n_step = st.number_input("Step n", 1, 50, 5, key=f"nstep_{run_hash}")
            with c1:
                if st.button("◀ −n", key=f"prev_{run_hash}"):
                    st.session_state[slider_key] = all_times[max(0, _cur_idx() - int(n_step))]
                    st.rerun()
            with c2:
                if st.button("▶ +n", key=f"next_{run_hash}"):
                    st.session_state[slider_key] = all_times[min(len(all_times)-1, _cur_idx() + int(n_step))]
                    st.rerun()
            with c4:
                st.select_slider("Time (s)", options=all_times,
                                 format_func=lambda t: f"{t:.1f} s", key=slider_key)
            st.image(frame_images[st.session_state[slider_key]], use_container_width=True)

        # ── SAT collision debug / visual validation ───────────────────────────
        with st.expander("🔬 SAT collision validation — corner-point overlay"):
            st.markdown(
                "**SAT (Separating Axis Theorem) proof of correctness**: for two convex polygons, "
                "if no separating axis exists among the 8 edge-normals (4 per polygon), the polygons "
                "overlap — this is a necessary AND sufficient condition (zero false positives/negatives). "
                "The yellow dots below are the computed corner positions. "
                "Verify they sit exactly at the physical edges of each vehicle."
            )
            debug_t = st.select_slider(
                "Frame to inspect", options=all_times,
                format_func=lambda t: f"{t:.1f} s",
                key=f"sat_debug_{run_hash}"
            )
            frame_dbg = merge_df_all[merge_df_all['time'] == debug_t]
            fig_d, ax_d = plt.subplots(figsize=(16, 5))
            _draw_bay(ax_d)
            for _, row in frame_dbg.iterrows():
                angle = float(row['heading']) if not pd.isna(row['heading']) else 0.0
                color = '#e74c3c' if row['type'] == 'Bus' else '#3498db'
                corners = rect_corners(row['x'], row['y_centre'],
                                       row['half_l'], row['half_w'], angle)
                ax_d.add_patch(mpatches.Polygon(corners, closed=True,
                               facecolor=color, edgecolor='white',
                               linewidth=1.2, alpha=0.55, zorder=5))
                ax_d.scatter(corners[:,0], corners[:,1],
                             c='yellow', s=30, zorder=8)
                arrow_len = row['half_l'] * 0.7
                ax_d.annotate('', xy=(row['x'] + arrow_len * np.cos(np.radians(angle)),
                                      row['y_centre'] + arrow_len * np.sin(np.radians(angle))),
                              xytext=(row['x'], row['y_centre']),
                              arrowprops=dict(arrowstyle='->', color='lime', lw=1.8), zorder=9)
                ax_d.text(row['x'], row['y_centre'],
                          f"{int(row['id'])} {angle:.1f}°",
                          color='white', ha='center', va='center', fontsize=7, zorder=10)
            ax_d.set_xlim(MERGE_X_MIN, MERGE_X_MAX)
            ax_d.set_ylim(BAY_OUTER_Y - 0.4, LANE_W / 2 + 0.5)
            ax_d.set_aspect('equal')
            ax_d.set_title(f"Corner verification  t = {debug_t:.1f} s  |  "
                           f"{'⚠️ SAT: COLLISION' if debug_t in collision_times else '✅ SAT: clear'}")
            st.pyplot(fig_d, use_container_width=True)
            plt.close(fig_d)

        # ── Bus Entry Telemetry Table ─────────────────────────────────────────
        with st.expander("📊 Bus Entry Telemetry (frame-by-frame)", expanded=False):
            st.markdown(
                "Frame-by-frame bus state from **first DECELERATING step** through "
                "**bay stop**.  Use this to diagnose steering oscillation and "
                "confirm the parking crawl behaviour."
            )
            bus_telem = merge_df_all[merge_df_all['type'] == 'Bus'][
                ['time', 'v', 'lateral_offset', 'heading_deg', 'steer_angle_deg', 'state']
            ].copy()
            entry_states = {'DECELERATING', 'IN_BAY'}
            bus_telem = bus_telem[bus_telem['state'].isin(entry_states)].copy()
            bus_telem = bus_telem.rename(columns={
                'time':           'Time (s)',
                'v':              'Vel (m/s)',
                'lateral_offset': 'Actual Y (m)',
                'heading_deg':    'Heading (°)',
                'steer_angle_deg':'Steer (°)',
                'state':          'State',
            })
            bus_telem = bus_telem.reset_index(drop=True)
            if bus_telem.empty:
                st.info("No DECELERATING/IN_BAY frames in the recorded window.")
            else:
                st.dataframe(
                    bus_telem.style.format({
                        'Time (s)':    '{:.2f}',
                        'Vel (m/s)':   '{:.3f}',
                        'Actual Y (m)':'{:.4f}',
                        'Heading (°)': '{:.4f}',
                        'Steer (°)':   '{:.4f}',
                    }),
                    use_container_width=True,
                    height=400,
                )

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

