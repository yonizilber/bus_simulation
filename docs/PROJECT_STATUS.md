# Bus Merge PPO Agent — Current Project Status

> **Last updated**: 2026-05-17
> This file is the **living technical reference** — architecture, physics, logic, known issues.
> Updated whenever the system changes significantly.

---

## How to Run Locally

Streamlit is installed in the **`bus_sim`** conda environment only. `conda activate` does not work from
a non-interactive shell (e.g., VS Code terminal launched from outside conda). Use one of these methods:

```bash
# Method 1 — full path (always works, no activation needed)
cd ~/bus_simulation
/home/yonizi/anaconda3/envs/bus_sim/bin/streamlit run app.py

# Method 2 — activate then run (works in a fresh interactive terminal)
conda activate bus_sim
cd ~/bus_simulation
streamlit run app.py
```

The app runs at **http://localhost:8501** by default.

> ⚠️ Do NOT run `streamlit run app.py` from the `base` conda environment — streamlit is not installed there.

## Deployment (Streamlit Cloud)

- GitHub repo: `yonizilber/bus_simulation` (must be **public**)
- Branch: `feature/agent-smart-idm-logic_4_4`
- Main file: `app.py`
- URL: configured at https://share.streamlit.io
- After any `git push`, Streamlit Cloud auto-redeploys. If stuck in error state: **Manage app → Reboot app**.

---

## Architecture Overview

```
traffic spawner ──► road simulation (simulation.py)
                       │
                       ├── Car × N   (vehicle.py: Vehicle class)
                       └── Bus × 1   (vehicle.py: Bus class)
                              │
                              └── PPO brain (rl/ppo_agent.py)
                                     reads 11-feature state → outputs merge delta
```

### Files

| File | Role |
|------|------|
| `simulation.py` | Main simulation loop, bus state machine, frame logger |
| `vehicle.py` | `Vehicle`, `Bus`, `BusState`, `CAR_PARAMS`, `sample_vehicle_params` |
| `app.py` | Streamlit dashboard — trajectory chart, GIF, 2D merge view |
| `graphics/renderer.py` | Matplotlib GIF renderer |
| `rl/ppo_agent.py` | PPO policy network |
| `rl/env_wrapper.py` | Gym-compatible env for training |
| `train.py` | Training loop |
| `docs/` | This documentation |

---

## Physics Model

### Road Geometry

| Parameter | Value |
|-----------|-------|
| `road_length` | 1000 m |
| `bus_stop_position` | 300 m |
| `lane_width` | 3.5 m |
| `dt` | 0.1 s |
| Bay flat section | x ∈ [285, 315] |
| Entry taper | 20 m (x ∈ [265, 285]) |
| Exit taper | 15 m (x ∈ [315, 330]) |
| Bay centre y | −3.5 m (one lane below traffic) |

### Car Physics (Vehicle.update_physics)

**Steering-first bicycle model** (as of Phase 4):

1. **Driver intent**: `lat_target` = `MAX_NUDGE × politeness` if squeezing, else 0
2. **Steering controller**: `δ_target = arctan2(lat_error, lookahead)`, clamped to `max_steer_angle_deg`
3. **Mechanical response**: `steer_angle` tracks `δ_target` at rate `steer_rate_deg_per_s`
4. **Bicycle model**: `dθ/dt = v × tan(δ) / wheelbase`
5. **Lateral output**: `lateral_velocity = v × sin(heading_angle)`
6. **O-U lane drift**: `lane_drift` follows Ornstein-Uhlenbeck (mean-reversion 0.4 s⁻¹)

**total_lateral_offset** = `lateral_offset + lane_drift`

### Bus Physics

- The bus uses `update_physics()` (inherited) for **longitudinal** motion (IDM forward velocity).
- **Lateral motion** is determined by the bus state machine in `_handle_bus_logic()`:
  - DECELERATING: `lateral_velocity = −v × width / entry_taper_length` (diagonal entry into bay)
  - WAITING_TO_MERGE: `lateral_velocity = (Δmerge_progress × 2.5) / dt` (merge step)
  - Other states: `lateral_velocity = 0`
- The bus's `heading_angle` attribute is **not reliable** (car steering model overwrites it to ~0).
- ⚠️ **Known issue**: `update_physics()` overwrites `bus.lateral_velocity` after `_handle_bus_logic()` sets it. The bug does not affect the simulation physics (presence/merge_progress are used for actual position), but it means `heading_angle` on the Bus object is wrong.
- **Workaround in effect**: `_record_frame` computes bus heading from consecutive `y_centre` frame delta — `arctan2(Δy, v·dt)`.

### Bus Presence Factor

```
state                   presence_factor
─────────────────────── ────────────────────────
MOVING                  1.0
DECELERATING (far)      1.0
DECELERATING (in taper) dist_to_stop / entry_taper_length (ramps 1→0)
IN_BAY                  0.0
STOPPED_IN_LANE         1.0
WAITING_TO_MERGE        _merge_progress (0→1)
MOVING (post merge)     1.0
```

Cars perceive the bus as `presence_factor` of its effective block width.

---

## RL State Vector (11 features)

| # | Feature | Description |
|---|---------|-------------|
| 0 | merge_progress | Bus merge fraction 0→1 |
| 1 | bus_angle_sin | sin of bus heading angle |
| 2 | lon_dist | Longitudinal gap nearest car → bus front |
| 3 | lat_clearance | Lateral clearance (side-to-side) |
| 4 | car_speed_norm | Nearest car speed (normalized) |
| 5 | rel_speed_norm | Relative speed bus−car |
| 6 | car_width_norm | Nearest car width |
| 7 | car_accel_norm | Nearest car acceleration |
| 8 | merge_progress_sq | Non-linear emphasis on late-merge |
| 9 | traffic_density | Vehicles per 100 m in window |
| 10 | (reserved) | — |

All features Weber-noised in training env (`σ = weber_k × |x|`).

---

## Vehicle Classes

| Class | Speed | Politeness (mean) | Aggression | Notes |
|-------|-------|--------------------|------------|-------|
| Compact | 11 m/s | Beta(2,2)≈0.5 | Low | Compliant yielder |
| Mid | 13 m/s | Beta(2,2)≈0.5 | Medium | Standard IDM |
| SUV | 12 m/s | Beta(2,2)≈0.5 | Medium | Wide, slow reaction |
| Truck | 10 m/s | Beta(2,2)≈0.5 | Low | Long, careful |
| Aggressive | 16 m/s | Beta(1,6)≈0.14 | High | Tailgates, ignores bus |

All params sampled per-driver via LogNormal/Normal spread (CV ≈ 20%).

---

## Frame Log Fields (simulation.py `_record_frame`)

| Field | Type | Description |
|-------|------|-------------|
| time | float | Simulation clock (s) |
| id | int | Vehicle ID (999 = bus) |
| type | str | "Bus" or "Car" |
| class | str | Vehicle class |
| x | float | Longitudinal position (m) |
| v | float | Velocity (m/s) |
| length, width | float | Vehicle dimensions |
| state | str | Bus BusState.name or car state label |
| presence | float | Presence factor (0..1) |
| lateral_offset | float | Intentional nudge offset (m) |
| lane_drift | float | O-U random lane-keeping error (m) |
| total_lat | float | total_lateral_offset = lateral_offset + lane_drift |
| steer_angle_deg | float | Current rack angle (car only) |
| heading_deg | float | Body yaw (degrees); for bus: derived from Δy/Δt |
| front_gap | float | Bumper-to-bumper to leader (m); NaN if no leader |
| lateral_gap | float | Side-to-side clearance to leader (m); NaN if no leader |
| physical_collision | bool | True if front_gap < 0 AND lateral_gap < 0 |
| leader_info | str | Leader telemetry |
| bus_visibility | str | Bus presence perceived |

---

## App.py UI Sections

1. **Config panel** — scenario selector, training hyperparams
2. **Run button** — triggers `Simulation.run()` → DataFrame
3. **Position-Time trajectory chart** (Altair) — all vehicles, hover tooltip
4. **Animated GIF** — full road view from renderer.py
5. **📐 Merge Zone 2D View** — per-frame top-down with:
   - Rotated polygon footprints (SAT collision)
   - True trapezoid bus bay
   - Pre-rendered PNG cache
   - ◀ −n / ▶ +n navigation + slider
   - Frame window: WAITING_TO_MERGE−3s … bus exits merge zone
6. **Results panel** — success/crash/timeout stats

---

## Known Issues & Workarounds

| Issue | Status | Workaround |
|-------|--------|------------|
| Bus `heading_angle` always ~0 | **fixed** | `_logged_heading_deg` stored in `_handle_bus_logic` before `update_physics` runs |
| Streamlit page-rerun on every button click | **mitigated** | Replaced slider with animated GIF (no re-render during playback). Step inspector inside expander still triggers rerun on button click, but is secondary. Full fix requires `@st.fragment` (Streamlit 1.33+) |
| `emergency_stop` deadlock (bus+car both frozen) | open — identified in Run #1 | Needs delta floor when car is fully stopped |
| Bus `update_physics` runs car steering model | open | Does not affect lat position (that uses presence), only affects heading_angle and lateral_offset. `_logged_heading_deg` workaround is in place. |

