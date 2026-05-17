# Bus Merge PPO Agent — Project Diary

> This file is the **chronological work log** — what was done, when, and why.
> It is append-only; old entries are never deleted.

---

## Phase 0 — Concept & Setup (early 2026)

- Road simulation in Python: single-lane traffic, one bus, `dt=0.1 s`, `road_length=1000 m`.
- Bus stops at `x=300 m` in a side bay, then merges back into traffic.
- PPO agent controls the bus's merge throttle (action = desired merge progress delta).
- Initial state vector: 6 features — `[gap, speed, accel, merge_progress, width, rel_speed]`.
- First training: basic convergence achieved, but agent was fragile across traffic densities.

---

## Phase 1 — Domain Randomization ✅ (2026-03-~20)

**Problem**: Agent trained at a fixed traffic density failed at lighter/heavier densities.

**Changes**:
- Randomized per-episode: traffic rate, car speeds, gaps between vehicles.
- Added vehicle-class diversity (Compact, Mid, SUV, Truck, Aggressive).
- Per-class IDM parameters (aggression, max speed, deceleration).

**Result**: Improved generalization across traffic weights.

---

## Phase 2 — Adversarial Profiles & Distribution ✅ (2026-03-25)

**Problem**: Agent struggled with tailgaters and non-yielders.

**Changes**:
- Added "Aggressive" class: low politeness, tight IDM gaps, ignores bus presence more.
- Added 3 adversarial training scenarios (A, B, C corresponding to density levels).
- Scenario D (10% probability): easy baseline to prevent catastrophic forgetting.

**Gemini review key points**:
- Noise σ curriculum: start at 0.05, grow to 0.30. Too much noise early = no convergence.
- C1 aggressive tailgater is the hardest and most important adversarial case.
- D1 Shield penalty formula: `Penalty = BasePenalty × (1 + Epoch/MaxEpoch)`.

---

## Phase 3 — Human Realism (E1/E2/E3) ✅ (2026-03-29 → 2026-04-04)

**Problem**: Cars were perfectly rational IDM machines. Real drivers are imperfect.

**State vector expanded 6→11 features**:
- Added: `lane_width`, `bus_angle`, `lat_clearance`, `nearest_car_width`, `merge_progress_sq` (non-linear).

**Three Pillars** (Weber noise, reaction delay, squeeze behavior):
- **Weber noise**: perceptual noise scales with magnitude (σ = k·x). Larger gaps = larger absolute uncertainty.
- **Reaction timer**: stochastic per-driver (Gaussian around class mean). Bus presence change triggers shock reaction delay.
- **Squeeze behavior**: politeness-driven. Aggressive class squeezes more; compact class is compliant.

**Run #1 results (2026-03-30)**:
- Validation: 378/400 (94.5%). Mild overfitting from earlier 120/120 on 3 seeds.
- Key bug found: `emergency_stop` deadlock — car freezes at gap=0.2m, bus stays stuck → 4910 shields/ep.

---

## Phase 4 — Steering-First Lateral Model ✅ (2026-04-12 → 2026-04-15)

**Problem 1**: Cars had an unrealistic "velocity first" lateral model — lateral velocity was directly set by S-curve. No physical realism.

**Changes — vehicle.py**:
- Replaced with **steering-first bicycle model** (driver intent → steer angle → heading → lateral position).
- Per-class: `max_steer_angle_deg`, `steer_rate_deg_per_s`, `wheelbase`.
- `heading_angle` (rad) is now the physical cause; `lateral_velocity` is the result (`v * sin(θ)`).
- **Politeness sampled per driver** (not fixed per class): Beta(2,2) normal; Beta(1,6) Aggressive.
- All behavior params sampled stochastically: steer_rate LogNormal CV=20%, lane_noise LogNormal.
- **Ornstein-Uhlenbeck lane drift** (`lane_drift`) — imperfect lane keeping for all vehicles incl. bus.
- `total_lateral_offset = lateral_offset + lane_drift` property.

**Problem 2**: 2D merge view in app.py was too rough.

**Changes — app.py 2D view (first draft, 2026-04-12)**:
- Added rotated rectangle patches using `heading_deg`.
- Added SAT (Separating Axis Theorem) polygon collision detection — replaces naive AABB.
- Trapezoid bus bay geometry (entry taper 20 m, flat section 30 m, exit taper 15 m).
- Pre-rendered PNG frames cached in `st.session_state` (compute once, display fast).
- ◀ −n / ▶ +n step buttons + `n` selector.

**Problem 3**: Bus heading was always showing 0° — `heading_angle` inherited from Vehicle but overwritten by car steering model.

**Root cause (found 2026-04-15)**:
- `_handle_bus_logic()` sets `bus.lateral_velocity` (entry/exit taper logic).
- Then `update_physics()` for the bus runs the car steering-first model (target = 0), which **overwrites** `lateral_velocity` with `v * sin(heading)` ≈ 0.
- So at frame-log time, `heading_angle` ≈ 0 for bus.

**Fix (2026-04-15)**:
- In `_record_frame`, compute bus heading from consecutive frame y-centre change:  
  `heading_deg = arctan2(cur_y − prev_y, v × dt)`  
  This uses `presence_factor` which IS correctly updated by the merge logic.

**Additional fixes (2026-04-15 — session 2)**:
- **Bus heading capture refactored**: Removed the fragile "Δy between frames" workaround. Instead, `_handle_bus_logic()` now stores `bus._logged_heading_deg = arctan2(lateral_velocity, velocity)` immediately after setting `lateral_velocity` — before `update_physics()` can overwrite it. Both DECELERATING and WAITING_TO_MERGE states covered.
  - DECELERATING (entry taper): heading ≈ −7° (bus angling into bay)
  - WAITING_TO_MERGE (merging): heading ≈ +9–13° depending on PPO merge speed
  - Paused waiting for gap: heading = 0° (correctly — bus is not moving laterally)
- **2D view redesigned as animated GIF** with fps speed selector (1–20 fps). PIL GIF creation is fast (no ffmpeg dependency). Cached per fps value. Zero Streamlit re-renders on playback.
- **Single-frame inspector** moved into a collapsible expander (step navigation + slider).
- **SAT validation expander** added: shows polygon corner dots (yellow) and heading arrows on any frame. User can visually confirm corner positions match physical vehicle edges.

---

## Next up (as of 2026-04-15)

- Training Run #4 with new Phase 4 state features and refined vehicle physics.
- Decision: fix bus `update_physics()` to not run the car steering model (prevents silent lateral_velocity override).
- Validate that the bus angle appears correctly in new GIF renders.

---

## Phase 5 — Physics Bug Fixes + Bus Stochasticity ✅ (2026-04-28 → 2026-05-11)

### Bug Fixes (from telemetry file `2026-04-28T13-48_export.csv`)

**Ghost 5 — Sudden Speed Drop at DECELERATING Entry (vehicle.py)**
- At 50 m from stop, bus entered DECELERATING. With v≈13.9 m/s and gap=50 m, uncapped IDM computed ≈−3.0 m/s² on frame 1 — a comfort brake that appeared as an instantaneous jerk in telemetry.
- Fix: `self.acceleration = max(self._calculate_idm_accel(...), -1.5)` — hard floor at −1.5 m/s².

**Ghost 6 — S-Curve Bay Entry (simulation.py)**
- Old `entry_taper_length = 20 m`. Bay-entry ramp formula: `look_frac = (20 - dist + 18) / 20`. At dist=20 m, this gave `lat_target = −3.15 m` immediately — bus S-curved sharply instead of smoothly entering.
- Fix: `entry_taper_length = 50 m`. New formula: `look_frac = (50 - dist) / 32`. True linear 0→−3.5 m ramp over 32 m starting from DECELERATING trigger.

**Ghost 7 — Virtual Leader Scope (confirmed NOT a bug)**
- Suspected `lead_vehicle` (fake stationary leader at stop) might leak into car IDM. Confirmed clean: reset to `None` at top of each loop iteration; only assigned inside `isinstance(veh, Bus)`. Documented with clarifying comment.

### Bus Stochasticity (simulation.py + vehicle.py)

Previously bus was fully deterministic: fixed spawn speed (10.0 m/s), fixed stop position (300 m), fixed steer rate. Same conditions every episode = agent memorizes one sequence. Six noise sources added:

1. **Spawn speed**: `Uniform(8.5, 11.5)` m/s
2. **Stop position**: `Normal(300, σ=5)` m, clipped [285, 315]
3. **Steer rate per episode**: `LogNormal(25°/s, CV=10%)`, clipped [18, 33] °/s
4. **IDM perception noise**: gap perceived with `σ = max(0.3 m, gap × 2%)` — radar uncertainty
5. **Micro-correction noise**: `σ = 0.02 + 0.06 × |accel|/1.5` — larger under hard braking
6. **Active OU lane perturbation**: OU process `_lat_perturb` added to `lat_target` before Pure Pursuit.
   - OLD: OU drift was additive to `total_lateral_offset` AFTER physics — controller never saw it.
   - NEW: OU perturbs `effective_lat_target`; Pure Pursuit controller actively steers to correct it.
   - Result: realistic small oscillations, not uncorrected drift.

### Pure Pursuit Controller (Bus Lateral)

The bus now uses a **kinematic bicycle model + Pure Pursuit controller** for lateral motion:
- `wheelbase = 7 m`, `max_steer = 15°`, `L_PP = 18 m`
- Steer rate is mechanically limited (rate-limited per step)
- Formula: `y_rel = Δy·cos(ψ) − L_PP·sin(ψ)` → `δ = arctan2(2L·y_rel, L_PP²)`
- The `−L_PP·sin(ψ)` term automatically generates counter-steer when heading overshoots bay angle.
  No separate "straighten" logic needed — intrinsic to Pure Pursuit geometry.

### Checkpoint Path Fix (simulation.py)

- Checkpoint was stored at `checkpoints/ppo_bus_brain.pth` locally but hard-coded as `rl/checkpoints/...`.
- Fix: runtime path search across 3 candidate locations (`checkpoints/`, `rl/checkpoints/`, `__file__`-relative).

### Deployment to Streamlit Cloud

- Created `requirements.txt` (streamlit, numpy, pandas, altair, matplotlib, Pillow, torch).
- Created `.streamlit/config.toml` (dark theme, headless=true).
- Copied `checkpoints/ppo_bus_brain.pth` → `rl/checkpoints/ppo_bus_brain.pth` (42 KB, committed).
- GitHub repo made **public** (Streamlit Cloud free tier requires public repos).
- Deployment URL: https://share.streamlit.io → `yonizilber/bus_simulation`, branch `feature/agent-smart-idm-logic_4_4`, main file `app.py`.

**Cloud bug discovered**: `action_mean[0, 1]` crashed on cloud because checkpoint was trained with `action_dim=1` but cloud used a different (older) `app.py` version where the shape mismatch was not caught.
- Fix 1: synced all modified files (`app.py`, `graphics/renderer.py`, `rl/env_wrapper.py`) to GitHub.
- Fix 2: added defensive check `if action_mean.shape[1] > 1 else 0.0` — crash-safe fallback.

---

## Next up (as of 2026-05-17)

See `PROJECT_PLANS.md` for prioritised list. Top items:
1. **Training Run #4** — retrain with Phase 4+5 physics (11-feature state, bicycle model, OU active correction)
2. **Fix bus `update_physics()` lateral override** (see Plans item #1 — `skip_lateral_steering` flag)
3. **Emergency stop deadlock fix** (Plans item #3 — IDM minimum gap behind stopped bus)
4. **Validate cloud deployment** — ensure Scenario B (bus bay) runs end-to-end without crash

