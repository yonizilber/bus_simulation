# Bus Merge PPO Agent — Plans & Future Work

> **Last updated**: 2026-05-17
> Ordered roughly by priority. Move items to PROJECT_DIARY when completed.

---

## Immediate (before Run #4 training)

### 1. Fix Bus `update_physics` Lateral Override (HIGH)
**Problem**: `_handle_bus_logic()` sets `bus.lateral_velocity` for taper/merge steps, then `update_physics()` (inherited from Vehicle) overwrites it with `v * sin(heading_angle)` ≈ 0. This means:
- `lateral_offset` accumulates using the wrong velocity
- `heading_angle` never reflects the bus's actual diagonal movement
- The workaround (deriving heading from Δy in `_record_frame`) is a patch, not a fix.

**Proposed fix**:
- Add `self.skip_lateral_steering = False` flag to `Vehicle.__init__`.
- Set it to `True` in `Bus.__init__`.
- In `update_physics()`, wrap the entire steering-lateral block:
  ```python
  if not getattr(self, 'skip_lateral_steering', False):
      # steering → heading → lateral_velocity update
  ```
- The bus's `lateral_velocity` then stays as set by the simulation (correct diagonal velocity).
- The bus's `heading_angle` can then be explicitly set from `arctan2(lateral_velocity, velocity)` after the move step.

### 2. Training Run #4 (HIGH)
- Delete old checkpoint: confirm with user before deleting `rl/checkpoints/ppo_bus_brain.pth`
- Train with Phase 4 features (11-feature state vector, Weber noise, steering-first model)
- Use same eval seeds as Run #1: `123,456,777,42,999,2024`
- Watch for emergency_stop deadlock regression

### 3. Fix Emergency Stop Deadlock (HIGH)
**Problem (from Run #1)**: When bus is at merge progress < 0.30 and a car stops directly alongside (gap ≈ 0.2 m), the shield fires `delta=0`. The car stays frozen in IDM equilibrium. Bus stays frozen because delta=0. Deadlock.

**Fix options**:
- Option A: If `emergency_stop` counter exceeds N in same episode, apply a small positive delta anyway (break deadlock).
- Option B: Give the bus a small forced `delta_min` whenever the car behind has zero velocity and merge_progress < 0.5 (car is clearly yielded).
- Option C: Car IDM minimum gap behind stopped bus — don't allow gap < 1.5 m when bus is in merge zone. **Preferred because it's physically realistic.**

---

## Short-Term (after Run #4)

### 4. Bus Angle in GIF Renderer
- `renderer.py` draws the bus using `get_turn_angle_degrees()` which relies on `lateral_velocity` (which is wrong due to issue #1 above).
- Fix #1 first, then verify GIF shows bus rotating correctly during taper phases.

### 5. 2D View: `@st.fragment` Isolation
- Once Streamlit ≥ 1.33 is confirmed, wrap the entire 2D merge zone section in `@st.fragment`.
- This makes only the 2D view re-render on button/slider interaction; the rest of the page stays static.
- Without this, every button press triggers a full page re-run (Streamlit's fundamental model).

### 6. `lateral_gap` Between Car and Bus (not just car-leader)
- Currently `lateral_gap` in frame_history is **car-to-its-leader** (another car).
- Add a separate `bus_lateral_gap` = side clearance between the car and the bus, computed only when the bus is in the merge zone alongside the car.
- This is the gap most relevant to safety — it determines whether the bus actually fits.

### 7. Validation Run (Clean Test)
- After Run #4, evaluate on the 10 validation seeds (not used during development).
- These are: `[seeds to be kept secret here — do not expose in training code]`
- Compare: 378/400 (94.5%) from the post-Phase-3 baseline.

---

## Medium-Term

### 8. Curriculum Learning for Bus Angle
- Currently all merges happen at the same stop position (x=300).
- Vary stop positions (x ∈ [250, 350]) across training episodes so the agent generalizes to different traffic densities at the merge point.

### 9. Multi-Lane Traffic (optional)
- Current model has 1 traffic lane. In real urban scenarios the bus shares a 2-lane road.
- Would require significant changes to the state vector and lateral clearance computation.

### 10. Real-Time Inference Demo
- Replace Streamlit slider with a live animation (Streamlit's `st.empty()` + loop).
- Or export as a standalone matplotlib animation.

### 11. Formal Safety Metric
- Define a "safety score" per episode: minimum lateral clearance × minimum longitudinal gap at any frame during merge.
- Log this alongside reward so training can directly optimize for safety margin, not just merge success.

---

## Research / Long-Term

### 12. RLHF-style Human Preference Feedback
- Show two merge trajectories to a human; record which one felt safer/smoother.
- Fine-tune reward from these preferences.

### 13. Publish
- Possible venue: ITSC (Intelligent Transportation Systems Conference) or IV (Intelligent Vehicles Symposium).
- Would need: formal problem formulation, hyperparameter table, ablation over Three Pillars.

---

## Deferred / Deprioritized

| Item | Reason deferred |
|------|-----------------|
| 7th RL state feature | `nearest_accel` already in state vector (D2 from Gemini review — already done) |
| Scenario A (stopped-in-lane) | Rarely triggered; deprioritized until core merge works reliably |
| Noise σ annealing | Implementation noted in ROBUSTNESS_PLAN.md; defer until after Run #4 baseline |

