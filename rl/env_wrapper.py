import numpy as np
import random
import sys
import os

# Ensure we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vehicle import Vehicle, Bus, BusState, CAR_PARAMS, sample_vehicle_params


# ---------------------------------------------------------------------------
# Kinematic constants — must match vehicle.py Bus.__init__
# ---------------------------------------------------------------------------
BUS_WIDTH   = 2.5
BUS_LENGTH  = 11.0
LANE_WIDTH  = 3.5
BAY_DEPTH   = -3.5          # lateral_offset when fully in bay
STRAIGHTEN_ZONE = 5.0       # metres from stop: autopilot begins counter-steer


class BusMergeEnv:
    """
    Bus-merge RL environment — Phase C steering-first rewrite.

    Action space  : 2D continuous in [-1, 1]^2
                    action[0] = steer_intent  (+1 = full right / into lane)
                    action[1] = gas_intent    (+1 = full throttle, -1 = brake)

    State vector  : 11 features (see _encode_state docstring)

    Physics       : The bicycle model in Vehicle.update_physics() is the ONLY
                    thing that moves the bus.  No direct writes to
                    lateral_velocity, position, or velocity.

    Merge progress: Derived from actual lateral_offset, not a stored scalar.
                    merge_progress = clip((offset - BAY_DEPTH) / (-BAY_DEPTH), 0, 1)
                    0 = bus fully in bay, 1 = bus fully back in traffic lane.
    """

    # Physical limits for the autopilot controller inputs
    BUS_MAX_SPEED   = 40.0 / 3.6   # m/s  (~11.1 m/s)
    MERGE_ACCEL_MAX = 2.0           # m/s² maximum longitudinal pull during merge

    def __init__(self, eval_mode=False):
        self.dt        = 0.1
        self.max_steps = 500
        self.eval_mode = eval_mode

        # Network dimensions
        self.state_size  = 11
        self.action_size = 2   # (steer_intent, gas_intent)

        self.bus      = None
        self.vehicles = []
        self.current_step = 0
        self.last_intervened = False
        self.last_shield_condition = ""
        self.episode_shields = 0
        self.last_success = False
        self.last_crash   = False
        self.last_timeout = False

        # Domain-randomization controls set externally by train_ppo
        self.noise_sigma          = 0.0
        self.shield_penalty_scale = 1.0
        self._prev_bus_vel        = 0.0

    # ------------------------------------------------------------------
    # Helper: lateral progress 0 (in bay) → 1 (in lane)
    # ------------------------------------------------------------------
    def _merge_progress(self):
        """Merge progress derived from actual lateral_offset of the bus."""
        return float(np.clip(
            (self.bus.lateral_offset - BAY_DEPTH) / (-BAY_DEPTH), 0.0, 1.0))

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------
    def _encode_state(self, merge_progress, bus_heading_rad, lon_dist, lat_clearance,
                      car_speed, car_accel, car_width, car_length,
                      rel_speed, rel_accel, car_heading_rad):
        """
        Encode 11 features into a normalised state vector for the PPO network.

        Feature  0 : merge_progress     — 0=in bay, 1=in lane   [0, 1]
        Feature  1 : bus_heading_rad    — kinematic bicycle model heading [-0.54, 0.54] rad (31°)
        Feature  2 : lon_dist           — longitudinal gap bus-rear to car-front  [-10, 60] m
        Feature  3 : lat_clearance      — side gap between bus outer edge and car  [-2, 4] m
        Feature  4 : car_speed          — nearest car longitudinal speed [0, 20] m/s
        Feature  5 : car_accel          — nearest car acceleration [-9, 3] m/s²
        Feature  6 : car_width          — nearest car width [1.70, 2.05] m
        Feature  7 : car_length         — nearest car length [3.5, 5.5] m
        Feature  8 : rel_speed          — bus_vel − car_vel [-15, 15] m/s
        Feature  9 : rel_accel          — bus_accel − car_accel [-15, 15] m/s²
        Feature 10 : car_heading_rad    — nearest car bicycle-model heading [-0.15, 0.15] rad
        """
        if not self.eval_mode and self.noise_sigma > 0.0:
            k = self.noise_sigma
            lon_dist      += float(np.random.normal(0.0, k * max(abs(lon_dist), 1.0) * 0.05))
            lat_clearance += float(np.random.normal(0.0, k * (abs(lat_clearance) + 0.5) * 0.03))
            car_speed     += float(np.random.normal(0.0, k * 0.05))
            rel_speed     += float(np.random.normal(0.0, k * 0.05))
            rel_accel     += float(np.random.normal(0.0, k * 0.02))
            bus_heading_rad += float(np.random.normal(0.0, k * np.radians(0.3)))

        merge_progress   = float(np.clip(merge_progress,     0.0,   1.0))
        bus_heading_rad  = float(np.clip(bus_heading_rad,   -0.54,  0.54))
        lon_dist         = float(np.clip(lon_dist,          -10.0,  60.0))
        lat_clearance    = float(np.clip(lat_clearance,      -2.0,   4.0))
        car_speed        = float(np.clip(car_speed,           0.0,  20.0))
        car_accel        = float(np.clip(car_accel,          -9.0,   3.0))
        car_width        = float(np.clip(car_width,           1.70,  2.05))
        car_length       = float(np.clip(car_length,          3.5,   5.5))
        rel_speed        = float(np.clip(rel_speed,          -15.0,  15.0))
        rel_accel        = float(np.clip(rel_accel,          -15.0,  15.0))
        car_heading_rad  = float(np.clip(car_heading_rad,   -0.15,  0.15))

        p_n   = merge_progress
        ang_n = bus_heading_rad / 0.54
        lon_n = (lon_dist + 10.0) / 70.0
        lat_n = (lat_clearance + 2.0) / 6.0
        spd_n = car_speed / 20.0
        acc_n = car_accel / 9.0
        wid_n = (car_width  - 1.85) / 0.20
        len_n = (car_length - 4.5)  / 0.75
        rs_n  = rel_speed  / 15.0
        ra_n  = rel_accel  / 15.0
        ca_n  = car_heading_rad / 0.15

        return np.array([p_n, ang_n, lon_n, lat_n, spd_n, acc_n,
                         wid_n, len_n, rs_n, ra_n, ca_n], dtype=np.float32)

    def _scan_nearest_car(self):
        """Return nearest relevant car context in physical units."""
        gap = 999.0
        nearest_speed = 0.0
        nearest_accel = 0.0
        nearest_width = 1.85
        nearest_car   = None

        for v in self.vehicles:
            current_gap = self.bus.position - self.bus.projected_length_along_x() - v.position
            if -10 < current_gap < 60:
                if abs(current_gap) < abs(gap):
                    gap           = current_gap
                    nearest_speed = v.velocity
                    nearest_accel = v.acceleration
                    nearest_width = v.params.width
                    nearest_car   = v

        relative_speed = self.bus.velocity - nearest_speed
        return gap, nearest_speed, nearest_accel, nearest_width, relative_speed, nearest_car

    def reset(self):
        self.current_step  = 0
        self._prev_bus_vel = 0.0
        self.episode_shields = 0
        self.last_shield_condition = ""

        # 1. Spawn Bus fully in the bay, waiting to merge.
        #    lateral_offset = BAY_DEPTH (-3.5 m) = physically inside the bay.
        #    heading_angle  = 0 (bus parked parallel to curb).
        #    lat_target     = 0 (destination: back in the traffic lane).
        self.bus = Bus(v_id=999, position=300.0, velocity=0.0, params=CAR_PARAMS)
        self.bus.state = BusState.WAITING_TO_MERGE
        self.bus.merge_start_x = self.bus.position
        # --- PHYSICS INITIALISATION (bicycle model state) ---
        self.bus.lateral_offset  = BAY_DEPTH   # bus is in bay
        self.bus.lat_target      = 0.0         # destination: traffic lane centre
        # heading_angle / steer_angle / lateral_velocity are left at their
        # Vehicle.__init__ defaults (0.0) for the RL spawn; in the full simulation
        # they are inherited from the parking manoeuvre.
        # Keep _merge_progress for reward shaping / backward compat with production logs
        self.bus._merge_progress = 0.0

        # Bus power variance (training only): ±20% on max merge speed authority.
        if not self.eval_mode:
            power_factor = random.uniform(0.80, 1.20)
            self.bus.max_merge_progress_step = 0.02 * power_factor

        # 2. Traffic scenario
        self.vehicles = []
        scenario_roll = 1.0 if self.eval_mode else random.random()

        if scenario_roll < 0.30:
            # --- SCENARIO A: JAM ---
            num_cars = random.randint(3, 5)
            for i in range(num_cars):
                dist = random.uniform(12.0, 22.0) + (i * 8.0)
                _, sampled_params = sample_vehicle_params()
                self._maybe_distract(sampled_params)
                speed = random.uniform(0.0, 3.0)
                car = Vehicle(v_id=i, position=300.0 - dist, velocity=speed, params=sampled_params)
                self.vehicles.append(car)

        elif scenario_roll < 0.60:
            # --- SCENARIO B: DENSE-MOVING ---
            num_cars = random.randint(3, 5)
            for i in range(num_cars):
                dist = random.uniform(10.0, 25.0) + (i * 10.0)
                _, sampled_params = sample_vehicle_params()
                self._maybe_distract(sampled_params)
                speed = max(2.0, sampled_params.max_speed * random.uniform(0.25, 0.65))
                car = Vehicle(v_id=i, position=300.0 - dist, velocity=speed, params=sampled_params)
                self.vehicles.append(car)

        elif scenario_roll < 0.90:
            # --- SCENARIO C: MID-MERGE RESILIENCE ---
            # Bus already partially committed; test agent under car-catching-up pressure.
            initial_lateral = BAY_DEPTH * random.uniform(0.35, 0.85)  # partial merge
            self.bus.lateral_offset  = initial_lateral
            self.bus._merge_progress = self._merge_progress()
            num_cars = random.randint(1, 4)
            for i in range(num_cars):
                dist = random.uniform(8.0, 35.0) + (i * 12.0)
                _, sampled_params = sample_vehicle_params()
                self._maybe_distract(sampled_params)
                speed = max(2.0, sampled_params.max_speed * random.uniform(0.30, 0.90))
                car = Vehicle(v_id=i, position=300.0 - dist, velocity=speed, params=sampled_params)
                self.vehicles.append(car)

        else:
            # --- SCENARIO D: OPEN ROAD ---
            num_cars = random.randint(1, 3)
            for i in range(num_cars):
                dist = random.uniform(10.0, 50.0) + (i * 25.0)
                _, sampled_params = sample_vehicle_params()
                self._maybe_distract(sampled_params)
                speed = max(4.0, sampled_params.max_speed * random.uniform(0.55, 0.95))
                car = Vehicle(v_id=i, position=300.0 - dist, velocity=speed, params=sampled_params)
                self.vehicles.append(car)

        return self._get_state()

    def step(self, action_value):
        """
        action_value : np.ndarray shape (2,) or (1,2)  — [steer_intent, gas_intent] in [-1,1]
        """
        self.current_step += 1
        self.last_intervened = False
        self.last_shield_condition = ""
        self.last_success = False
        self.last_crash   = False
        self.last_timeout = False

        # ── 1. PARSE 2D ACTION ────────────────────────────────────────────────────
        # Flatten to 1D regardless of input shape
        action_arr = np.asarray(action_value, dtype=np.float32).flatten()
        steer_intent = float(np.clip(action_arr[0], -1.0, 1.0))  # +1 = steer into lane
        gas_intent   = float(np.clip(action_arr[1], -1.0, 1.0))  # +1 = accelerate

        # Map gas_intent → target speed for the autopilot controller
        # [-1, 0] = brake toward 0 m/s; [0, +1] = accelerate toward BUS_MAX_SPEED
        if gas_intent >= 0.0:
            speed_target = gas_intent * self.BUS_MAX_SPEED
        else:
            speed_target = 0.0   # negative intent = brake to zero

        # Map steer_intent → lateral target offset
        # steer_intent=+1 → lat_target=0 (traffic lane); steer_intent=-1 → lat_target=BAY_DEPTH
        lat_target = steer_intent * 0.0 + (1.0 - abs(steer_intent)) * self.bus.lateral_offset
        # Simpler linear mapping: +1 pushes fully into lane, -1 stays in bay
        lat_target = np.interp(steer_intent, [-1.0, 1.0], [BAY_DEPTH, 0.0])

        self._prev_bus_vel = self.bus.velocity

        # ── 2. SAFETY SHIELD ─────────────────────────────────────────────────────
        nearest_gap, nearest_speed, _, nearest_width, rel_speed, _ = self._scan_nearest_car()
        intervened = False
        raw_steer  = steer_intent
        raw_gas    = gas_intent

        adaptive_min_gap = 4.0 + max(0.0, nearest_width - 1.80) * 2.2
        progress = self._merge_progress()

        if steer_intent > 0 and nearest_gap > -5.0:  # only shield when moving INTO lane
            if 0.0 <= nearest_gap < 0.50:
                if progress >= 0.30:
                    # Committed — push forward to clear
                    steer_intent = 1.0
                    gas_intent   = max(gas_intent, 0.5)
                    intervened   = True
                    self.last_shield_condition = "emergency_forward"
                else:
                    if nearest_speed >= 2.0:
                        steer_intent = 0.0   # car still moving — wait
                        gas_intent   = 0.0
                    else:
                        steer_intent = 0.1   # deadlock escape: tiny crawl
                        gas_intent   = 0.1
                    intervened = True
                    self.last_shield_condition = "emergency_stop"
            elif progress > 0.65 and 0.0 <= nearest_gap < 1.20 and rel_speed > 0.2:
                steer_intent = 0.0
                gas_intent   = 0.0
                intervened   = True
                self.last_shield_condition = "late_overlap"
            elif progress > 0.75 and 0.0 <= nearest_gap < 2.5 and rel_speed > 1.5:
                steer_intent = 0.0
                gas_intent   = 0.0
                intervened   = True
                self.last_shield_condition = "late_overlap"
            elif 0.0 <= nearest_gap < adaptive_min_gap and rel_speed > 1.2:
                gap_opening_safely = (
                    (nearest_gap > 1.4 and progress > 0.40 and rel_speed < 3.5)
                    or (nearest_gap > 2.5 and progress < 0.30 and rel_speed < 2.5)
                )
                if not gap_opening_safely:
                    risk = np.clip((adaptive_min_gap - nearest_gap) / max(adaptive_min_gap, 1e-3), 0.0, 1.0)
                    steer_intent *= (1.0 - 0.85 * risk)
                    intervened    = True
                    self.last_shield_condition = "adaptive_gap"
            elif steer_intent > 0.65 and 0.0 <= nearest_gap < (adaptive_min_gap + 6.0) and rel_speed > 2.5:
                steer_intent *= 0.35
                intervened    = True
                self.last_shield_condition = "high_pressure"

        # Already overlapping at late stage — keep pushing to clear
        if nearest_gap < 0.0 and progress > 0.60:
            steer_intent = max(steer_intent, 0.5)
            gas_intent   = max(gas_intent,   0.5)

        self.last_intervened = intervened
        if intervened:
            self.episode_shields += 1
        intervention_strength_steer = abs(raw_steer - steer_intent) / max(abs(raw_steer), 1e-8)
        intervention_strength = float(np.clip(intervention_strength_steer, 0.0, 1.0))

        # ── 3. TRANSLATE SHIELD OUTPUT BACK TO AUTOPILOT TARGETS ─────────────────
        lat_target   = np.interp(steer_intent, [-1.0, 1.0], [BAY_DEPTH, 0.0])
        if gas_intent >= 0.0:
            speed_target = gas_intent * self.BUS_MAX_SPEED
        else:
            speed_target = 0.0

        # ── 4. APPLY TARGETS TO BUS (autopilot writes lat_target; bicycle does the rest) ──
        self.bus.lat_target = lat_target

        # Longitudinal autopilot: simple proportional controller on speed
        # accel = k*(speed_target - current_velocity), clamped to comfort bounds
        speed_error = speed_target - self.bus.velocity
        desired_accel = np.clip(2.0 * speed_error, -3.0, self.MERGE_ACCEL_MAX)
        # Override IDM during merge: set acceleration directly (bus is out of lane)
        self.bus.acceleration = desired_accel

        # ── 5. PHYSICS TICK ──────────────────────────────────────────────────────
        mock_leader = {
            'position': self.bus.position,
            'velocity': self.bus.velocity,
            'length':   self.bus.projected_length_along_x(),
            'presence': self.bus.presence_factor,
        }

        # Update bus physics (bicycle model — uses lat_target set above)
        self.bus.velocity = max(0.0, self.bus.velocity + self.bus.acceleration * self.dt)
        self.bus.position += self.bus.velocity * self.dt
        # Bicycle model for lateral motion
        max_steer  = np.radians(self.bus.params.max_steer_angle_deg)
        steer_rate = np.radians(self.bus.params.steer_rate_deg_per_s)
        lookahead   = max(self.bus.velocity * 1.0, 5.0)
        lat_error   = self.bus.lat_target - self.bus.lateral_offset
        delta_target = np.clip(np.arctan2(lat_error, lookahead), -max_steer, max_steer)
        steer_change = np.clip(delta_target - self.bus.steer_angle, -steer_rate * self.dt, steer_rate * self.dt)
        self.bus.steer_angle   += steer_change
        wheelbase = self.bus.params.wheelbase
        self.bus.heading_angle += (self.bus.velocity * np.tan(self.bus.steer_angle) / max(wheelbase, 0.1)) * self.dt
        self.bus.heading_angle  = np.clip(self.bus.heading_angle, -np.radians(60.0), np.radians(60.0))
        self.bus.lateral_velocity = self.bus.velocity * np.sin(self.bus.heading_angle)
        self.bus.lateral_offset  += self.bus.lateral_velocity * self.dt
        self.bus.lateral_offset   = np.clip(self.bus.lateral_offset, -4.5, 0.5)

        # O-U lane-keeping noise for bus
        _ou_rev = 0.4
        _ou_sig = getattr(self.bus.params, 'lane_keeping_noise', 0.05)
        self.bus.lane_drift = ((1.0 - _ou_rev * self.dt) * self.bus.lane_drift
                               + _ou_sig * np.sqrt(self.dt) * np.random.randn())
        self.bus.lane_drift = np.clip(self.bus.lane_drift, -0.20, 0.20)

        # Keep _merge_progress in sync for backward compat (production state machine, logs)
        self.bus._merge_progress = self._merge_progress()

        # Update cars (IDM sees the bus through presence_factor)
        for car in self.vehicles:
            car.update_physics(self.dt, leader=mock_leader, bus=self.bus)

        # ── 6. COMPUTE REWARDS ───────────────────────────────────────────────────
        reward = -1.0   # time penalty
        done   = False

        gap, car_speed, car_accel, car_width, rel_speed, _ = self._scan_nearest_car()
        progress = self._merge_progress()
        state = self._get_state()

        # Shield penalty
        if intervened:
            reward -= (4.0 + 20.0 * (raw_steer ** 2)) * intervention_strength * self.shield_penalty_scale
            if self.last_shield_condition == "adaptive_gap":
                reward -= 10.0 * (raw_steer ** 2) * self.shield_penalty_scale

        # Conditional push reward: crossing into the lane earns reward only when safe
        is_safe_to_merge = (gap > 5.0) or (gap > 0.5 and rel_speed > -0.5)
        lateral_delta = self.bus.lateral_velocity * self.dt   # positive = into lane
        if is_safe_to_merge and lateral_delta > 0:
            reward += lateral_delta / (-BAY_DEPTH) * 55.0

        # Progress bonus (unconditional)
        reward += progress * 8.0

        # Near-completion push
        if progress > 0.9:
            reward += 20.0

        # Completion urgency
        if progress > 0.5 and gap > 5.0:
            reward += (lateral_delta / (-BAY_DEPTH)) * 40.0

        # Braking penalty
        if progress > 0.1 and car_accel < -2.5 and gap > -10.0:
            reward -= abs(car_accel) * 3.0

        # Continuous near-risk shaping
        if 0.0 < gap < 12.0:
            gap_risk     = (12.0 - gap) / 12.0
            speed_risk   = np.clip((-rel_speed) / 8.0, 0.0, 1.0)
            merge_risk   = np.clip((progress - 0.2) / 0.8, 0.0, 1.0)
            width_factor = 1.0 + (max(0.0, car_width - 1.8) * 0.7)
            reward -= 70.0 * gap_risk * (0.6 + 0.4 * speed_risk) * merge_risk * width_factor

        # TTC shaping
        if gap > 0.0 and rel_speed > 0.1:
            ttc = gap / rel_speed
            if ttc < 4.0:
                ttc_risk = (4.0 - ttc) / 4.0
                reward -= 120.0 * ttc_risk * (0.4 + 0.6 * progress) * max(steer_intent, 0.0)

        # Terminal rewards
        if progress >= 1.0:
            reward += 2500.0
            if self.episode_shields == 0:
                reward += 200.0
            done = True
            self.last_success = True
        elif gap <= 0.0 and gap > -5.0 and progress >= 0.75:
            reward -= 4500.0
            done = True
            self.last_crash = True
        elif gap <= 0.0 and gap > -5.0 and progress >= 0.3:
            overlap = min(5.0, -gap) / 5.0
            reward -= 600.0 * ((progress - 0.3) / 0.5) * (0.5 + overlap)

        if self.current_step >= self.max_steps and not done:
            reward -= 2000.0
            done = True
            self.last_timeout = True

        return state, reward, done

    def _get_state(self):
        gap, nearest_speed, nearest_accel, nearest_width, relative_speed, nearest_car = self._scan_nearest_car()

        bus_heading_rad = self.bus.heading_angle  # direct from bicycle model
        progress        = self._merge_progress()

        # Angle-corrected lateral clearance at the bus outer edge
        bus_lat_edge   = self.bus.lateral_offset + BUS_WIDTH / 2.0
        car_lat_offset = getattr(nearest_car, 'total_lateral_offset', 0.0) if nearest_car else 0.0
        lat_clearance  = LANE_WIDTH - max(0.0, bus_lat_edge) - (nearest_width - car_lat_offset)

        # Relative acceleration
        bus_vel_now = self.bus.velocity
        bus_accel   = (bus_vel_now - self._prev_bus_vel) / self.dt
        rel_accel   = bus_accel - nearest_accel

        car_length    = nearest_car.params.length if nearest_car else 4.5
        car_heading_rad = getattr(nearest_car, 'heading_angle', 0.0) if nearest_car else 0.0

        return self._encode_state(
            progress,
            bus_heading_rad,
            gap,
            lat_clearance,
            nearest_speed,
            nearest_accel,
            nearest_width,
            car_length,
            relative_speed,
            rel_accel,
            car_heading_rad,
        )

    def _maybe_distract(self, params):
        """15% chance a spawned car is 'distracted' — slower reaction time."""
        if not self.eval_mode and random.random() < 0.15:
            factor = random.uniform(1.5, 3.0)
            params.base_reaction_time   *= factor
            params.shock_reaction_time   = max(params.shock_reaction_time,
                                               params.base_reaction_time + 0.5)
