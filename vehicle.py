import numpy as np
from dataclasses import dataclass
import random
from typing import Dict

@dataclass
class VehicleParams:
    # URBAN SETTINGS
    max_speed: float = 13.9      
    max_accel: float = 2.0       
    comfortable_brake: float = 2.0 
    min_gap: float = 2.0         
    time_headway: float = 1.5    
    length: float = 4.5          # REAL CAR LENGTH
    width: float = 1.9           # NEW: REAL CAR WIDTH
    delta: float = 4.0           

    # Human Factors
    base_reaction_time: float = 0.5 
    shock_reaction_time: float = 1.0 
    vehicle_class: str = "Mid"
    politeness_factor: float = 0.7   # 0=aggressive/no yield, 1=fully polite/max nudge
    wheelbase: float = 2.7           # metres between axles (bicycle model turning geometry)
    max_steer_angle_deg: float = 35.0    # peak rack angle (degrees) — physical hardware limit
    steer_rate_deg_per_s: float = 75.0   # how fast the steering wheel can rotate (deg/s)
    lane_keeping_noise: float = 0.08     # O-U σ for imperfect lane tracking (m)

CAR_PARAMS = VehicleParams()    

# Real-world anchor points used for class ranges (Wikipedia model pages):
# - Compact reference: Toyota Corolla (E210) ~4.63 m x 1.78 m
# - Mid-size reference: Toyota Camry (XV70) ~4.88-4.91 m x 1.84 m
# - Large reference: Chevrolet Impala (Gen 10) ~5.11 m x 1.85 m
# - SUV reference: Honda CR-V (5th gen) / Toyota RAV4 (XA50) ~4.58-4.63 m x 1.855 m
CAR_CLASS_PROFILES: Dict[str, Dict[str, object]] = {
    "Compact": {
        "weight": 0.30,
        "length_range": (4.20, 4.55),
        "width_range": (1.74, 1.80),
        "max_speed": 14.5,
        "max_accel": 2.3,
        "comfortable_brake": 2.2,
        "time_headway": 1.35,
        "base_reaction_time": 0.45,
        "wheelbase": 2.62,          # Toyota Corolla E210: 2.64 m
        "max_steer_angle_deg": 38.0,    # short wheelbase → nimble steering
        "steer_rate_deg_per_s": 90.0,   # light car → fast wheel rotation
        "lane_keeping_noise": 0.06,     # typical suburban commuter
    },
    "Mid": {
        "weight": 0.30,
        "length_range": (4.60, 4.90),
        "width_range": (1.80, 1.88),
        "max_speed": 13.9,
        "max_accel": 2.0,
        "comfortable_brake": 2.0,
        "time_headway": 1.50,
        "base_reaction_time": 0.50,
        "wheelbase": 2.75,          # Toyota Camry XV70: 2.825 m
        "max_steer_angle_deg": 35.0,
        "steer_rate_deg_per_s": 75.0,
        "lane_keeping_noise": 0.08,
    },
    "Large": {
        "weight": 0.15,
        "length_range": (4.90, 5.20),
        "width_range": (1.86, 1.95),
        "max_speed": 13.0,
        "max_accel": 1.7,
        "comfortable_brake": 1.8,
        "time_headway": 1.70,
        "base_reaction_time": 0.58,
        "wheelbase": 2.92,          # Chevrolet Impala Gen10: 2.84 m
        "max_steer_angle_deg": 30.0,    # large car, limited turn-in
        "steer_rate_deg_per_s": 60.0,   # heavier steering system
        "lane_keeping_noise": 0.10,     # lazy lane-keeping in large sedans
    },
    "SUV": {
        "weight": 0.15,
        "length_range": (4.55, 4.75),
        "width_range": (1.84, 1.93),
        "max_speed": 13.5,
        "max_accel": 1.9,
        "comfortable_brake": 1.9,
        "time_headway": 1.60,
        "base_reaction_time": 0.55,
        "wheelbase": 2.68,          # Honda CR-V 5th gen: 2.662 m
        "max_steer_angle_deg": 33.0,
        "steer_rate_deg_per_s": 70.0,
        "lane_keeping_noise": 0.09,
    },
    # Phase 2 (C1): Aggressive driver — tailgater, fast reactor, tight headway.
    # Weight 0.10 → 1 in 10 spawned cars will behave aggressively.
    # This is the hardest adversarial case: if the agent handles this, normal cars are easy.
    "Aggressive": {
        "weight": 0.10,
        "length_range": (4.20, 4.55),   # Compact-sized
        "width_range":  (1.74, 1.80),
        "max_speed": 15.5,              # 11% above normal
        "max_accel": 2.8,
        "comfortable_brake": 2.5,
        "time_headway": 1.0,            # Tailgating (normal = 1.5s)
        "base_reaction_time": 0.35,     # Alert and fast
        "politeness_factor": 0.0,       # refuses to yield; only brakes at last moment
        "wheelbase": 2.62,              # Compact-based
        "max_steer_angle_deg": 40.0,    # sharp, reckless inputs
        "steer_rate_deg_per_s": 120.0,  # very fast wheel feedback
        "lane_keeping_noise": 0.15,     # erratic lane position
    },
}


def sample_vehicle_params(vehicle_class: str = None):
    """Sample one car profile so physics is not tied to a single car size."""
    if vehicle_class is None:
        classes = list(CAR_CLASS_PROFILES.keys())
        weights = [CAR_CLASS_PROFILES[c]["weight"] for c in classes]
        vehicle_class = random.choices(classes, weights=weights, k=1)[0]

    profile = CAR_CLASS_PROFILES[vehicle_class]
    length = random.uniform(*profile["length_range"])
    width = random.uniform(*profile["width_range"])

    # ── DRIVER PERSONALITY: all behaviour params sampled per individual ──────
    # Class profiles define the *centre* of each distribution.
    # The driver drawn from the same class can still be calm or excited that day.
    # This prevents the bus from learning fixed archetypes; it must drive safely
    # for ANY combination of steering speed, aggression and headway.

    # Politeness: Beta(2,2) → bell-shaped 0–1, centred ~0.5.
    # Aggressive archetype is drawn from Beta(1,6) → heavily skewed toward 0
    # (usually 0.0–0.25) but not literally clamped to 0.
    if vehicle_class == "Aggressive":
        politeness = float(np.random.beta(1.0, 6.0))       # almost always near 0, rarely ~0.3
    else:
        politeness = float(np.random.beta(2.0, 2.0))

    # Steering rate: LogNormal so it's always positive.
    # μ_log, σ_log chosen so the median equals the profile value and ±1σ spans
    # roughly ±25% of that value.
    _sr_mean = profile.get("steer_rate_deg_per_s", 75.0)
    _sr_sigma_log = 0.20                                    # CV ≈ 20 %
    steer_rate = float(np.clip(
        np.random.lognormal(np.log(_sr_mean), _sr_sigma_log),
        _sr_mean * 0.40, _sr_mean * 2.0))

    # Max steering angle: Normal, σ = 3°.
    _ma_mean = profile.get("max_steer_angle_deg", 35.0)
    max_steer = float(np.clip(
        np.random.normal(_ma_mean, 3.0),
        max(15.0, _ma_mean - 8.0), _ma_mean + 8.0))

    # Lane-keeping noise: LogNormal, σ_log = 0.20 (same relative spread).
    _lk_mean = profile.get("lane_keeping_noise", 0.08)
    lane_noise = float(np.clip(
        np.random.lognormal(np.log(_lk_mean), 0.25),
        _lk_mean * 0.30, _lk_mean * 3.0))

    params = VehicleParams(
        max_speed=profile["max_speed"],
        max_accel=profile["max_accel"],
        comfortable_brake=profile["comfortable_brake"],
        min_gap=2.0,
        time_headway=profile["time_headway"],
        length=length,
        width=width,
        delta=4.0,
        base_reaction_time=profile["base_reaction_time"],
        shock_reaction_time=max(0.9, profile["base_reaction_time"] + 0.5),
        vehicle_class=vehicle_class,
        politeness_factor=politeness,
        wheelbase=profile["wheelbase"],
        max_steer_angle_deg=max_steer,
        steer_rate_deg_per_s=steer_rate,
        lane_keeping_noise=lane_noise,
    )
    return vehicle_class, params

class Vehicle:
    def __init__(self, v_id: int, position: float, velocity: float, params: VehicleParams = None):
        self.id = v_id
        self.position = position
        self.velocity = velocity
        self.acceleration = 0.0
        self.params = params if params else CAR_PARAMS
        self.vehicle_class = self.params.vehicle_class
        
        self.reaction_timer = 0.0
        self.last_presence_seen = 0.0
        self.is_squeezing = False
        # Lateral motion — Steering-First bicycle model (Driver Mind + Vehicle Body)
        self.lateral_offset = 0.0      # metres intentional nudge from lane centre
        self.lateral_velocity = 0.0    # m/s lateral speed (result of heading, not cause)
        self.steer_angle = 0.0         # current steering wheel angle (rad)
        self.lat_target = 0.0          # driver's desired lateral offset (m)
        self.heading_angle = 0.0       # vehicle body yaw (rad) — drives lateral_velocity
        self.lane_drift = 0.0          # O-U lane-keeping uncertainty (m)

    def update_physics(self, dt: float, leader=None, bus=None):
        # --- 1. SQUEEZE EXIT LOGIC ---
        if self.is_squeezing:
            if bus:
                safe_pass_point = bus.position + bus.params.length + 5.0
                if self.position > safe_pass_point:
                    self.is_squeezing = False # Safe exit
            elif leader is None:
                self.is_squeezing = False

        # --- 2. REACTION TIMER ---
        self.reaction_timer -= dt
        if leader:
            current_presence = leader.get('presence', 1.0)
            if current_presence < 1.0 and current_presence > self.last_presence_seen:
                diff = current_presence - self.last_presence_seen
                if diff > 0.1 and self.reaction_timer <= 0 and not self.is_squeezing:
                    self.reaction_timer += self.params.shock_reaction_time
            self.last_presence_seen = current_presence

        # --- 3. PHYSICS STEP ---
        if self.reaction_timer <= 0:
            self.acceleration = self._calculate_idm_accel(leader, bus)
            self.reaction_timer = self.params.base_reaction_time + random.uniform(-0.1, 0.1)

        # The bus's own braking profile must be smooth (no reaction-time staircase).
        # Re-compute IDM every frame when the bus is decelerating toward its stop;
        # this removes the velocity step-chunks visible in the entry telemetry.
        # Cap at -1.5 m/s² (comfortable_brake) so the first IDM frame at 50 m out
        # never dumps more than -1.5 m/s² — the sudden-speed-drop bug.
        if isinstance(self, Bus) and leader is not None:
            # PERCEPTION NOISE: the bus perceives the gap to its leader with a small
            # Gaussian error (σ=2 %, min 0.3 m).  Mimics imperfect radar/lidar ranging
            # and causes slightly different braking curves across episodes.
            noisy_leader = dict(leader)
            raw_gap = max(0.1, leader['position'] - leader['length'] - self.position)
            gap_noise = np.random.normal(0.0, max(0.3, raw_gap * 0.02))
            # Shift the perceived leader position by the gap error (positive = appears farther)
            noisy_leader['position'] = leader['position'] + gap_noise
            self.acceleration = max(self._calculate_idm_accel(noisy_leader, bus), -1.5)
            # MICRO-CORRECTION NOISE: foot-on/off-pedal variability.
            # σ scales with braking demand so the bus is calm when coasting
            # (σ≈0.02 m/s²) and shows more variability when braking hard
            # toward the stop or a decelerating car (σ≈0.08 m/s²).
            _idm_demand = float(np.clip(abs(self.acceleration) / 1.5, 0.0, 1.0))
            _sigma_micro = 0.02 + 0.06 * _idm_demand
            self.acceleration += float(np.random.normal(0.0, _sigma_micro))
            self.acceleration = max(self.acceleration, -1.5)

        self.velocity += self.acceleration * dt
        if self.velocity < 0:
            self.velocity = 0.0
            self.acceleration = 0.0
        self.position += self.velocity * dt

        # ── STEERING-FIRST LATERAL MODEL ─────────────────────────────────────────
        # Decoupled: Driver Mind (intent) → Steering Wheel (hardware) → Bicycle Physics → Position
        is_bus = isinstance(self, Bus)

        # 1. DRIVER INTENT: cars compute from squeeze logic; bus lat_target is set
        #    externally by _handle_bus_logic (the "mind") before update_physics runs.
        if not is_bus:
            MAX_NUDGE = 0.5   # metres max intentional lateral nudge
            pol = getattr(self.params, 'politeness_factor', 0.7)
            self.lat_target = (MAX_NUDGE * pol) if (self.is_squeezing and bus is not None) else 0.0

        # Geometry constants used by both the steer controller and the bicycle model.
        wheelbase  = getattr(self.params, 'wheelbase', 2.7)
        max_steer  = np.radians(getattr(self.params, 'max_steer_angle_deg', 35.0))
        steer_rate = np.radians(getattr(self.params, 'steer_rate_deg_per_s', 75.0))

        # BUS ACTIVE LANE-CORRECTION ─────────────────────────────────────────────
        # A small OU process drives _lat_perturb, which is added to lat_target before
        # the pure-pursuit step.  The controller then steers to correct it, producing
        # natural oscillations around the intended path instead of invisible drift.
        #   • Only active during MOVING (straight driving) — suppressed during bay
        #     entry and merge so precise path-following is unaffected.
        #   • σ=0.03 m → steady-state std ≈ 3 cm, bounded ±10 cm.
        if is_bus:
            _OU_LAT_REVERT = 0.5   # s⁻¹ — faster reversion = quicker correction
            _OU_LAT_SIG    = 0.03  # m   — ~3 cm std at steady state
            if self.state == BusState.MOVING:
                self._lat_perturb = ((1.0 - _OU_LAT_REVERT * dt) * self._lat_perturb
                                     + _OU_LAT_SIG * np.sqrt(dt) * np.random.randn())
                self._lat_perturb = float(np.clip(self._lat_perturb, -0.10, 0.10))
            else:
                # Decay to zero when not in straight-line driving so the bay
                # entry and merge are not perturbed by residual lateral noise.
                self._lat_perturb *= max(0.0, 1.0 - 5.0 * dt)
            effective_lat_target = self.lat_target + self._lat_perturb
        else:
            effective_lat_target = self.lat_target

        # 2. STEERING CONTROLLER
        if is_bus:
            # PURE PURSUIT: compute the steering command as the arc required to meet
            # the look-ahead point (bus.position + L_PP, lat_target) in the bus's own
            # body frame.  Because it subtracts the bus's current heading_angle, it
            # automatically generates counter-steer whenever the accumulated heading has
            # overshot the desired path angle — this is what keeps the bus parallel to
            # the kerb when parking without a separate straighten-zone.
            #
            # y_rel = (effective_lat_target - lateral_offset) * cos(ψ) − L_PP * sin(ψ)
            #   where ψ = heading_angle (negative = nose pointing into bay)
            # δ_cmd  = atan2(2 * L * y_rel,  L_PP²)
            #
            # When ψ < 0 and lateral error ≈ 0 (bus at bay depth):
            #   y_rel ≈ −L_PP * sin(ψ) > 0  →  δ_cmd > 0 (counter-steer back to road)
            L_PP    = 18.0   # look-ahead distance (m) — longer = smoother, less oscillation
            lat_err = effective_lat_target - self.lateral_offset
            y_rel   = lat_err * np.cos(self.heading_angle) - L_PP * np.sin(self.heading_angle)
            delta_target = np.clip(np.arctan2(2.0 * wheelbase * y_rel, L_PP * L_PP),
                                   -max_steer, max_steer)
        else:
            # Simple proportional look-ahead for cars.
            lookahead    = max(self.velocity * 1.0, 5.0)
            lat_error    = self.lat_target - self.lateral_offset
            delta_target = np.clip(np.arctan2(lat_error, lookahead), -max_steer, max_steer)

        # 3. MECHANICAL RESPONSE: steering wheel can only rotate at steer_rate (deg/s)
        steer_change = np.clip(delta_target - self.steer_angle,
                               -steer_rate * dt, steer_rate * dt)
        self.steer_angle += steer_change

        # 4. BICYCLE MODEL: heading rate = v × tan(δ) / L
        #    Longer wheelbase → lower yaw rate for the same steering input
        self.heading_angle += (self.velocity * np.tan(self.steer_angle)
                               / max(wheelbase, 0.1)) * dt
        # Heading can exceed steer angle through integration; clip to generous physical limit
        self.heading_angle = np.clip(self.heading_angle, -np.radians(60.0), np.radians(60.0))

        # 5. POSITION UPDATE: lateral_velocity is now a RESULT of heading, not a cause
        self.lateral_velocity = self.velocity * np.sin(self.heading_angle)
        self.lateral_offset  += self.lateral_velocity * dt
        if is_bus:
            # Bus range: 0 (lane centre) to −3.5 (bay depth).
            # Hard wall at −3.5 prevents overshoot; clip at +0.5 is a soft road shoulder.
            self.lateral_offset = np.clip(self.lateral_offset, -3.5, 0.5)
        else:
            self.lateral_offset = np.clip(self.lateral_offset, -0.05, 0.55)

        # ── LANE-KEEPING UNCERTAINTY (Ornstein-Uhlenbeck random drift) ─────────────
        # Models the fact that no driver holds exactly the lane centre.
        # Each class has its own noise level (aggressive wanders more, compact less).
        OU_REVERT = 0.4   # mean-reversion rate (s⁻¹): higher → snaps back faster
        noise_sig = getattr(self.params, 'lane_keeping_noise', 0.08)
        self.lane_drift = ((1.0 - OU_REVERT * dt) * self.lane_drift
                           + noise_sig * np.sqrt(dt) * np.random.randn())
        self.lane_drift = np.clip(self.lane_drift, -0.40, 0.40)

    @property
    def total_lateral_offset(self):
        """Effective lateral offset: intentional nudge + passive lane-keeping drift."""
        return self.lateral_offset + self.lane_drift

    def _calculate_idm_accel(self, leader, bus=None) -> float:
        p = self.params
        target_speed = p.max_speed
        squeeze_brake_penalty = 0.0

        if leader is not None:
            presence = leader.get('presence', 1.0)
            gap = leader['position'] - leader['length'] - self.position
            if gap < 0.1: gap = 0.1
            
            # Calculate Braking Need
            req_brake = (self.velocity ** 2) / (2 * gap)

            # --- PRESSURE REACTION LOGIC ---
            
            if presence == 0.0:
                self.is_squeezing = False
            elif presence == 1.0:
                self.is_squeezing = False
            
            else:
                # DYNAMIC SQUEEZE CALCULATION
                lane_width = 3.5
                safety_gap = 0.5
                bus_width  = 2.5   # BUS_WIDTH
                bus_length = 11.0  # BUS_LENGTH
                # Pillar C: Geometric Awareness — sense from the bus REAR corner, not the centre.
                # When the bus is angled mid-merge, the rear extends LESS into the lane than
                # the centre does.  Using the centre overstates the threat by up to ~0.9 m at
                # peak yaw (13.1°), causing premature panic braking.
                if bus is not None:
                    bus_angle_rad = np.radians(bus.get_turn_angle_degrees())
                    bus_rear_lat = max(0.0, presence * bus_width
                                      - (bus_length / 2.0) * np.sin(bus_angle_rad))
                else:
                    bus_rear_lat = presence * bus_width  # fallback: flat-centre estimate

                # How much lane is the bus eating up (at its rear corner)?
                bus_occupancy = bus_rear_lat
                # How much space is left for the car?
                space_left = lane_width - bus_occupancy
                # How much space does this specific car need?
                space_needed = self.params.width + safety_gap

                # --- NEW: THE SOFT WALL ---
                squeeze_urgency = space_needed - space_left

                # Start reacting slightly BEFORE the exact boundary to avoid a hard wall.
                # soft_zone defines the transition region around squeeze_urgency == 0.
                soft_zone = 0.60  # meters
                urgency_norm = np.clip((squeeze_urgency + soft_zone) / (2.0 * soft_zone), 0.0, 1.0)
                smooth = urgency_norm * urgency_norm * (3.0 - 2.0 * urgency_norm)  # smoothstep

                # Gradual penalty: mild braking as lane tightens, stronger when physically blocked.
                base_soft_brake = -3.5 * smooth
                overflow = max(0.0, squeeze_urgency) / soft_zone
                extra_hard_brake = -5.5 * overflow
                squeeze_brake_penalty = max(-9.0, base_soft_brake + extra_hard_brake)

                self.is_squeezing = squeeze_urgency > 0.05
                if not self.is_squeezing:
                    presence = 1.0
                
                # # If the space left is less than what the car needs, the lane is blocked!
                # if space_left < space_needed:
                #     if req_brake > 6.0:
                #         self.is_squeezing = True
                #         presence = 1.0 # The car is still physically there
                #         return -9.0    # Emergency brake to survive
                #     else:
                #         self.is_squeezing = False
                #         presence = 1.0
                
                # # BUS IS CREEPING (< 0.85)
                # else:
                #     panic_squeeze = (req_brake > 4.0)
                #     opportunistic_squeeze = (gap < 20.0 and presence < 0.5)
                #     keep_squeezing = self.is_squeezing

                #     if panic_squeeze or opportunistic_squeeze or keep_squeezing:
                #         self.is_squeezing = True 
                #         target_speed = p.max_speed * 1.3 
                #         presence = 0.0 
                #     else:
                #         self.is_squeezing = False
                #         presence = 1.0 

        # --- STANDARD IDM MATH ---
        if self.velocity < 0.1:
            speed_ratio = 0
        else:
            speed_ratio = (self.velocity / target_speed)
        
        free_road_term = 1.0 - (speed_ratio ** p.delta)

        if leader is None:
            interaction_term = 0.0
        else:
            delta_v = self.velocity - leader['velocity']
            desired_gap = (p.min_gap + 
                           (self.velocity * p.time_headway) + 
                           (self.velocity * delta_v) / (2 * np.sqrt(p.max_accel * p.comfortable_brake)))
            interaction_term = (desired_gap / gap) ** 2
            
            leader_presence = 1.0 if leader is None else presence
            interaction_term *= leader_presence

        accel = p.max_accel * (free_road_term - interaction_term)
        accel += squeeze_brake_penalty
        return max(accel, -9.0)

class Bus(Vehicle):
    def __init__(self, v_id, position, velocity, params):
        # Give the bus its own separate copy of the parameters so it doesn't overwrite the cars!
        import copy
        bus_params = copy.deepcopy(params) if params else copy.deepcopy(CAR_PARAMS)
        bus_params.length = 11.0 # REAL BUS LENGTH
        bus_params.width = 2.5   # NEW: REAL BUS WIDTH
        # Bus steering limits: large vehicle, professional driver.
        # steer_rate and lane_keeping_noise are sampled per-episode (LogNormal)
        # so that PPO must generalise across bus individuals, not memorise one vehicle.
        bus_params.wheelbase = 7.0              # long urban bus: rear-to-front axle
        bus_params.max_steer_angle_deg = 15.0   # professional driver, limited rack
        # steer_rate: median 25 °/s, CV≈10 % (tighter than cars — professional driver)
        bus_params.steer_rate_deg_per_s = float(np.clip(
            np.random.lognormal(np.log(25.0), 0.10), 18.0, 33.0))
        # lane_keeping_noise: median 0.05 m, CV≈15 %
        # lane_keeping_noise is now intentionally tiny: the OU perturbation is fed
        # into the pure-pursuit target (_lat_perturb below) so the driver actively
        # corrects it.  This residual represents actuator-level micro-wobble only.
        bus_params.lane_keeping_noise = float(np.clip(
            np.random.lognormal(np.log(0.01), 0.15), 0.005, 0.025))

        super().__init__(v_id, position, velocity, bus_params)
        self.state = BusState.MOVING
        self.stop_duration = 15.0
        self.target_stop_x = 300.0
        self.dwell_timer = 0.0
        self.wait_time = 0.0
        self._merge_progress = 0.0 
        self.merge_start_x = position
        self.exit_taper_length = 15.0
        self.entry_taper_length = 50.0   # matches the 50 m DECELERATING trigger → ramp starts immediately
        self.max_merge_progress_step = 0.02
        # Active lane-correction OU state: perturbation added to lat_target so
        # pure pursuit actively steers back (visible small oscillations around centre).
        self._lat_perturb = 0.0

    def set_stop_schedule(self, location, duration):
        self.target_stop_x = location
        self.stop_duration = duration

    def get_turn_angle_degrees(self):
        """Heading angle from the kinematic bicycle model (persistent state).
        When v=0 the heading stays exactly where it was — no snap to zero."""
        return float(np.degrees(self.heading_angle))

    def projected_length_along_x(self):
        """Projected longitudinal footprint when bus is rotated during tapers."""
        theta = np.radians(abs(self.get_turn_angle_degrees()))
        return (self.params.length * np.cos(theta)) + (self.params.width * np.sin(theta))

    @property
    def presence_factor(self):
        """Fraction of the traffic lane currently occupied by the bus (0=clear, 1=full block).

        Uses the maximum Y-coordinate of the bus's 4 rotated polygon corners so that
        tail-swing is correctly detected when the bus is angled.  This means a bus
        parked at -8° in the bay with its tail still poking into the lane will block
        cars exactly as much as its physical footprint demands — no approximation.

        Note: IN_BAY is no longer short-circuited to 0.0.  When the bus parks with a
        residual heading the tail corner may still reach the traffic lane; the polygon
        maths returns the correct small-but-nonzero value in that case and 0.0
        automatically once the bus is truly parallel to the kerb (heading ≈ 0°)."""
        if self.state == BusState.STOPPED_IN_LANE:
            return 1.0
        # Explicit 4-corner polygon computation.
        # Body-frame corners: (±half_l, ±half_w)
        # World y-coordinate: lateral_offset  +  corner_x*sin(θ)  +  corner_y*cos(θ)
        theta  = getattr(self, 'heading_angle', 0.0)
        c, s   = np.cos(theta), np.sin(theta)
        half_l = self.params.length / 2.0
        half_w = self.params.width  / 2.0
        lo     = self.lateral_offset
        # Y-coordinates of the four corners in the world (road) frame.
        corners_y = np.array([
            lo + half_l * s + half_w * c,   # front-left
            lo + half_l * s - half_w * c,   # front-right
            lo - half_l * s + half_w * c,   # rear-left
            lo - half_l * s - half_w * c,   # rear-right
        ])
        bus_max_y = float(np.max(corners_y))
        bus_min_y = float(np.min(corners_y))
        # Traffic lane spans [-1.75, +1.75]
        LANE_W   = 3.5
        overlap  = max(0.0, min(bus_max_y, LANE_W / 2) - max(bus_min_y, -LANE_W / 2))
        return float(np.clip(overlap / LANE_W, 0.0, 1.0))

from enum import Enum
class BusState(Enum):
    MOVING = 0
    DECELERATING = 1
    STOPPED_IN_LANE = 2
    IN_BAY = 3
    WAITING_TO_MERGE = 4