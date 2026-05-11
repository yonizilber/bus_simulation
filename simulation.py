import pandas as pd
import random
import numpy as np
import torch
from typing import List
from vehicle import Vehicle, Bus, BusState, CAR_PARAMS, sample_vehicle_params
from config import SimConfig
from rl.ppo_agent import PPOAgent

class SimulationEngine:
    def __init__(self, scenario="A", config: SimConfig = None):
        self.scenario = scenario
        self.cfg = config if config else SimConfig()
        self.road_length = self.cfg.road_length
        self.dt = 0.1
        self.time = 0.0
        self.vehicles: List[Vehicle] = []
        self.frame_history = []
        self.global_id_counter = 0 
        self.cars_finished_count = 0
        self.last_spawn_time = 0.0
        self.traffic_rate = self.cfg.traffic_rate
        self.cars_spawned_count = 0
        
        # --- LOAD THE TRAINED PYTORCH BRAIN ---
        # Search for the checkpoint in multiple locations so the app works both
        # locally (checkpoints/ root) and on Streamlit Cloud (rl/checkpoints/).
        import os as _os
        _candidates = [
            "checkpoints/ppo_bus_brain.pth",
            "rl/checkpoints/ppo_bus_brain.pth",
            _os.path.join(_os.path.dirname(__file__), "checkpoints", "ppo_bus_brain.pth"),
        ]
        _ckpt = next((p for p in _candidates if _os.path.exists(p)),
                     "checkpoints/ppo_bus_brain.pth")
        self.brain = PPOAgent(state_dim=11, filename=_ckpt)
        self.brain.load() # Loads the trained weights from your Hard Mode run
        self._prev_bus_vel = 0.0  # for rel_accel computation in brain state

    @staticmethod
    def _encode_brain_state(merge_progress, bus_angle_rad, lon_dist, lat_clearance,
                             car_speed, car_accel, car_width, car_length,
                             rel_speed, rel_accel, car_angle_rad):
        """Encode 11-feature state vector for inference. No noise (production mode)."""
        merge_progress = float(np.clip(merge_progress,   0.0,   1.0))
        bus_angle_rad  = float(np.clip(bus_angle_rad,   -0.30,  0.30))
        lon_dist       = float(np.clip(lon_dist,        -10.0,  60.0))
        lat_clearance  = float(np.clip(lat_clearance,   -2.0,   4.0))
        car_speed      = float(np.clip(car_speed,        0.0,  20.0))
        car_accel      = float(np.clip(car_accel,       -9.0,   3.0))
        car_width      = float(np.clip(car_width,        1.70,  2.05))
        car_length     = float(np.clip(car_length,       3.5,   5.5))
        rel_speed      = float(np.clip(rel_speed,       -15.0,  15.0))
        rel_accel      = float(np.clip(rel_accel,       -15.0,  15.0))
        car_angle_rad  = float(np.clip(car_angle_rad,   -0.15,  0.15))

        p_n    = merge_progress
        ang_n  = bus_angle_rad / 0.30
        lon_n  = (lon_dist + 10.0) / 70.0
        lat_n  = (lat_clearance + 2.0) / 6.0
        spd_n  = car_speed / 20.0
        acc_n  = car_accel / 9.0
        wid_n  = (car_width  - 1.85) / 0.20
        len_n  = (car_length - 4.5)  / 0.75
        rs_n   = rel_speed / 15.0
        ra_n   = rel_accel / 15.0
        ca_n   = car_angle_rad / 0.15

        return np.array([p_n, ang_n, lon_n, lat_n, spd_n, acc_n,
                         wid_n, len_n, rs_n, ra_n, ca_n], dtype=np.float32)

    def init_scenario(self):
        self.vehicles = []
        self.global_id_counter = 0 
        self.time = 0.0
        self._prev_bus_vel = 0.0
        self._bus_heading_prev = 0.0   # sniffer: heading at start of last step
        self.heading_anomaly = False   # set True for the rest of episode on first snap
        self.frame_history = []
        self.cars_spawned_count = 0
        self.cars_finished_count = 0 
        
        # Spawn the bus at the start line (0.0).
        # Initial speed is uniformly sampled ±15 % around 10 m/s so that
        # different episodes produce different entry speeds and stop positions.
        spawn_speed = np.random.uniform(8.5, 11.5)
        bus = Bus(v_id=999, position=0.0, velocity=spawn_speed, params=CAR_PARAMS)

        # Stop position: nominal 300 m ± Gaussian σ=5 m (professional driver varies
        # aim point slightly).  Keeps stop within visible road section (285–315 m).
        stop_x = float(np.clip(np.random.normal(300.0, 5.0), 285.0, 315.0))
        bus.set_stop_schedule(stop_x, self.cfg.bus_dwell_time)
        
        self.vehicles.append(bus)

    def step(self):
        self.time += self.dt
        self._try_spawn_car()
        self.vehicles.sort(key=lambda v: v.position, reverse=True)
        the_bus = next((v for v in self.vehicles if isinstance(v, Bus)), None)

        for i, veh in enumerate(self.vehicles):
            # 1. FIND THE LEADER
            lead_vehicle = None
            if i > 0:
                potential_lead = self.vehicles[i-1]
                lead_length_for_dist = potential_lead.params.length
                if isinstance(potential_lead, Bus):
                    lead_length_for_dist = potential_lead.projected_length_along_x()
                dist = potential_lead.position - veh.position - lead_length_for_dist
                ignore_bus = False
                
                if isinstance(potential_lead, Bus):
                    if potential_lead.state == BusState.IN_BAY or dist > 100.0:
                        ignore_bus = True

                if ignore_bus:
                    if i > 1: lead_vehicle = self._get_veh_state(self.vehicles[i-2])
                else:
                    lead_vehicle = self._get_veh_state(potential_lead)

            # 2. RUN LOGIC & TELEMETRY
            if isinstance(veh, Bus):
                self._handle_bus_logic(veh)
                
                # Hard stop the bus physically if it is in these states
                if veh.state in [BusState.IN_BAY, BusState.STOPPED_IN_LANE]:
                    veh.velocity = 0.0
                    veh.acceleration = 0.0
            else:
                # Car Telemetry
                if lead_vehicle:
                    # Front-to-rear gap: rear of leader minus front of follower
                    # leader.position = leader's FRONT (centre in simulation, but leader.length
                    # is already subtracted here giving correct bumper-to-bumper)
                    gap = lead_vehicle['position'] - lead_vehicle['length'] - veh.position
                    veh.telemetry_leader = f"ID {lead_vehicle.get('id', '?')} (front-rear gap: {gap:.1f}m)"
                    veh._telemetry_front_gap = gap
                    # Lateral gap: side-to-side clearance between the two cars
                    lv_obj = next((u for u in self.vehicles if u.id == lead_vehicle.get('id')), None)
                    if lv_obj is not None:
                        lv_y  = getattr(lv_obj, 'total_lateral_offset', 0.0)
                        car_y = getattr(veh,    'total_lateral_offset', 0.0)
                        veh._telemetry_lateral_gap = abs(lv_y - car_y) - (veh.params.width + lv_obj.params.width) / 2.0
                    else:
                        veh._telemetry_lateral_gap = float('nan')
                else:
                    veh.telemetry_leader = "Clear Road"
                    veh._telemetry_front_gap   = float('nan')
                    veh._telemetry_lateral_gap = float('nan')
                
                if the_bus:
                    dist_to_bus = the_bus.position - veh.position
                    if dist_to_bus > 100.0 or dist_to_bus < 0 or the_bus.state == BusState.IN_BAY:
                        veh.telemetry_bus = "Out of Sight"
                    else:
                        veh.telemetry_bus = f"{the_bus.presence_factor:.2f}"
                else:
                    veh.telemetry_bus = "No Bus"

            # Frozen states: skip bicycle model.
            if isinstance(veh, Bus) and veh.state in (BusState.IN_BAY, BusState.STOPPED_IN_LANE):
                pass
            else:
                # PHYSICS SNIFFER: capture heading BEFORE physics so we can measure
                # the per-frame change and flag any super-physical snap.
                if isinstance(veh, Bus):
                    _heading_before = veh.heading_angle

                # For DECELERATING bus: virtual stationary leader at bus stop gives
                # the IDM a smooth braking target instead of a sudden snap-stop.
                # SCOPE: lead_vehicle is reset to None at the top of every loop iteration,
                # so this assignment only affects the Bus vehicle's own IDM calculation —
                # it CANNOT leak to cars even though lead_vehicle is a loop-local variable.
                if isinstance(veh, Bus) and veh.state == BusState.DECELERATING:
                    # Offset by min_gap so IDM converges to gap=min_gap AT the stop position.
                    stop_leader = {'id': -1, 'position': veh.target_stop_x + veh.params.min_gap,
                                   'velocity': 0.0, 'length': 0.0, 'presence': 1.0}
                    if lead_vehicle is None or (lead_vehicle['position'] - lead_vehicle['length']) > veh.target_stop_x:
                        lead_vehicle = stop_leader

                # 3. UPDATE PHYSICS (FOR EVERYONE!)
                # If the bus is MOVING or DECELERATING, it will use the IDM physics to drive forward
                veh.update_physics(self.dt, lead_vehicle, bus=the_bus)

                # PHYSICS SNIFFER: compare heading after update against the
                # maximum physically possible change (steer_rate × dt + 0.01 rad
                # tolerance for floating-point rounding).
                if isinstance(veh, Bus):
                    heading_change = abs(veh.heading_angle - _heading_before)
                    steer_rate_rad = np.radians(
                        getattr(veh.params, 'steer_rate_deg_per_s', 25.0))
                    max_allowed    = steer_rate_rad * self.dt + 0.01   # rad
                    if heading_change > max_allowed:
                        self.heading_anomaly = True
                        print(
                            f"[CRITICAL WARNING] Heading Snap Detected! "
                            f"t={self.time:.2f}s  Δψ={np.degrees(heading_change):.3f}°  "
                            f"(max allowed {np.degrees(max_allowed):.3f}°)"
                        )

                # POST-PHYSICS PARKING CRAWL FLOOR:
                # The IDM may brake the bus to zero even after _handle_bus_logic raised
                # the velocity floor, because IDM sees the virtual stop-leader and
                # produces a very large negative acceleration.  Re-apply the floor here
                # so the bicycle model always has enough forward momentum to finish
                # rotating the 7 m wheelbase to road-parallel.
                if (isinstance(veh, Bus) and veh.state == BusState.DECELERATING
                        and abs(veh.heading_angle) > np.radians(2.0)):
                    veh.velocity     = max(veh.velocity,     1.5)
                    veh.acceleration = max(veh.acceleration, 0.0)

        self._record_frame()
        self._remove_finished_cars()

    def _handle_bus_logic(self, bus: Bus):
        dist_to_stop = bus.target_stop_x - bus.position

        # ── STEERING-FIRST AUTOPILOT ─────────────────────────────────────────────
        # Set lat_target for the bicycle model in update_physics.
        # No direct writes to lateral_velocity or position for these states.
        if bus.state == BusState.MOVING:
            bus.lat_target = 0.0                     # stay centred in lane
        elif bus.state == BusState.DECELERATING:
            if 0.0 < dist_to_stop <= bus.entry_taper_length:
                # PATH LOOK-AHEAD: linear ramp from lat=0 at the taper entry point
                # (dist = entry_taper_length = 50 m, matching the DECELERATING trigger)
                # to lat=-3.5 at PATH_LOOK_AHEAD metres before the stop.
                # This starts the bay approach immediately when DECELERATING begins —
                # no intermediate re-alignment to lane centre (the S-curve bug).
                # The bus reaches full bay depth with 18 m to go, giving the 7 m
                # wheelbase enough road to straighten back to 0° heading.
                PATH_LOOK_AHEAD = 18.0
                ramp = bus.entry_taper_length - PATH_LOOK_AHEAD   # 32 m
                look_frac = (bus.entry_taper_length - dist_to_stop) / ramp
                bus.lat_target = max(-3.5, -3.5 * look_frac)
            elif dist_to_stop <= 0.0 and abs(bus.heading_angle) > np.radians(2.0):
                # PARKING CRAWL: bus has passed the nominal stop but is still
                # angled more than 2°.  Keep it at bay depth so the pure-pursuit
                # controller has a lateral target to steer against while the
                # 7 m wheelbase finishes straightening.
                bus.lat_target = -3.5
            else:
                # dist_to_stop <= 0 and heading < 2°: bus aligned in bay,
                # one transitional frame before IN_BAY takes over.
                bus.lat_target = -3.5
        elif bus.state in (BusState.IN_BAY, BusState.STOPPED_IN_LANE):
            bus.lat_target = bus.lateral_offset       # hold position (v=0 → frozen)
        elif bus.state == BusState.WAITING_TO_MERGE:
            bus.lat_target = 0.0   # destination: traffic lane centre

        # 1. ARRIVAL LOGIC
        if bus.state == BusState.MOVING and 0 < dist_to_stop < 50:
            bus.state = BusState.DECELERATING
        
        if bus.state == BusState.DECELERATING and dist_to_stop <= 0.5:
            if abs(bus.heading_angle) > np.radians(2.0):
                # PARKING CRAWL: the 7 m wheelbase needs forward momentum to
                # rotate the nose back to road-parallel.  Refuse the final stop
                # until heading is within 2°; maintain a minimum crawl speed so
                # the pure-pursuit controller can finish the correction.
                bus.velocity = max(bus.velocity, 1.5)   # m/s crawl floor
            else:
                # Bus has stopped and is road-parallel: accept whatever longitudinal
                # position the crawl left it at — do NOT snap back to target_stop_x,
                # as that would teleport the bus backward 8–10 m.
                bus.velocity = 0.0
                bus.dwell_timer = bus.stop_duration
                bus.state = BusState.STOPPED_IN_LANE if self.scenario == "A" else BusState.IN_BAY
                bus.wait_time = 0.0 

        # 2. DWELL LOGIC
        if bus.state in [BusState.IN_BAY, BusState.STOPPED_IN_LANE]:
            bus.dwell_timer -= self.dt
            if bus.dwell_timer <= 0:
                if self.scenario == "A":
                    bus.state = BusState.MOVING
                else:
                    bus.state = BusState.WAITING_TO_MERGE
                    bus._merge_progress = 0.0
                    bus.merge_start_x = bus.position
                    bus.wait_time = 0.0
                    # Physics init: bus is parked in bay, parallel to curb.
                    bus.lateral_offset   = -3.5
                    bus.velocity         = 0.0
                    bus.lat_target       = 0.0   # destination: traffic lane
                    # NOTE: heading_angle, steer_angle, lateral_velocity are
                    # intentionally NOT reset here so the bus inherits the exact
                    # physical state it had at the end of the parking manoeuvre.

        # 3. AI MERGE LOGIC — BICYCLE MODEL (The PyTorch Brain)
        if bus.state == BusState.WAITING_TO_MERGE:
            bus.wait_time += self.dt

            # ── SENSE (─────────────────────────────────────────────────────────────
            gap = 999.0
            nearest_speed = 0.0
            nearest_accel = 0.0
            nearest_width = 1.85
            nearest_car   = None
            for v in self.vehicles:
                if v.id != bus.id and not isinstance(v, Bus):
                    bus_length_x = bus.projected_length_along_x()
                    current_gap  = bus.position - bus_length_x - v.position
                    if -10 < current_gap < 60:
                        if abs(current_gap) < abs(gap):
                            gap           = current_gap
                            nearest_speed = v.velocity
                            nearest_accel = v.acceleration
                            nearest_width = v.params.width
                            nearest_car   = v

            # ── BRAIN (─────────────────────────────────────────────────────────────
            BUS_WIDTH, LANE_WIDTH = 2.5, 3.5
            # merge_progress derived from actual lateral position (no stored scalar)
            merge_progress = float(np.clip(
                (bus.lateral_offset - (-3.5)) / 3.5, 0.0, 1.0))
            bus._merge_progress = merge_progress   # keep for legacy telemetry
            bus_angle_rad  = bus.heading_angle
            car_lat_offset = getattr(nearest_car, 'total_lateral_offset', 0.0) if nearest_car else 0.0
            bus_lat_edge   = bus.lateral_offset + BUS_WIDTH / 2.0
            lat_clearance  = LANE_WIDTH - max(0.0, bus_lat_edge) - (nearest_width - car_lat_offset)
            bus_accel_now  = (bus.velocity - self._prev_bus_vel) / self.dt
            rel_accel      = bus_accel_now - nearest_accel
            self._prev_bus_vel = bus.velocity
            car_length     = nearest_car.params.length if nearest_car else 4.5
            car_angle_rad  = getattr(nearest_car, 'heading_angle', 0.0) if nearest_car else 0.0
            relative_speed = bus.velocity - nearest_speed

            state_array = self._encode_brain_state(
                merge_progress, bus_angle_rad, gap, lat_clearance,
                nearest_speed, nearest_accel, nearest_width, car_length,
                relative_speed, rel_accel, car_angle_rad)
            state_tensor = torch.FloatTensor(state_array).unsqueeze(0)

            with torch.no_grad():
                action_mean, _, _ = self.brain.policy(state_tensor)
            # 2D action: (steer_intent, gas_intent) in [-1, 1]
            raw_steer = float(action_mean[0, 0].item())
            raw_gas   = float(action_mean[0, 1].item())

            # ── SAFETY SHIELD — overrides steer/gas scalars only, never kinematics ──
            steer_intent = np.clip(raw_steer, -1.0, 1.0)
            gas_intent   = np.clip(raw_gas,   -1.0, 1.0)
            intervened   = False
            rel_speed    = bus.velocity - nearest_speed
            adaptive_min_gap = 4.0 + max(0.0, nearest_width - 1.80) * 2.2
            if gas_intent > 0 and gap > -5.0:
                if 0.0 <= gap < 0.50:
                    if merge_progress >= 0.30:   # committed — keep momentum
                        gas_intent = 1.0
                    else:
                        gas_intent = -1.0        # emergency stop
                    intervened = True
                elif merge_progress > 0.65 and 0.0 <= gap < 1.20 and rel_speed > 0.2:
                    gas_intent = -1.0
                    intervened = True
                elif 0.0 <= gap < adaptive_min_gap and rel_speed > 1.2:
                    gap_opening_safely = (
                        (gap > 1.4 and merge_progress > 0.40 and rel_speed < 3.5)
                        or (gap > 2.5 and merge_progress < 0.30 and rel_speed < 2.5)
                    )
                    if not gap_opening_safely:
                        risk = np.clip((adaptive_min_gap - gap) / max(adaptive_min_gap, 1e-3), 0.0, 1.0)
                        gas_intent = gas_intent * (1.0 - 0.85 * risk)
                        intervened = True
                elif raw_gas > 0.30 and 0.0 <= gap < (adaptive_min_gap + 6.0) and rel_speed > 2.5:
                    gas_intent = gas_intent * 0.35
                    intervened = True
            # Deadlock escape: guarantee a minimum gas after prolonged waiting.
            if bus.wait_time > 20.0 and gas_intent <= 0.0:
                gas_intent = 0.2

            # ── APPLY TARGETS — bicycle model does all movement ──
            BAY_DEPTH   = -3.5
            BUS_MAX_SPD = 8.33   # 30 km/h
            lat_target   = float(np.interp(steer_intent, [-1.0, 1.0], [BAY_DEPTH, 0.0]))
            speed_target = max(0.0, float(gas_intent)) * BUS_MAX_SPD
            bus.lat_target = lat_target
            speed_error    = speed_target - bus.velocity
            bus.acceleration = float(np.clip(2.0 * speed_error, -3.0, 2.5))

            # Merge complete: lateral_offset has reached lane centre
            if bus.lateral_offset >= -0.1:
                bus.state    = BusState.MOVING
                bus.lat_target = 0.0

        # heading_angle is updated in Phase D (WAITING_TO_MERGE) or by the bicycle
        # model (DECELERATING/MOVING) — no separate workaround needed.

    def _try_spawn_car(self):
        if self.cars_spawned_count >= self.cfg.total_cars_to_spawn: return
        if self.time - self.last_spawn_time < 0.5: return
        
        if len(self.vehicles) > 0:
            if self.vehicles[-1].position < 15.0: return 

        if random.random() < (self.traffic_rate * self.dt):
            self.global_id_counter += 1
            _, sampled_params = sample_vehicle_params()
            start_speed = max(5.5, sampled_params.max_speed * random.uniform(0.68, 0.95))
            c = Vehicle(self.global_id_counter, position=0.0, velocity=start_speed, params=sampled_params)
            self.vehicles.append(c)
            self.last_spawn_time = self.time
            self.cars_spawned_count += 1

    def all_vehicles_finished(self) -> bool:
        if self.cars_spawned_count < self.cfg.total_cars_to_spawn: return False
        if len(self.vehicles) == 0: return True
        return False

    def _get_veh_state(self, veh):
        length_for_gap = veh.params.length
        if isinstance(veh, Bus):
            length_for_gap = veh.projected_length_along_x()

        return {
            'id': veh.id,
            'position': veh.position,
            'velocity': veh.velocity,
            'length': length_for_gap,
            'presence': getattr(veh, 'presence_factor', 1.0)
        }

    def _record_frame(self):
        for v in self.vehicles:
            state_desc = v.state.name if isinstance(v, Bus) else "Cruising"
            if not isinstance(v, Bus):
                if v.is_squeezing: state_desc = "Squeezing!"
                elif v.reaction_timer > 0: state_desc = "Reacting..."

            final_presence = getattr(v, 'presence_factor', 1.0)
            leader_info = getattr(v, 'telemetry_leader', 'N/A')
            bus_vis_info = getattr(v, 'telemetry_bus', 'N/A')

            # Heading angle:
            # Bus: use _logged_heading_deg captured in _handle_bus_logic BEFORE update_physics
            #      overwrites lateral_velocity. This gives the true geometric heading.
            # Car: use heading_angle from the bicycle model (computed in update_physics).
            if isinstance(v, Bus):
                # heading_angle is kept in sync for all states:
                # DECELERATING → bicycle model; WAITING_TO_MERGE → arctan2 from motion
                heading_deg = float(np.degrees(v.heading_angle))
            else:
                heading_deg = float(np.degrees(getattr(v, 'heading_angle', 0.0)))

            # Physical collision: only flag as real overlap when BOTH gaps are simultaneously negative
            fg = getattr(v, '_telemetry_front_gap',   float('nan'))
            lg = getattr(v, '_telemetry_lateral_gap', float('nan'))
            physical_collision = (not np.isnan(fg) and not np.isnan(lg)
                                  and fg < 0.0 and lg < 0.0)

            self.frame_history.append({
                'time': self.time,
                'id': v.id,
                'type': "Bus" if isinstance(v, Bus) else "Car",
                'class': "Bus" if isinstance(v, Bus) else getattr(v, 'vehicle_class', 'Mid'),
                'x': v.position,
                'v': v.velocity,
                'length': v.params.length,
                'width': v.params.width,
                'state': state_desc,
                'presence': final_presence,
                'leader_info': leader_info,
                'bus_visibility': bus_vis_info,
                'lateral_offset': getattr(v, 'lateral_offset', 0.0),
                'lane_drift':     getattr(v, 'lane_drift', 0.0),
                'total_lat':      getattr(v, 'total_lateral_offset', 0.0),
                'steer_angle_deg': float(np.degrees(getattr(v, 'steer_angle', 0.0))),
                'heading_deg':    heading_deg,
                'politeness':      getattr(getattr(v, 'params', None), 'politeness_factor', float('nan')),
                'steer_rate':      getattr(getattr(v, 'params', None), 'steer_rate_deg_per_s', float('nan')),
                'front_gap':        fg,
                'lateral_gap':      lg,
                'physical_collision': physical_collision,
                'heading_anomaly':  self.heading_anomaly if isinstance(v, Bus) else False,
            })

    def _remove_finished_cars(self):
        active = []
        for v in self.vehicles:
            if v.position > self.road_length:
                self.cars_finished_count += 1
            else:
                active.append(v)
        self.vehicles = active

    def get_results(self):
        return pd.DataFrame(self.frame_history)

class AnalyticsEngine:
    def __init__(self, dataframe: pd.DataFrame, config: SimConfig):
        self.df = dataframe
        self.cfg = config

    def compute_kpis(self):
        if self.df.empty:
            return {"avg_person_speed": 0.0, "throughput_people": 0}

        vehicles = self.df.groupby('id')
        total_w_dist = 0.0
        total_w_time = 0.0
        total_people_arrived = 0

        for v_id, trip in vehicles:
            v_type = trip.iloc[0]['type']
            pax = self.cfg.pax_per_bus if v_type == 'Bus' else self.cfg.pax_per_car
            
            dist = trip['x'].max() - trip['x'].min()
            duration = trip['time'].max() - trip['time'].min()
            
            if duration < 0.1: continue

            total_w_dist += (dist * pax)
            total_w_time += (duration * pax)
            
            if trip['x'].max() > (self.cfg.road_length - 20):
                total_people_arrived += pax

        if total_w_time > 0:
            avg_person_speed = (total_w_dist / total_w_time) * 3.6
        else:
            avg_person_speed = 0.0
            
        return {
            "avg_person_speed": avg_person_speed,
            "throughput_people": int(total_people_arrived) 
        }