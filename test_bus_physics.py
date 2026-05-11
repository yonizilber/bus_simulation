"""
Test: Bus entering the bay with steering-first kinematic bicycle model.

Runs standalone -- uses only vehicle.py physics (no torch / simulation.py).

Verifies:
  1. heading_angle is continuous (no jumps > 2 deg between steps)
  2. heading_angle is NON-ZERO when the bus stops in the bay
  3. heading_angle PERSISTS (frozen) while velocity is zero
  4. Bus reaches approximately -3.5 m lateral offset (the bay)
"""
import sys
import numpy as np
from vehicle import Bus, BusState, CAR_PARAMS


def test_bus_entry_heading():
    dt = 0.1
    bus = Bus(v_id=999, position=0.0, velocity=10.0, params=CAR_PARAMS)
    bus.set_stop_schedule(300.0, 15.0)
    bus.state = BusState.MOVING

    headings = []
    positions = []
    states = []
    lat_offsets = []
    velocities = []
    steer_angles = []
    lat_targets = []   # track autopilot target for telemetry

    max_steps = 1200
    for step_i in range(max_steps):
        dist_to_stop = bus.target_stop_x - bus.position

        # -- STATE MACHINE (mirrors _handle_bus_logic) --
        if bus.state == BusState.MOVING:
            bus.lat_target = 0.0
        elif bus.state == BusState.DECELERATING:
            # PATH LOOK-AHEAD: linear ramp from lat=0 at taper entry (dist=entry_taper_length)
            # to lat=-3.5 at PATH_LOOK_AHEAD metres before the stop.
            # entry_taper_length=50 matches the DECELERATING trigger so the ramp starts immediately.
            if 0.0 < dist_to_stop <= bus.entry_taper_length:
                PATH_LOOK_AHEAD = 18.0
                ramp = bus.entry_taper_length - PATH_LOOK_AHEAD   # 32 m
                look_frac = (bus.entry_taper_length - dist_to_stop) / ramp
                bus.lat_target = max(-3.5, -3.5 * look_frac)
            elif dist_to_stop <= 0.0 and abs(bus.heading_angle) > np.radians(2.0):
                # PARKING CRAWL: hold at bay depth while heading recovers.
                bus.lat_target = -3.5
            else:
                bus.lat_target = -3.5
        elif bus.state == BusState.IN_BAY:
            bus.lat_target = bus.lateral_offset

        # Arrival logic
        if bus.state == BusState.MOVING and 0 < dist_to_stop < 50:
            bus.state = BusState.DECELERATING

        if bus.state == BusState.DECELERATING and dist_to_stop <= 0.5:
            if abs(bus.heading_angle) > np.radians(2.0):
                # Parking crawl: keep rolling to let pure-pursuit straighten the wheelbase.
                bus.velocity = max(bus.velocity, 1.5)
            else:
                # Accept the crawl position — no snap back to target_stop_x.
                bus.velocity = 0.0
                bus.dwell_timer = bus.stop_duration
                bus.state = BusState.IN_BAY
                bus.wait_time = 0.0

        # Hard stop while in bay — skip physics entirely (frozen state)
        if bus.state == BusState.IN_BAY:
            bus.velocity = 0.0
            bus.acceleration = 0.0
        else:
            # Virtual stop leader for IDM braking (offset by min_gap so bus stops at target)
            lead_vehicle = None
            if bus.state == BusState.DECELERATING:
                lead_vehicle = {'id': -1, 'position': bus.target_stop_x + bus.params.min_gap,
                                'velocity': 0.0, 'length': 0.0, 'presence': 1.0}

            # Physics step (bicycle model + IDM)
            bus.update_physics(dt, lead_vehicle, bus=None)

            # POST-PHYSICS PARKING CRAWL FLOOR: re-apply after IDM may have braked
            if bus.state == BusState.DECELERATING and abs(bus.heading_angle) > np.radians(2.0):
                bus.velocity     = max(bus.velocity,     1.5)
                bus.acceleration = max(bus.acceleration, 0.0)

        # Record
        headings.append(float(np.degrees(bus.heading_angle)))
        positions.append(bus.position)
        states.append(bus.state.name)
        lat_offsets.append(bus.lateral_offset)
        velocities.append(bus.velocity)
        steer_angles.append(float(np.degrees(bus.steer_angle)))
        lat_targets.append(bus.lat_target)

        # Stop once bus has been in bay for a few seconds
        if bus.state == BusState.IN_BAY and bus.dwell_timer < bus.stop_duration - 2.0:
            break
        if bus.state == BusState.IN_BAY:
            bus.dwell_timer -= dt

    # == RESULTS ==
    print("")
    print("=" * 60)
    print("  STEERING-FIRST BUS PHYSICS -- ENTRY TEST")
    print("=" * 60)
    print(f"Total steps simulated: {len(headings)}")
    print(f"Final state:           {states[-1]}")
    print(f"Final position:        {positions[-1]:.2f} m")
    print(f"Final lateral offset:  {lat_offsets[-1]:.3f} m")
    print(f"Final heading:         {headings[-1]:.4f} deg")
    print(f"Final steer angle:     {steer_angles[-1]:.4f} deg")
    print(f"Final velocity:        {velocities[-1]:.3f} m/s")

    # -- CHECK 1: Heading continuity --
    max_jump = 0.0
    max_jump_step = 0
    for i in range(1, len(headings)):
        jump = abs(headings[i] - headings[i - 1])
        if jump > max_jump:
            max_jump = jump
            max_jump_step = i
    print(f"\nCheck 1 -- Max heading jump: {max_jump:.4f} deg (at step {max_jump_step})")
    assert max_jump < 2.0, f"FAIL: Heading jumped {max_jump:.2f} deg in one step (limit: 2)"
    print("  PASS: heading is continuous")

    # -- CHECK 2: Heading non-zero at stop --
    in_bay_headings = [h for h, s in zip(headings, states) if s == 'IN_BAY']
    if in_bay_headings:
        bay_heading = in_bay_headings[0]
        print(f"\nCheck 2 -- Heading when first in bay: {bay_heading:.4f} deg")
        assert abs(bay_heading) > 0.1, \
            f"FAIL: Heading snapped to zero at stop ({bay_heading:.4f} deg)"
        print("  PASS: heading is non-zero at stop")
    else:
        print("\n  WARNING: Bus never reached IN_BAY state")
        sys.exit(1)

    # -- CHECK 3: Heading persists while stopped --
    in_bay_unique = set(round(h, 6) for h in in_bay_headings)
    print(f"\nCheck 3 -- Unique headings while in bay: {len(in_bay_unique)}")
    assert len(in_bay_unique) == 1, \
        f"FAIL: Heading changed while bus was stopped ({len(in_bay_unique)} distinct values)"
    print("  PASS: heading frozen while v=0")

    # -- CHECK 4: Bus reached the bay --
    # With lat_target = -3.5 always, the bus should reach close to full bay depth.
    final_lat = lat_offsets[-1]
    print(f"\nCheck 4 -- Lateral offset at bay: {final_lat:.3f} m (target: -3.5)")
    assert final_lat < -2.5, \
        f"FAIL: Bus didn't steer into bay (lateral offset = {final_lat:.3f})"
    print("  PASS: bus reached bay depth")

    # -- CHECK 5: Heading at stop is near-parallel to road --
    print(f"\nCheck 5 -- Heading at stop: {final_lat:.3f} m bay, {in_bay_headings[0]:.2f}° heading")
    assert abs(in_bay_headings[0]) < 10.0, \
        f"FAIL: Bus parks at {in_bay_headings[0]:.1f}° (limit 10°) — look-ahead not effective"
    print("  PASS: bus parks near-parallel to curb (< 10°)")

    # -- ENTRY MANOEUVRE TELEMETRY (every frame from taper entry to bay stop) --
    # Format: Time | Velocity | Lat_Target | Actual_Y | Heading | Steer_Angle
    print("")
    print("=" * 80)
    print("  ENTRY MANOEUVRE TELEMETRY  (frame-by-frame from taper entry)")
    print("=" * 80)
    tel_header = (f"{'Time':>7} | {'Vel m/s':>7} | {'LatTgt m':>8} | "
                  f"{'ActualY m':>9} | {'Heading°':>8} | {'Steer°':>7}")
    print(tel_header)
    print("-" * 80)
    in_manoeuvre = False
    for i in range(len(states)):
        t_i = round(i * dt, 2)
        if states[i] in ('DECELERATING', 'IN_BAY'):
            in_manoeuvre = True
        if in_manoeuvre:
            print(f"{t_i:7.2f} | {velocities[i]:7.3f} | {lat_targets[i]:8.3f} | "
                  f"{lat_offsets[i]:9.4f} | {headings[i]:8.4f} | {steer_angles[i]:7.4f}")
        # Stop telemetry 3 s after entering IN_BAY
        if states[i] == 'IN_BAY' and i > 0 and states[i - 1] == 'IN_BAY':
            if (t_i - next((i2 * dt for i2, s in enumerate(states) if s == 'IN_BAY'), t_i)) > 3.0:
                break

    # -- TRAJECTORY SAMPLE --
    print("")
    print("=" * 60)
    print("  TRAJECTORY (sampled)")
    print("=" * 60)
    header = f"{'Step':>5} {'State':>15} {'X':>8} {'V':>6} {'Heading':>9} {'Steer':>8} {'LatOff':>8}"
    print(header)
    prev_state = None
    for i in range(len(headings)):
        show = (i % 20 == 0) or (states[i] != prev_state) or i == len(headings) - 1
        if show:
            print(f"{i:5d} {states[i]:>15} {positions[i]:8.2f} {velocities[i]:6.2f} "
                  f"{headings[i]:9.4f} {steer_angles[i]:8.4f} {lat_offsets[i]:8.4f}")
        prev_state = states[i]

    print("")
    print("=" * 60)
    print("  ALL CHECKS PASSED")
    print("  Heading is continuous, non-zero at stop, and persistent.")
    print("  The kinematic bicycle model is the single source of truth.")
    print("=" * 60)


if __name__ == "__main__":
    test_bus_entry_heading()
