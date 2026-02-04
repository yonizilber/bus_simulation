import unittest
from vehicle import Vehicle, Bus, BusState, CAR_PARAMS, VehicleParams
from simulation import SimulationEngine

# --- Test Configuration ---
# We define a "Standard Test Car" to make the math easier to predict.
# max_speed=20m/s, max_accel=2m/s^2.
TEST_PARAMS = VehicleParams(
    max_speed=20.0, 
    max_accel=2.0, 
    comfortable_brake=2.0,
    min_gap=2.0, 
    time_headway=1.0, 
    length=5.0
)

import unittest
import numpy as np
from vehicle import Vehicle, Bus, BusState, CAR_PARAMS, VehicleParams
from simulation import SimulationEngine

# Standard Parameters
TEST_PARAMS = VehicleParams(
    max_speed=20.0, max_accel=2.0, comfortable_brake=2.0,
    min_gap=2.0, time_headway=1.0, length=5.0
)

class TestEdgeCases(unittest.TestCase):
    
    def setUp(self):
        self.sim = SimulationEngine()
        self.sim.vehicles = []

    def test_zero_division_crash(self):
        """
        EDGE CASE: TWO CARS AT EXACT SAME POSITION
        The IDM equation divides by gap^2. If gap is 0, this usually crashes.
        Our code handles this with a 'max(gap, 0.1)' clamp. Let's verify it works.
        """
        print("\n--- Edge Case 1: Zero Division (Crash Test) ---")
        
        # Place two cars exactly at x=100
        car_a = Vehicle(v_id=1, position=100, velocity=0, params=TEST_PARAMS)
        car_b = Vehicle(v_id=2, position=100, velocity=0, params=TEST_PARAMS)
        
        self.sim.vehicles = [car_a, car_b]
        
        try:
            self.sim.step()
            print(f"   [Result] Car B Accel: {car_b.acceleration}")
            print("   [Pass] Engine did not crash.")
        except ZeroDivisionError:
            self.fail("Engine crashed due to Division by Zero!")

    def test_negative_velocity_clamp(self):
        """
        EDGE CASE: EXTREME BRAKING
        If a car is 1m behind a stopped truck at 50m/s, IDM requests 
        insane braking (-1000 m/s^2).
        One step later, velocity could become negative (-50 m/s). 
        Physics engine MUST clamp velocity to 0.
        """
        print("\n--- Edge Case 2: Negative Velocity (Reversing) ---")
        
        car_a = Vehicle(v_id=1, position=100, velocity=0, params=TEST_PARAMS)
        # Car B is super close and super fast
        car_b = Vehicle(v_id=2, position=99, velocity=50, params=TEST_PARAMS)
        
        self.sim.vehicles = [car_a, car_b]
        
        self.sim.step() # Step 1: Huge deceleration calculated
        self.sim.step() # Step 2: Velocity update
        
        print(f"   [Result] Car B Velocity: {car_b.velocity}")
        
        self.assertGreaterEqual(car_b.velocity, 0.0, "FAILURE: Car is driving backwards!")
        print("   [Pass] Velocity correctly clamped to 0.")

    def test_bus_overshoot(self):
        """
        EDGE CASE: HIGH SPEED BUS ENTRY
        If the bus arrives at the stop too fast, it might skip the 'DECELERATING' 
        phase and miss the stop entirely.
        """
        print("\n--- Edge Case 3: High Speed Bus Entry ---")
        
        # Bus is 20m away, moving at 30 m/s (108 km/h). 
        # In 0.1s it moves 3m. It will hit the stop in ~0.6s.
        bus = Bus(v_id=999, position=280, velocity=30.0) 
        bus.set_stop_schedule(location=300.0, duration=10.0)
        
        self.sim.vehicles = [bus]
        
        # Run 20 steps (2 seconds)
        stop_achieved = False
        for _ in range(20):
            self.sim.step()
            if bus.state in [BusState.STOPPED_IN_LANE, BusState.IN_BAY]:
                stop_achieved = True
                break
        
        print(f"   [Result] Final State: {bus.state}, Final X: {bus.position}")
        
        self.assertTrue(stop_achieved, "FAILURE: Bus overshot the stop!")
        # Verify it snapped to exactly 300
        self.assertAlmostEqual(bus.position, 300.0, delta=0.5, msg="Bus stopped at wrong location")
        print("   [Pass] Bus stopped despite high entry speed.")

    def test_empty_road(self):
        """
        SYSTEM CHECK: EMPTY LIST
        Does the loop crash if list is empty?
        """
        print("\n--- Edge Case 4: Empty Road ---")
        self.sim.vehicles = []
        try:
            self.sim.step()
            print("   [Pass] No crash on empty list.")
        except Exception as e:
            self.fail(f"Crashed on empty list: {e}")

if __name__ == '__main__':
    unittest.main(verbosity=0)

class TestTrafficPhysics(unittest.TestCase):
    
    def setUp(self):
        """
        Runs before EVERY test. 
        Resets the Simulation Engine to a clean slate.
        """
        self.sim = SimulationEngine()
        self.sim.vehicles = []

    def test_basic_following(self):
        """
        --- TEST 1: IDM BRAKING LOGIC ---
        Scenario: 
            Car B (Fast, 15 m/s) is approaching Car A (Slow, 10 m/s).
            Gap is 10 meters.
        
        Expected Result:
            Car B must calculate a NEGATIVE acceleration (Braking).
            If acc >= 0, the physics engine would cause a crash.
        """
        print(self.test_basic_following.__doc__) # Print the docstring above
        
        # Setup: Place cars manually
        # Note: 'v_id' is used instead of 'id' to match vehicle.py
        car_a = Vehicle(v_id=1, position=100, velocity=10, params=TEST_PARAMS)
        car_b = Vehicle(v_id=2, position=90, velocity=15, params=TEST_PARAMS)
        
        # Inject into engine
        self.sim.vehicles = [car_a, car_b]
        
        # Action: Run 1 physics step (0.1s)
        self.sim.step()
        
        print(f"   [Result] Car B Acceleration: {car_b.acceleration:.2f} m/s^2")
        
        # Assertion
        self.assertLess(car_b.acceleration, 0.0, "FAILURE: Car B did not brake! Physics engine is unsafe.")
        print("   [Pass] Car B is braking as expected.")

    def test_bus_stop_trigger(self):
        """
        --- TEST 2: BUS STATE MACHINE ---
        Scenario: 
            Bus is driving at 10 m/s.
            Stop target is at x=300.
            Bus is at x=260 (40m away).
        
        Expected Result:
            The Bus Logic should detect the upcoming stop.
            State should switch from MOVING to DECELERATING.
        """
        print(self.test_bus_stop_trigger.__doc__)
        
        bus = Bus(v_id=999, position=260, velocity=10)
        bus.set_stop_schedule(location=300.0, duration=10.0)
        
        self.sim.vehicles = [bus]
        self.sim.step()
        
        print(f"   [Result] Bus State: {bus.state}")
        
        self.assertEqual(bus.state, BusState.DECELERATING, "FAILURE: Bus logic ignored the bus stop.")
        print("   [Pass] Bus correctly switched to DECELERATING state.")

    def test_scenario_b_bypass(self):
        """
        --- TEST 3: GHOST VEHICLE LOGIC (SCENARIO B) ---
        """
        print(self.test_scenario_b_bypass.__doc__)
        
        # 1. Car A: Far ahead
        car_a = Vehicle(v_id=1, position=500, velocity=20, params=TEST_PARAMS)
        
        # 2. Bus: stopped IN THE BAY
        bus = Bus(v_id=999, position=300, velocity=0)
        bus.state = BusState.IN_BAY 
        bus.set_stop_schedule(location=300, duration=10)
        
        # --- THE FIX IS HERE ---
        # We must give the bus 'time on the clock' so it doesn't leave immediately
        bus.dwell_timer = 5.0 
        # -----------------------

        # 3. Car B: Approaching
        car_b = Vehicle(v_id=2, position=200, velocity=20, params=TEST_PARAMS)
        
        self.sim.vehicles = [car_a, bus, car_b]
        
        # Action: Run 1 step
        self.sim.step()
        
        print(f"   [Result] Car B Acceleration: {car_b.acceleration:.2f} m/s^2")
        
        # Now it should pass
        self.assertGreater(car_b.acceleration, -0.5, "FAILURE: Car B is braking for the invisible bus!")
        print("   [Pass] Car B ignored the bus in the bay.")


if __name__ == '__main__':
    unittest.main(verbosity=0) # verbosity=0 clean output since we have print statements