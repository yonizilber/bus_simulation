import unittest
from vehicle import Vehicle, Bus, BusState, CAR_PARAMS, VehicleParams
from simulation import SimulationEngine
from config import SimConfig

class TestHumanBehavior(unittest.TestCase):
    
    def setUp(self):
        self.cfg = SimConfig()
        self.sim = SimulationEngine(scenario="B", config=self.cfg)
        self.sim.vehicles = []

    def _make_bus(self, presence=0.0):
        """Helper to create a properly initialized bus"""
        bus = Bus(v_id=999, position=300.0, velocity=0.0)
        bus.set_stop_schedule(location=300.0, duration=15.0) 
        bus.state = BusState.WAITING_TO_MERGE
        bus.presence_factor = presence
        return bus

    def test_squeeze_1_success(self):
        """
        Scenario: Dilemma Zone.
        Car is fast & close. Braking is unsafe. Bus is barely out.
        Expected: SQUEEZE (Maintain/Increase Speed).
        """
        print("\n--- Test: Squeeze Success (Dilemma Zone) ---")
        bus = self._make_bus(presence=0.2)
        
        car = Vehicle(v_id=1, position=255.0, velocity=25.0, params=CAR_PARAMS)
        car.reaction_timer = 0.0 
        # FIX: Prime memory so car isn't "Shocked" by seeing the bus
        car.last_presence_seen = 0.2 
        
        # Manual Physics Step
        leader_state = {'position': bus.position, 'velocity': 0, 'length': 5, 'presence': 0.2}
        car.update_physics(0.1, leader_state)
        
        print(f"   [Result] Car Accel: {car.acceleration:.2f}")
        print(f"   [Result] Is Squeezing: {car.is_squeezing}")
        
        self.assertTrue(car.is_squeezing, "FAILURE: Car did not enter SQUEEZE mode.")
        # Accel should be high positive (speeding up) or at least not hard braking
        self.assertGreater(car.acceleration, -1.0, "FAILURE: Car is braking hard instead of squeezing!")

    def test_squeeze_2_unnecessary(self):
        """
        Scenario: Safe Stop Available.
        Car is fast but far away.
        Expected: YIELD (Normal Braking), NOT Squeeze.
        """
        print("\n--- Test: Squeeze Unnecessary (Safe Stop) ---")
        bus = self._make_bus(presence=0.2)
        
        # FIX: Increase gap to 150m (Position 145.0 vs Bus 300.0)
        # Old Gap (100m) -> Req Brake 3.12 m/s^2 (Too hard -> Squeeze)
        # New Gap (150m) -> Req Brake 2.08 m/s^2 (Comfortable -> Yield)
        car = Vehicle(v_id=1, position=145.0, velocity=25.0, params=CAR_PARAMS)
        car.reaction_timer = 0.0
        car.last_presence_seen = 0.2
        
        leader_state = {'position': bus.position, 'velocity': 0, 'length': 5, 'presence': 0.2}
        car.update_physics(0.1, leader_state)
        
        print(f"   [Result] Car Accel: {car.acceleration:.2f}")
        print(f"   [Result] Is Squeezing: {car.is_squeezing}")

        self.assertFalse(car.is_squeezing, "FAILURE: Car squeezed when it could have stopped safely.")
        self.assertLess(car.acceleration, -0.5, "FAILURE: Car should be braking for the bus.")

    def test_squeeze_3_panic_abort(self):
        """
        Scenario: Squeeze started, but Bus gets aggressive.
        Expected: ABORT SQUEEZE (Panic Brake).
        """
        print("\n--- Test: Squeeze Panic Abort ---")
        bus = self._make_bus(presence=0.9) # Bus blocking
        
        # 5m Gap. 
        car = Vehicle(v_id=1, position=290.0, velocity=25.0, params=CAR_PARAMS)
        car.is_squeezing = True 
        car.reaction_timer = 0.0
        # FIX: Prime memory to avoid shock (we want to test the panic logic, not shock logic)
        car.last_presence_seen = 0.9
        
        leader_state = {'position': bus.position, 'velocity': 0, 'length': 5, 'presence': 0.9}
        car.update_physics(0.1, leader_state)
        
        print(f"   [Result] Car Accel: {car.acceleration:.2f}")
        print(f"   [Result] Is Squeezing: {car.is_squeezing}")

        self.assertFalse(car.is_squeezing, "FAILURE: Car continued squeezing despite imminent crash!")
        # Should rely on standard IDM braking now (-9.0)
        self.assertEqual(car.acceleration, -9.0, "FAILURE: Car did not slam emergency brakes!")

    def test_bus_shock_recovery(self):
        print("\n--- Test: Bus Shock Recovery ---")
        bus = self._make_bus(presence=0.9)
        maniac = Vehicle(v_id=1, position=290.0, velocity=40.0, params=CAR_PARAMS)
        
        self.sim.vehicles = [bus, maniac]
        
        # Step 1: Scare
        self.sim.step()
        self.assertTrue(bus.shock_timer > 0, "FAILURE: Shock timer not set.")
        
        # Step 2: Clear Road
        self.sim.vehicles = [bus]
        self.sim.step()
        
        self.assertEqual(bus.state, BusState.WAITING_TO_MERGE, "FAILURE: Bus merged while shocked!")
        print("   [Pass] Bus waited while in shock.")

    def test_driver_latency(self):
        """
        Scenario:
        - Leader slams brakes instantly.
        - Follower has 'reaction_timer' > 0.
        - EXPECTED: Follower does NOT update acceleration in the first frame.
        """
        print("\n--- Test: Driver Latency (OODA Loop) ---")
        
        # 1. Setup
        leader = {'position': 100.0, 'velocity': 0.0, 'length': 5.0, 'presence': 1.0} # Stopped wall
        car = Vehicle(v_id=1, position=80.0, velocity=20.0, params=CAR_PARAMS)
        
        # Set latency to 0.5s
        car.reaction_timer = 0.5
        car.acceleration = 5.0 # Assume car was accelerating previously
        
        # 2. Update Physics
        car.update_physics(0.1, leader)
        
        # 3. Assertions
        # Even though there is a wall ahead, acceleration should remain 5.0 
        # because the driver hasn't "reacted" yet.
        self.assertEqual(car.acceleration, 5.0, 
                         "FAILURE: Driver reacted instantly (ignored latency timer).")
        print(f"   [Result] Car Accel maintained at {car.acceleration} (Reaction Timer Active)")


if __name__ == '__main__':
    unittest.main()

