import unittest
from vehicle import Vehicle, Bus, BusState, CAR_PARAMS, VehicleParams
from simulation import SimulationEngine
from config import SimConfig

class TestSafetyOverride(unittest.TestCase):
    
    def setUp(self):
        self.cfg = SimConfig()
        self.sim = SimulationEngine(scenario="B", config=self.cfg)
        self.sim.vehicles = []

    def test_emergency_abort_maniac_driver(self):
        """
        Scenario: 
        - Bus is 'Committed' (presence > 0.8). It SHOULD merge under normal rules.
        - However, a 'Maniac' car is approaching at 30 m/s (108 km/h).
        - Gap is small (15m). Crash is certain if Bus moves.
        
        Expected Result:
        - The Safety Override must trigger.
        - Bus must NOT switch to MOVING.
        - Bus physics must reflect braking/holding (velocity = 0).
        """
        print("\n--- Test: Emergency Abort (Maniac Driver) ---")

        # 1. Setup Bus: Fully committed (0.9 presence)
        # Normally, this bus would switch to MOVING immediately.
        bus = Bus(v_id=999, position=300.0, velocity=0.0)
        bus.state = BusState.WAITING_TO_MERGE
        bus.dwell_timer = 0.0
        bus.presence_factor = 0.90 
        
        # 2. Setup Maniac Car: 15m behind, moving at 30 m/s
        # TTC = 0.5 seconds. Extremely dangerous.
        car_maniac = Vehicle(v_id=1, position=285.0, velocity=30.0, params=CAR_PARAMS)
        
        self.sim.vehicles = [bus, car_maniac]
        
        # 3. Run a step
        print(f"   [Start] Bus State: {bus.state}, Presence: {bus.presence_factor:.2f}")
        self.sim.step()
        print(f"   [Step 1] Bus State: {bus.state}, Velocity: {bus.velocity:.2f}")
        
        # 4. Assertions
        # The state should STILL be WAITING_TO_MERGE.
        # If it switched to MOVING, the Safety Override failed.
        self.assertEqual(bus.state, BusState.WAITING_TO_MERGE, 
                         "FAILURE: Bus merged into a maniac driver!")
        
        # The physics should satisfy 'Case B' (Braking/Holding)
        # Since it started at 0, it should stay at 0.
        self.assertEqual(bus.velocity, 0.0, 
                         "FAILURE: Bus velocity increased despite emergency!")
        
        print("   [Pass] Bus correctly aborted merge due to high-speed danger.")

if __name__ == '__main__':
    unittest.main()