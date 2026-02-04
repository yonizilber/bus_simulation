import unittest
from vehicle import Vehicle, Bus, BusState, CAR_PARAMS, VehicleParams
from simulation import SimulationEngine
from config import SimConfig

class TestMergeLogic(unittest.TestCase):
    
    def setUp(self):
        self.cfg = SimConfig()
        self.sim = SimulationEngine(scenario="B", config=self.cfg)
        self.sim.vehicles = []

    def test_yield_recognition_and_commitment(self):
        """
        Scenario: 
        - Bus is waiting to merge. It has 'nudged' out significantly (presence=0.7).
        - Car behind is moving SLOWLY (yielding).
        - Gap is small (safe for yielding, unsafe for full speed).
        
        Expected Result:
        - Bus should recognize the yield.
        - Bus should increase presence to > 0.8 (Commitment).
        - Bus should switch to MOVING.
        """
        print("\n--- Test: Yield Recognition & Commitment ---")

        # 1. Setup Bus: Waiting, mostly out (0.75 presence)
        bus = Bus(v_id=999, position=300.0, velocity=0.0)
        bus.state = BusState.WAITING_TO_MERGE
        bus.dwell_timer = 0.0
        bus.presence_factor = 0.75 
        
        # 2. Setup Car: Close behind (10m gap), but Slow (yielding)
        # 4 m/s is slow enough to trigger our new "Yield" logic
        car_behind = Vehicle(v_id=1, position=285.0, velocity=4.0, params=CAR_PARAMS)
        
        self.sim.vehicles = [bus, car_behind]
        
        # 3. Run a few frames to let logic evolve
        # Frame 1: Presence 0.75 -> 0.80. Check yield.
        # Frame 2: Presence 0.80 -> 0.85 (Committed!). State -> MOVING.
        print(f"   [Start] Bus State: {bus.state}, Presence: {bus.presence_factor:.2f}")
        
        for i in range(5):
            self.sim.step()
            print(f"   [Step {i+1}] Bus State: {bus.state}, Presence: {bus.presence_factor:.2f}, Velocity: {bus.velocity:.2f}")
            
            # Stop if merged to avoid running off map
            if bus.state == BusState.MOVING:
                break

        # 4. Assertions
        self.assertEqual(bus.state, BusState.MOVING, 
                         "FAILURE: Bus failed to commit/merge despite yielding car.")
        self.assertGreater(bus.velocity, 0.1, 
                           "FAILURE: Bus state is MOVING but velocity is 0.")
        print("   [Pass] Bus recognized yield and committed to the merge.")

if __name__ == '__main__':
    unittest.main()