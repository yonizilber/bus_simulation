import unittest
from vehicle import Vehicle, Bus, BusState, CAR_PARAMS, VehicleParams
from simulation import SimulationEngine
from config import SimConfig

# Define standard parameters for the test
TEST_PARAMS = VehicleParams(
    max_speed=20.0, max_accel=2.0, comfortable_brake=2.0,
    min_gap=2.0, time_headway=1.0, length=5.0
)

class TestBusMerging(unittest.TestCase):
    
    def setUp(self):
        # Initialize a clean engine before each test
        self.cfg = SimConfig()
        self.sim = SimulationEngine(scenario="B", config=self.cfg)
        self.sim.vehicles = []

    def test_unsafe_merge_blocked_by_car(self):
        """
        Scenario: Bus is in the bay, waiting to merge.
        A car is right behind it in the main lane, moving fast.
        Expected Result: The bus MUST stay in WAITING_TO_MERGE state.
        """
        print("\n--- Test: Unsafe Merge (Blocked) ---")

        # 1. Place the Bus in the Bay, ready to merge
        bus = Bus(v_id=999, position=300.0, velocity=0.0)
        bus.state = BusState.WAITING_TO_MERGE
        # Important: Ensure dwell timer is zero so it *tries* to merge
        bus.dwell_timer = 0.0 
        
        # 2. Place a Car immediately behind it, moving fast
        # Bus rear is at 300 - 12 = 288m.
        # Car front at 285m. Gap is 3 meters. Speed diff is high.
        # This is DANGEROUS.
        car_behind = Vehicle(v_id=1, position=285.0, velocity=15.0, params=TEST_PARAMS)
        
        self.sim.vehicles = [bus, car_behind]
        
        # 3. Run one simulation step
        # The engine sorts vehicles: Bus (300) is index 0, Car (285) is index 1.
        self.sim.step()
        
        print(f"   [Result] Bus State: {bus.state}")
        print(f"   [Result] Bus Velocity: {bus.velocity}")

        # 4. Assertions
        self.assertEqual(bus.state, BusState.WAITING_TO_MERGE, 
                         "FAILURE: Bus merged into oncoming traffic!")
        self.assertEqual(bus.velocity, 0.0, 
                         "FAILURE: Bus started moving while blocked!")
        print("   [Pass] Bus correctly waited for the gap.")

    def test_safe_merge_clear_road(self):
        """
        Scenario: Bus is waiting to merge. The road behind is clear.
        Expected Result: The bus transitions to MOVING state.
        """
        print("\n--- Test: Safe Merge (Clear Road) ---")

        # 1. Bus waiting
        bus = Bus(v_id=999, position=300.0, velocity=0.0)
        bus.state = BusState.WAITING_TO_MERGE
        bus.dwell_timer = 0.0
        
        # 2. Car is far behind (100m back)
        car_behind = Vehicle(v_id=1, position=200.0, velocity=15.0, params=TEST_PARAMS)
        
        self.sim.vehicles = [bus, car_behind]
        
        # 3. Run step
        self.sim.step()
        
        print(f"   [Result] Bus State: {bus.state}")
        
        # 4. Assertions
        self.assertEqual(bus.state, BusState.MOVING, 
                         "FAILURE: Bus did not merge when the road was clear.")
        print("   [Pass] Bus successfully merged.")

if __name__ == '__main__':
    unittest.main(verbosity=0)