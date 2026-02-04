from simulation import SimulationEngine

# Test Scenario A (Blocking)
print("--- Running Scenario A: Bus Stops in Lane ---")
sim = SimulationEngine(scenario="A")
sim.init_scenario()

# Run for 300 frames (30 seconds)
for _ in range(300):
    sim.step()

df = sim.get_results()

# Check: Does the bus actually stop?
bus_data = df[df['type'] == 'Bus']
print(f"Min Bus Speed: {bus_data['v'].min()} (Should be 0.0)")

# Check: Do cars stop behind it?
# Get the car immediately behind the bus
if len(df) > 0:
    cars = df[df['type'] == 'Car']
    print(f"Total cars spawned: {cars['id'].nunique()}")
    print("Snapshot at t=15s (Bus is stopped):")
    print(df[df['time'] == 15.0].sort_values('x', ascending=False).head(5))

print("\nDone. If you see velocities near 0 for cars behind the bus, Physics works.")