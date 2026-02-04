import streamlit as st
import json

def traffic_animation(scenario_type, stop_time, traffic_rate, merge_time):
    """
    Advanced JS Animation with Smooth Physics and Visual Details.
    """
    
    # 60 frames = 1 second.
    spawn_frames = max(30, int(3600 / traffic_rate))

    sim_config = json.dumps({
        "scenario": scenario_type,
        "stopDuration": stop_time * 5,
        "spawnRate": spawn_frames,
        "mergeDuration": merge_time * 5
    })

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            canvas {{ 
                border: 1px solid #333; 
                background: #888; 
                border-radius: 4px; 
                width: 100%; 
            }}
            .container {{ position: relative; width: 100%; }}
            .status {{ 
                color: #333; 
                font-family: sans-serif; 
                margin-top: 5px; 
                font-size: 12px; 
                background: #eee;
                padding: 4px 8px;
                border-radius: 4px;
                display: inline-block;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <canvas id="simCanvas_{scenario_type}" width="800" height="150"></canvas>
            <div id="status_{scenario_type}" class="status">Initializing...</div>
        </div>

        <script>
            (function() {{
                const canvas = document.getElementById('simCanvas_{scenario_type}');
                const ctx = canvas.getContext('2d');
                const statusDiv = document.getElementById('status_{scenario_type}');
                const config = {sim_config};
                
                // --- GEOMETRY ---
                const ROAD_Y = 80;
                const ROAD_HEIGHT = 40;
                const BAY_Y = 35; 
                const BAY_X_START = 300;
                const BAY_WIDTH = 150;
                
                const STOP_SIGN_X = BAY_X_START + 120; // Sign near end of bay/stop
                
                // --- PHYSICS CONSTANTS ---
                const MAX_SPEED = 4.0;
                const ACCEL = 0.1;
                const BRAKE = 0.2;
                const SAFETY_GAP = 20;     // Min distance to stop
                const LOOKAHEAD = 150;     // Distance to start slowing down
                
                // --- STATE ---
                let cars = [];
                let bus = {{ 
                    x: -150, 
                    y: ROAD_Y + (ROAD_HEIGHT - 22)/2, 
                    speed: MAX_SPEED * 0.8,
                    state: 'moving', 
                    timer: 0 
                }};
                
                let frame = 0;
                let carsStuckCount = 0;

                function reset() {{
                    cars = [];
                    bus.x = -150;
                    bus.y = ROAD_Y + (ROAD_HEIGHT - 22)/2;
                    bus.state = 'moving';
                    bus.speed = MAX_SPEED * 0.8;
                    frame = 0;
                }}

                function update() {{
                    frame++;
                    carsStuckCount = 0;

                    // 1. Spawn Logic
                    let entranceClear = true;
                    if (bus.x < 60) entranceClear = false;
                    for (let c of cars) {{ if (c.x < 60) entranceClear = false; }}
                    
                    if (frame % config.spawnRate === 0 && entranceClear) {{
                        cars.push({{ 
                            x: -40, 
                            y: ROAD_Y + (ROAD_HEIGHT - 18)/2, 
                            speed: MAX_SPEED, 
                            currentSpeed: MAX_SPEED 
                        }});
                    }}

                    updateBus();
                    updateCars();
                    draw();
                    requestAnimationFrame(update);
                }}

                function updateBus() {{
                    const stopTargetX = BAY_X_START + 20;

                    if (bus.state === 'moving') {{
                        bus.x += bus.speed;
                        // Decelerate if nearing stop
                        if (bus.x > stopTargetX - 100 && bus.x < stopTargetX) {{
                             bus.x += (2 - bus.speed) * 0.1; // Slow approach
                        }}
                        
                        if (bus.x >= stopTargetX) {{
                            if (config.scenario === 'bay') {{
                                bus.state = 'pulling_in';
                            }} else {{
                                bus.state = 'dwelling';
                                bus.timer = config.stopDuration;
                            }}
                        }}
                    }}
                    else if (bus.state === 'pulling_in') {{
                        bus.x += 1.5;
                        bus.y -= 1.5; // Slide up
                        if (bus.y <= BAY_Y + 5) {{
                            bus.y = BAY_Y + 5;
                            bus.state = 'dwelling';
                            bus.timer = config.stopDuration;
                        }}
                    }}
                    else if (bus.state === 'dwelling') {{
                        bus.timer--;
                        if (bus.timer <= 0) {{
                            if (config.scenario === 'bay') {{
                                bus.state = 'merging';
                                bus.timer = config.mergeDuration;
                            }} else {{
                                bus.state = 'leaving';
                            }}
                        }}
                    }}
                    else if (bus.state === 'merging') {{
                        bus.timer--;
                        // Simple logic: wait for timer, then try to merge
                        if (bus.timer <= 0) bus.state = 'pulling_out';
                    }}
                    else if (bus.state === 'pulling_out') {{
                        bus.x += 1.5;
                        bus.y += 1.5; // Slide down
                        if (bus.y >= ROAD_Y + (ROAD_HEIGHT - 22)/2) {{
                            bus.y = ROAD_Y + (ROAD_HEIGHT - 22)/2;
                            bus.state = 'leaving';
                        }}
                    }}
                    else if (bus.state === 'leaving') {{
                        bus.x += bus.speed;
                        if (bus.x > 850) reset();
                    }}
                }}

                function updateCars() {{
                    for (let i = 0; i < cars.length; i++) {{
                        let car = cars[i];
                        let obstacleDist = 9999;
                        let obstacleSpeed = MAX_SPEED;

                        // A. Check Car Ahead
                        if (i > 0) {{
                            obstacleDist = cars[i-1].x - car.x - 35; // 35 is car width + buffer
                            obstacleSpeed = cars[i-1].currentSpeed;
                        }}

                        // B. Check Bus (Only if blocking)
                        let busBlocking = false;
                        if (config.scenario === 'lane' && bus.state !== 'leaving' && bus.state !== 'moving') busBlocking = true;
                        if (config.scenario === 'bay' && bus.state === 'pulling_out') busBlocking = true;
                        // Also consider bus if it's just driving normally ahead of us
                        if (bus.state === 'moving' || bus.state === 'leaving') busBlocking = true;

                        if (busBlocking && bus.x > car.x) {{
                            let distToBus = bus.x - car.x - 35;
                            if (distToBus < obstacleDist) {{
                                obstacleDist = distToBus;
                                obstacleSpeed = (bus.state === 'moving' || bus.state === 'leaving') ? bus.speed : 0;
                            }}
                        }}

                        // --- PHYSICS: Simple Car Following Logic ---
                        let targetSpeed = MAX_SPEED;

                        if (obstacleDist < SAFETY_GAP) {{
                            targetSpeed = 0; // Emergency stop
                        }} else if (obstacleDist < LOOKAHEAD) {{
                            // Gradual Slowdown: Match obstacle speed or go slower if close
                            let factor = obstacleDist / LOOKAHEAD;
                            targetSpeed = Math.min(MAX_SPEED * factor, obstacleSpeed); 
                        }}

                        // Apply Acceleration / Braking
                        if (car.currentSpeed < targetSpeed) {{
                            car.currentSpeed += ACCEL;
                        }} else if (car.currentSpeed > targetSpeed) {{
                            car.currentSpeed -= BRAKE;
                        }}
                        
                        // Sanity Check
                        if (car.currentSpeed < 0.1) car.currentSpeed = 0;
                        
                        car.x += car.currentSpeed;
                        
                        if (car.currentSpeed < 0.1 && obstacleDist < LOOKAHEAD) carsStuckCount++;
                    }}
                    
                    cars = cars.filter(c => c.x < 900);
                }}

                function draw() {{
                    // Background
                    ctx.fillStyle = '#999'; // Lighter gray background
                    ctx.fillRect(0, 0, canvas.width, canvas.height);

                    // --- DRAW BUS STOP SIGN ---
                    // Pole
                    ctx.fillStyle = '#333';
                    ctx.fillRect(STOP_SIGN_X, BAY_Y - 20, 4, 60); 
                    // Sign Board
                    ctx.fillStyle = '#0033cc'; // Israel Bus Blue
                    ctx.fillRect(STOP_SIGN_X - 10, BAY_Y - 20, 24, 24);
                    // Icon
                    ctx.fillStyle = 'white';
                    ctx.font = '16px Arial';
                    ctx.fillText('🚌', STOP_SIGN_X - 9, BAY_Y - 2);

                    // --- DRAW ROAD ---
                    ctx.fillStyle = '#333'; // Dark Asphalt
                    ctx.fillRect(0, ROAD_Y, canvas.width, ROAD_HEIGHT);
                    
                    // Road Markings
                    ctx.strokeStyle = '#fff';
                    ctx.setLineDash([20, 20]);
                    ctx.beginPath();
                    ctx.moveTo(0, ROAD_Y + ROAD_HEIGHT/2);
                    ctx.lineTo(canvas.width, ROAD_Y + ROAD_HEIGHT/2);
                    ctx.stroke();
                    ctx.setLineDash([]); // Reset

                    // --- DRAW BUS BAY (Cutout) ---
                    if (config.scenario === 'bay') {{
                        // Asphalt extension
                        ctx.fillRect(BAY_X_START, BAY_Y, BAY_WIDTH, ROAD_Y - BAY_Y + 5);
                        
                        // Yellow Box Marking
                        ctx.strokeStyle = '#fdd835';
                        ctx.lineWidth = 2;
                        ctx.strokeRect(BAY_X_START + 10, BAY_Y + 5, BAY_WIDTH - 20, BUS_HEIGHT = 22);
                    }}

                    // --- DRAW BUS ---
                    let busColor = '#4CAF50';
                    if (bus.state === 'merging') busColor = '#ff9800'; // Orange
                    if (bus.state === 'dwelling' && config.scenario === 'lane') busColor = '#f44336'; // Red
                    
                    ctx.fillStyle = busColor;
                    ctx.fillRect(bus.x, bus.y, 70, 22);
                    
                    // Bus Lights
                    if (bus.currentSpeed < 0.1 || bus.state === 'dwelling') {{
                         ctx.fillStyle = '#ff0000'; // Brake lights
                         ctx.fillRect(bus.x, bus.y, 4, 22);
                    }}

                    // --- DRAW CARS ---
                    for (let car of cars) {{
                        // Car Body
                        ctx.fillStyle = car.currentSpeed < 0.5 ? '#e53935' : '#1e88e5'; // Red if stopped, Blue if moving
                        ctx.fillRect(car.x, car.y, 30, 18);
                        
                        // Brake Lights visual
                        if (car.currentSpeed < MAX_SPEED * 0.5) {{
                             ctx.fillStyle = '#ff0000';
                             ctx.fillRect(car.x, car.y, 3, 18);
                        }}
                    }}
                    
                    // UI
                    statusDiv.innerHTML = `Bus: <b>${{bus.state.toUpperCase()}}</b> | Traffic Speed: <b>${{cars.length > 0 ? cars[0].currentSpeed.toFixed(1) : 0}}</b>`;
                }}

                update();
            }})();
        </script>
    </body>
    </html>
    """
    st.components.v1.html(html_code, height=200)