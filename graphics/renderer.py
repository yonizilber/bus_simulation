import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import matplotlib.transforms as transforms


class SimulationRenderer:
    def __init__(self, dataframe: pd.DataFrame, config):
        self.df = dataframe
        self.cfg = config
        self.road_length = self.cfg.road_length
        self.bay_start = self.cfg.bus_stop_position - 15.0
        self.bay_end = self.cfg.bus_stop_position + 15.0
        
        self.lane_width = 3.5 # REAL LANE WIDTH
        self.y_center = 0.0          
        self.y_bay = -self.lane_width 

        if 'width' in self.df.columns:
            car_rows = self.df[self.df['type'] == 'Car']
            self.reference_car_width = float(car_rows['width'].median()) if not car_rows.empty else 1.9
        else:
            self.reference_car_width = 1.9

    def draw_frame(self, ax, current_time: float, window_size=150.0):
        # Clear the old frame
        ax.clear()
        
        # --- FIXED CAMERA LOGIC ---
        # Center the camera exactly on the bus stop (+/- 75m)
        camera_x = self.cfg.bus_stop_position
        ax.set_xlim(camera_x - (window_size / 2), camera_x + (window_size / 2))
        ax.set_ylim(-6.0, 4.0) 
        ax.set_aspect('equal') # <--- ADD THIS LINE HERE
        ax.axis('off')
        ax.set_facecolor('#e6e6e6')
        
        # Draw Environment
        self._draw_road(ax, camera_x, window_size)

        # Draw Vehicles
        frame_data = self.df[self.df['time'] == current_time]
        for _, row in frame_data.iterrows():
            self._draw_vehicle(ax, row)

    def _draw_road(self, ax, camera_x, window_size):
        main_road = patches.Rectangle(
            (0, -self.lane_width/2), self.road_length, self.lane_width, 
            facecolor='#555555', zorder=0
        )
        ax.add_patch(main_road)

        ax.plot([0, self.road_length], [self.lane_width/2, self.lane_width/2], color='white', linestyle='--', linewidth=2, zorder=1)
        ax.plot([0, self.road_length], [-self.lane_width/2, -self.lane_width/2], color='white', linestyle='-', linewidth=2, zorder=1)

        bay_length = self.bay_end - self.bay_start
        bus_bay = patches.Rectangle(
            (self.bay_start, self.y_bay - self.lane_width/2), 
            bay_length, self.lane_width, 
            facecolor='#444444', zorder=0
        )
        ax.add_patch(bus_bay)
        
        taper_in = patches.Polygon([[self.bay_start - 10, -self.lane_width/2], [self.bay_start, -self.lane_width/2], [self.bay_start, self.y_bay - self.lane_width/2]], facecolor='#444444', zorder=0)
        taper_out = patches.Polygon([[self.bay_end, self.y_bay - self.lane_width/2], [self.bay_end, -self.lane_width/2], [self.bay_end + 10, -self.lane_width/2]], facecolor='#444444', zorder=0)
        ax.add_patch(taper_in)
        ax.add_patch(taper_out)

    def _draw_vehicle(self, ax, row):
        
        v_type = row['type']
        x_pos = row['x']
        presence = float(row['presence'])
        state_str = str(row['state'])
        
        length = float(row.get('length', 11.0 if v_type == 'Bus' else 4.5))
        width = float(row.get('width', 2.5 if v_type == 'Bus' else 1.9))
        
        # Determine Y Position
        if v_type == 'Bus':
            y_pos = self.y_bay + (presence * (self.y_center - self.y_bay))
        else:
            y_pos = self.y_center + 1.5 if state_str == 'Squeezing!' else self.y_center
                
        # Determine Color
        color = '#3498db' 
        if v_type == 'Bus':
            if state_str == 'IN_BAY': color = '#333333'
            elif state_str == 'WAITING_TO_MERGE':
                space_left = 3.5 - (presence * 2.5)
                space_needed = self.reference_car_width + 0.5
                color = '#FFD700' if space_left >= space_needed else '#FF0000'
            else:
                color = '#e74c3c' 

        # --- NEW: CALCULATE ROTATION ANGLE (SMOOTH SINE WAVE) ---
        angle_degrees = 0.0
        
        if v_type == 'Bus':
            if state_str == 'DECELERATING' and presence < 1.0:
                # Max angle for entering (approx -10 degrees)
                max_angle = np.degrees(np.arctan2(-self.lane_width, 20.0))
                # Smooth ease-in, ease-out steering
                angle_degrees = np.sin((1.0 - presence) * np.pi) * max_angle
                
            elif state_str == 'WAITING_TO_MERGE' and presence > 0.0:
                # Max angle for exiting (approx +13 degrees)
                max_angle = np.degrees(np.arctan2(self.lane_width, 15.0))
                # Smooth ease-in, ease-out steering
                angle_degrees = np.sin(presence * np.pi) * max_angle

        # --- DRAW WITH ROTATION ---
        # We use a transform to rotate the rectangle around its center
        
        rect = patches.Rectangle(
            (x_pos - length/2, y_pos - width/2), # Shift to center for proper rotation
            length, width, 
            facecolor=color, edgecolor='black', linewidth=1.5, zorder=5, clip_on=True
        )
        
        # Apply the rotation
        t = transforms.Affine2D().rotate_deg_around(x_pos, y_pos, angle_degrees) + ax.transData
        rect.set_transform(t)
        
        ax.add_patch(rect)
        
        # ID Text
        ax.text(x_pos, y_pos, str(int(row['id'])), 
                color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=6, clip_on=True)