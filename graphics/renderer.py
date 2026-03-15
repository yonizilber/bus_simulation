import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

class SimulationRenderer:
    def __init__(self, dataframe: pd.DataFrame, config):
        self.df = dataframe
        self.cfg = config
        self.road_length = self.cfg.road_length
        self.bay_start = self.cfg.bus_stop_position - 15.0
        self.bay_end = self.cfg.bus_stop_position + 15.0
        
        self.lane_width = 4.0
        self.y_center = 0.0          
        self.y_bay = -self.lane_width 

    def draw_frame(self, ax, current_time: float, window_size=150.0):
        # Clear the old frame
        ax.clear()
        
        # --- FIXED CAMERA LOGIC ---
        # Center the camera exactly on the bus stop (+/- 75m)
        camera_x = self.cfg.bus_stop_position
        ax.set_xlim(camera_x - (window_size / 2), camera_x + (window_size / 2))
        ax.set_ylim(-6.0, 4.0) 
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
        
        length = 10.0 if v_type == 'Bus' else 5.0
        width = 2.5 if v_type == 'Bus' else 2.0
        
        if v_type == 'Bus':
            y_pos = self.y_bay + (presence * (self.y_center - self.y_bay))
        else:
            if state_str == 'Squeezing!':
                y_pos = self.y_center + 1.5 
            else:
                y_pos = self.y_center
                
        color = '#3498db' 
        if v_type == 'Bus':
            if state_str == 'IN_BAY': color = '#333333'
            elif state_str == 'WAITING_TO_MERGE':
                color = '#FFD700' if presence <= 0.85 else '#FF0000'
            else:
                color = '#e74c3c' 

        # clip_on=True ensures the camera window doesn't resize when cars drive out of view
        rect = patches.Rectangle(
            (x_pos - length, y_pos - width/2), 
            length, width, 
            facecolor=color, edgecolor='black', linewidth=1.5, zorder=5, clip_on=True
        )
        ax.add_patch(rect)
        
        ax.text(x_pos - length/2, y_pos, str(int(row['id'])), 
                color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=6, clip_on=True)