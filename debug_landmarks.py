import pandas as pd
import numpy as np

# Quick diagnostic script to check landmark positions
# Load a specific frame's landmarks and print coordinates

# This would be used to verify what the landmark detector is actually seeing
# Usage: python debug_landmarks.py <path_to_landmarks_csv> <frame_number>

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python debug_landmarks.py <landmarks_csv> <frame_idx>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    frame_idx = int(sys.argv[2])
    
    df = pd.read_csv(csv_path)
    frame_data = df[df['frame_idx'] == frame_idx]
    
    if frame_data.empty:
        print(f"No data for frame {frame_idx}")
        sys.exit(1)
    
    row = frame_data.iloc[0]
    
    # Extract relevant landmarks
    l_hip_x, l_hip_y = row.get('left_hip_x'), row.get('left_hip_y')
    r_hip_x, r_hip_y = row.get('right_hip_x'), row.get('right_hip_y')
    l_ank_x, l_ank_y = row.get('left_ankle_x'), row.get('left_ankle_y')
    r_ank_x, r_ank_y = row.get('right_ankle_x'), row.get('right_ankle_y')
    
    pelvis_x = (l_hip_x + r_hip_x) / 2
    pelvis_y = (l_hip_y + r_hip_y) / 2
    
    print(f"\n=== Frame {frame_idx} Landmarks ===")
    print(f"Left Hip:      ({l_hip_x:.2f}, {l_hip_y:.2f})")
    print(f"Right Hip:     ({r_hip_x:.2f}, {r_hip_y:.2f})")
    print(f"Pelvis Center: ({pelvis_x:.2f}, {pelvis_y:.2f})")
    print(f"Left Ankle:    ({l_ank_x:.2f}, {l_ank_y:.2f})")
    print(f"Right Ankle:   ({r_ank_x:.2f}, {r_ank_y:.2f})")
    
    # Compute angles
    dx_left = l_ank_x - pelvis_x
    dy_left = l_ank_y - pelvis_y
    angle_left = np.degrees(np.arctan2(dy_left, dx_left))
    
    dx_right = r_ank_x - pelvis_x
    dy_right = r_ank_y - pelvis_y
    angle_right = np.degrees(np.arctan2(dy_right, dx_right))
    
    separation = abs(angle_right - angle_left)
    if separation > 180:
        separation = 360 - separation
    
    print(f"\n=== Angular Positions ===")
    print(f"Left ankle angle:  {angle_left:.2f}°")
    print(f"Right ankle angle: {angle_right:.2f}°")
    print(f"Separation:        {separation:.2f}°")
