import numpy as np
import pandas as pd

# Test the angle calculation with known vectors
def test_angle_calculation():
    """Test with simple known angles"""
    
    # Test case 1: 90 degree angle
    # Left ankle at (-1, -1), Right ankle at (1, -1), Pelvis at (0, 0)
    v_left = np.array([-1, -1])
    v_right = np.array([1, -1])
    
    dot = np.dot(v_left, v_right)
    nm_l = np.linalg.norm(v_left)
    nm_r = np.linalg.norm(v_right)
    cosang = dot / (nm_l * nm_r)
    angle = np.degrees(np.arccos(cosang))
    
    print(f"Test 1 - 90° angle:")
    print(f"  v_left: {v_left}, v_right: {v_right}")
    print(f"  dot: {dot}, nm_l: {nm_l}, nm_r: {nm_r}")
    print(f"  cosang: {cosang}, angle: {angle}°")
    print()
    
    # Test case 2: Wide straddle - 120 degrees
    # Left at (-1, -0.577), Right at (1, -0.577), Pelvis at (0, 0)
    # This forms 120° angle
    v_left = np.array([-1, -0.577])
    v_right = np.array([1, -0.577])
    
    dot = np.dot(v_left, v_right)
    nm_l = np.linalg.norm(v_left)
    nm_r = np.linalg.norm(v_right)
    cosang = dot / (nm_l * nm_r)
    angle = np.degrees(np.arccos(cosang))
    
    print(f"Test 2 - 120° angle:")
    print(f"  v_left: {v_left}, v_right: {v_right}")
    print(f"  dot: {dot}, nm_l: {nm_l}, nm_r: {nm_r}")
    print(f"  cosang: {cosang}, angle: {angle}°")
    print()
    
    # Test case 3: Legs together - 20 degrees
    # Small angle
    v_left = np.array([-0.174, -1])  # ~10° from vertical
    v_right = np.array([0.174, -1])   # ~10° from vertical
    
    dot = np.dot(v_left, v_right)
    nm_l = np.linalg.norm(v_left)
    nm_r = np.linalg.norm(v_right)
    cosang = dot / (nm_l * nm_r)
    angle = np.degrees(np.arccos(cosang))
    
    print(f"Test 3 - ~20° angle (legs together):")
    print(f"  v_left: {v_left}, v_right: {v_right}")
    print(f"  dot: {dot}, nm_l: {nm_l}, nm_r: {nm_r}")
    print(f"  cosang: {cosang}, angle: {angle}°")

if __name__ == "__main__":
    test_angle_calculation()
