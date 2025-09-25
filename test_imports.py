#!/usr/bin/env python3
"""
Test script to verify that all required packages are properly installed
"""

print("Testing imports...")

try:
    import cv2
    print("✓ OpenCV imported successfully")
    print(f"  OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV import failed: {e}")

try:
    import mediapipe as mp
    print("✓ MediaPipe imported successfully")
    print(f"  MediaPipe version: {mp.__version__}")
except ImportError as e:
    print(f"✗ MediaPipe import failed: {e}")

try:
    import numpy as np
    print("✓ NumPy imported successfully")
    print(f"  NumPy version: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")

# Test camera access
print("\nTesting camera access...")
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✓ Camera accessible, frame shape: {frame.shape}")
        else:
            print("✗ Camera opened but couldn't read frame")
        cap.release()
    else:
        print("✗ Could not open camera")
except Exception as e:
    print(f"✗ Camera test failed: {e}")

print("\nTest completed!")