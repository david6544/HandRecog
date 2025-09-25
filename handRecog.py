#!/usr/bin/env python3
"""
Hand Recognition and Finger Joint Detection Script
This script uses MediaPipe and OpenCV to detect hands and track individual finger joints
through the webcamera in real-time.

Author: AI Assistant
Date: September 2025
"""

import cv2
import mediapipe as mp
import numpy as np
import time

class HandTracker:
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        """
        Initialize HandTracker with MediaPipe components
        
        Args:
            mode: Whether to treat input images as static images
            max_hands: Maximum number of hands to detect
            detection_con: Minimum detection confidence
            track_con: Minimum tracking confidence
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        
        # Initialize MediaPipe components
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Hand landmark labels for better understanding
        self.landmark_names = [
            'WRIST',
            'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
            'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
            'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
            'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
            'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
        ]
    
    def find_hands(self, img, draw=True):
        """
        Detect hands in the image
        
        Args:
            img: Input image
            draw: Whether to draw hand landmarks and connections
            
        Returns:
            img: Image with drawn landmarks (if draw=True)
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
        return img
    
    def find_position(self, img, hand_no=0, draw=True, draw_labels=False):
        """
        Find positions of hand landmarks
        
        Args:
            img: Input image
            hand_no: Which hand to track (0 for first detected hand)
            draw: Whether to draw landmark points
            draw_labels: Whether to draw landmark labels
            
        Returns:
            landmark_list: List of landmark positions [(id, x, y), ...]
        """
        landmark_list = []
        
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > hand_no:
                my_hand = self.results.multi_hand_landmarks[hand_no]
                
                for id, lm in enumerate(my_hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append([id, cx, cy])
                    
                    if draw:
                        # Draw landmark points with different colors for different finger parts
                        color = self.get_landmark_color(id)
                        cv2.circle(img, (cx, cy), 7, color, cv2.FILLED)
                        
                        if draw_labels and id < len(self.landmark_names):
                            # Draw landmark labels
                            cv2.putText(img, f"{id}:{self.landmark_names[id][:8]}", 
                                      (cx - 40, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.4, color, 1)
        
        return landmark_list
    
    def get_landmark_color(self, landmark_id):
        """
        Get color for different landmark types
        
        Args:
            landmark_id: ID of the landmark
            
        Returns:
            tuple: BGR color tuple
        """
        if landmark_id == 0:  # WRIST
            return (255, 255, 255)  # White
        elif 1 <= landmark_id <= 4:  # THUMB
            return (0, 255, 0)  # Green
        elif 5 <= landmark_id <= 8:  # INDEX FINGER
            return (255, 0, 0)  # Blue
        elif 9 <= landmark_id <= 12:  # MIDDLE FINGER
            return (0, 255, 255)  # Yellow
        elif 13 <= landmark_id <= 16:  # RING FINGER
            return (255, 0, 255)  # Magenta
        elif 17 <= landmark_id <= 20:  # PINKY
            return (0, 165, 255)  # Orange
        else:
            return (128, 128, 128)  # Gray
    
    def get_finger_status(self, landmark_list):
        """
        Determine which fingers are up or down
        
        Args:
            landmark_list: List of landmark positions
            
        Returns:
            fingers_up: List of boolean values for each finger [thumb, index, middle, ring, pinky]
        """
        fingers_up = []
        
        if len(landmark_list) >= 21:
            # Thumb (special case due to different orientation)
            if landmark_list[4][1] > landmark_list[3][1]:  # Thumb tip x > thumb ip x
                fingers_up.append(1)
            else:
                fingers_up.append(0)
            
            # Other fingers (index, middle, ring, pinky)
            finger_tips = [8, 12, 16, 20]
            finger_pips = [6, 10, 14, 18]
            
            for i in range(4):
                if landmark_list[finger_tips[i]][2] < landmark_list[finger_pips[i]][2]:  # tip y < pip y
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)
        
        return fingers_up
    
    def calculate_distances(self, landmark_list):
        """
        Calculate distances between key landmarks
        
        Args:
            landmark_list: List of landmark positions
            
        Returns:
            dict: Dictionary with distance measurements
        """
        distances = {}
        
        if len(landmark_list) >= 21:
            # Distance between thumb tip and index finger tip
            thumb_tip = landmark_list[4]
            index_tip = landmark_list[8]
            distances['thumb_index'] = np.sqrt(
                (thumb_tip[1] - index_tip[1])**2 + (thumb_tip[2] - index_tip[2])**2
            )
            
            # Distance between index and middle finger tips
            middle_tip = landmark_list[12]
            distances['index_middle'] = np.sqrt(
                (index_tip[1] - middle_tip[1])**2 + (index_tip[2] - middle_tip[2])**2
            )
            
            # Wrist to middle finger MCP (hand size reference)
            wrist = landmark_list[0]
            middle_mcp = landmark_list[9]
            distances['hand_size'] = np.sqrt(
                (wrist[1] - middle_mcp[1])**2 + (wrist[2] - middle_mcp[2])**2
            )
        
        return distances


def main():
    """
    Main function to run the hand tracking application
    """
    
    # Initialize webcam
    cap = cv2.VideoCapture(1)  # Change this index as needed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize hand tracker
    detector = HandTracker()
    
    # Variables for FPS calculation
    prev_time = 0
    
    print("Starting hand recognition...")
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 'l' to toggle landmark labels")
    print("- Press 's' to save screenshot")
    
    show_labels = False
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from camera")
            break
        
        # Flip image horizontally for mirror effect
        img = cv2.flip(img, 1)
        
        # Find hands
        img = detector.find_hands(img)
        landmark_list = detector.find_position(img, draw=True, draw_labels=show_labels)
        
        # Display information if hands are detected
        if len(landmark_list) != 0:
            # Get finger status
            fingers = detector.get_finger_status(landmark_list)
            finger_count = sum(fingers)
            
            # Calculate distances
            distances = detector.calculate_distances(landmark_list)
            
            # Display information on screen
            cv2.putText(img, f'Fingers Up: {finger_count}', (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.putText(img, f'Finger Status: {fingers}', (50, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display key distances
            y_offset = 130
            for name, distance in distances.items():
                cv2.putText(img, f'{name}: {distance:.1f}px', (50, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                y_offset += 30
            
            # Highlight specific landmarks
            if len(landmark_list) >= 21:
                # Draw connections between fingertips
                fingertips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
                for i in range(len(fingertips)-1):
                    pt1 = (landmark_list[fingertips[i]][1], landmark_list[fingertips[i]][2])
                    pt2 = (landmark_list[fingertips[i+1]][1], landmark_list[fingertips[i+1]][2])
                    cv2.line(img, pt1, pt2, (0, 255, 0), 2)
        
        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(img, f'FPS: {int(fps)}', (img.shape[1] - 150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display instructions
        cv2.putText(img, "Press 'q' to quit, 'l' for labels, 's' to save", 
                   (50, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display the image
        cv2.imshow('Hand Recognition - Finger Joint Detection', img)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            show_labels = not show_labels
            print(f"Landmark labels {'enabled' if show_labels else 'disabled'}")
        elif key == ord('s'):
            timestamp = int(time.time())
            filename = f'hand_capture_{timestamp}.jpg'
            cv2.imwrite(filename, img)
            print(f"Screenshot saved as {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Hand recognition stopped.")


if __name__ == "__main__":
    main()