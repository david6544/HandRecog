# Hand Recognition and Finger Joint Detection

This project provides a real-time hand recognition system that can detect hands and track individual finger joints through your webcamera using MediaPipe and OpenCV.

## Features

- **Real-time hand detection**: Detects up to 2 hands simultaneously
- **21-point hand landmarks**: Tracks all major hand joints and fingertips
- **Finger counting**: Automatically counts raised fingers
- **Distance measurements**: Calculates distances between key landmarks
- **Color-coded landmarks**: Different colors for different finger parts
- **Interactive controls**: Toggle labels, save screenshots
- **FPS monitoring**: Real-time performance display

## Installation

1. Make sure you're in the project directory:
   ```bash
   cd /Users/dmaslov/Desktop/Projects/Hand_Recog
   ```

2. The required packages should already be installed in the virtual environment:
   - opencv-python
   - mediapipe
   - numpy

## Usage

Run the main script:
```bash
/Users/dmaslov/Desktop/Projects/Hand_Recog/.venv/bin/python handRecog.py
```

### Controls

- **'q'**: Quit the application
- **'l'**: Toggle landmark labels on/off
- **'s'**: Save screenshot of current frame

## Hand Landmarks

The script tracks 21 landmarks per hand:

### Landmark IDs and Names:
- **0**: WRIST (white)
- **1-4**: THUMB joints (green)
  - 1: THUMB_CMC
  - 2: THUMB_MCP
  - 3: THUMB_IP
  - 4: THUMB_TIP
- **5-8**: INDEX FINGER joints (blue)
  - 5: INDEX_FINGER_MCP
  - 6: INDEX_FINGER_PIP
  - 7: INDEX_FINGER_DIP
  - 8: INDEX_FINGER_TIP
- **9-12**: MIDDLE FINGER joints (yellow)
- **13-16**: RING FINGER joints (magenta)
- **17-20**: PINKY joints (orange)

## Features Explanation

### Finger Counting
The script automatically detects which fingers are extended and displays:
- Total count of raised fingers
- Status array showing which fingers are up [thumb, index, middle, ring, pinky]

### Distance Measurements
Calculates key distances:
- **thumb_index**: Distance between thumb tip and index finger tip
- **index_middle**: Distance between index and middle finger tips  
- **hand_size**: Reference measurement from wrist to middle finger base

### Visual Feedback
- Hand landmarks are drawn with connections
- Different colors for different finger parts
- Optional landmark labels with IDs and names
- Fingertip connections are highlighted in green
- Real-time FPS display

## Troubleshooting

### Camera Issues
- Ensure your camera is not being used by another application
- Check camera permissions in System Preferences > Security & Privacy > Camera
- Try changing the camera index in the code if you have multiple cameras

### Import Errors
If you get import errors, ensure the virtual environment is activated and packages are installed:
```bash
/Users/dmaslov/Desktop/Projects/Hand_Recog/.venv/bin/python -m pip install opencv-python mediapipe numpy
```

### Performance Issues
- Reduce camera resolution in the code if needed
- Close other applications using the camera
- Ensure good lighting conditions for better detection

## Customization

You can modify the `HandTracker` class parameters:
- `max_hands`: Maximum number of hands to detect (default: 2)
- `detection_con`: Minimum detection confidence (default: 0.5)
- `track_con`: Minimum tracking confidence (default: 0.5)

## Technical Details

- Uses MediaPipe Hands solution for robust hand detection
- Processes video at real-time frame rates
- Implements gesture recognition through landmark analysis
- Provides extensible framework for hand-based applications

## Applications

This foundation can be extended for:
- Sign language recognition
- Gesture-based controls
- Hand measurement applications  
- Interactive interfaces
- Accessibility tools