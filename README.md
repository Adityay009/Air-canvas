A real-time AI-powered virtual drawing application built using OpenCV and MediaPipe that allows users to draw on screen using hand gestures.
This project uses computer vision to track hand landmarks and enables drawing, erasing, clearing, and saving artwork without touching the screen.

Features:

•✋ Real-time hand tracking using MediaPipe
•🎨 Draw using index finger
•🧽 Eraser mode
•🗑 Clear canvas button
•💾 Save drawing as PNG
•🎚 Adjustable brush thickness (slider control)
•🎯 Modern dark toolbar UI
•📊 FPS counter display
•⌨ Press Q to exit

Technologies Used

•Python 3.10
•OpenCV
•MediaPipe
•NumPy

Installation Guide:

1. Clone the Repository
git clone https://github.com/your-username/Air_Canvas-main.git
cd Air_Canvas-main

2. Create Virtual Environment
python3.10 -m venv venv
source venv/bin/activate

3. Install Dependencies
pip install mediapipe==0.10.8 opencv-python numpy

4. Run the application
python air_canvas_hand.py

How To Use:

Gesture                                                    Action
Index + Middle Finger Up                                   Select toolbar button
Only Index Finger Up                                       Draw
ERASER                                                     Switch to erase mode
CLEAR                                                      Clear entire canvas
SAVE                                                       Save drawing
Press Q                                                    Exit application


 How It Works

1. MediaPipe detects hand landmarks in real-time.
2. Index finger tip coordinates are tracked.
3. Based on finger gestures:
	• Drawing mode is activated
	• Toolbar buttons are selected
4. OpenCV overlays strokes onto a virtual canvas.
5. Canvas is merged with live webcam feed.