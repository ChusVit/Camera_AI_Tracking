# AI Camera Tracking System

A project implementing an AI-powered camera tracking system using Computer Vision and embedded hardware control.

## System Architecture (Pipeline)
The system is designed with a modular architecture consisting of 3 main layers:
1.  **Perception Layer**: Utilizes YOLOv8 for real-time human detection, extracting bounding box coordinates and the target's center.
2.  **Control Logic Layer**: Calculates the tracking error (offset from the screen center) and applies a PID controller to ensure smooth and stable camera movement.
3.  **Actuation Layer**: Handles Serial/Wireless communication to send positional commands to a microcontroller (ESP32/Yolo:Bit) to drive the Pan-Tilt Servo motors.

## Folder Structure
```text
Camera_Tracking_Project/
├── main.py                 # Main execution file (The Orchestrator)
├── config.py               # System configurations (PID constants, Camera ID, Ports)
├── requirements.txt        # Python dependencies list
├── vision/                 # Layer 1: AI Detection models (YOLOv8/MediaPipe)
├── control/                # Layer 2: Tracking logic & PID controller algorithms
├── comm/                   # Layer 3: Serial/WiFi communication bridge
└── mcu_code/               # Firmware for the microcontroller (MicroPython/C++)